//! One TESSERA stage: the 9-pillar pipeline, streaming (one token at a time),
//! with per-step caches for exact analytical backward.

use crate::attention::{AttnCache, DiffAttn};
use crate::conv::HwyConv;
use crate::ffn::SwiGLU;
use crate::mrm::{Mrm, MrmCfg, MrmGrads, MrmSnapshot};
use crate::norms::{rmsnorm_bwd, rmsnorm_fwd, safe_inv_norm};
use crate::num::Num;

pub struct StageCache<T: Num> {
    pub n1: Vec<T>, // conv input = rmsnorm1 output
    pub xhat1: Vec<T>,
    pub s1: T,
    pub c: Vec<T>, // conv output
    pub attn: Option<AttnCache<T>>,
    pub out64: Vec<T>,
    pub out_mix: Vec<T>,
    pub xhat2: Vec<T>,
    pub s2: T,
    pub lora_a: Vec<T>,
    pub hn: Vec<T>, // safe-normalized h (write path input)
    pub hnorm: T,
    pub k_raw: Vec<T>,
    pub v_raw: Vec<T>,
    pub glin: T,
    pub snap: Option<MrmSnapshot<T>>,
}

pub struct Stage<T: Num> {
    pub idx: usize, // 0-based stage index
    pub d: usize,
    pub m: usize,
    pub g1: Vec<T>,
    pub dg1: Vec<T>,
    pub conv: HwyConv<T>,
    pub conv_carry: Vec<T>,
    pub zero_row: Vec<T>,
    pub attn: DiffAttn<T>,
    pub vres_g: Vec<T>, // len 1, layers > 0
    pub dvres_g: Vec<T>,
    pub wo: Vec<T>, // [m][d]
    pub dwo: Vec<T>,
    pub g2: Vec<T>,
    pub dg2: Vec<T>,
    pub ffn: SwiGLU<T>,
    pub wkw: Vec<T>, // [d][m]
    pub dwkw: Vec<T>,
    pub wvw: Vec<T>,
    pub dwvw: Vec<T>,
    pub gu: Vec<T>,
    pub dgu: Vec<T>,
    pub gb: Vec<T>, // len 1
    pub dgb: Vec<T>,
    pub mem: Mrm<T>,
    pub mgrads: MrmGrads<T>,
    pub cache: Vec<StageCache<T>>,
}

impl<T: Num> Stage<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(idx: usize, d: usize, heads: usize, dh: usize, ff: usize, lora_r: usize, k_conv: usize, mcfg: MrmCfg, n_read: usize, rope_base: f64) -> Self {
        let m = mcfg.d;
        Self {
            idx,
            d,
            m,
            g1: vec![T::ONE; d],
            dg1: vec![T::ZERO; d],
            conv: HwyConv::new(k_conv, d),
            conv_carry: vec![T::ZERO; (k_conv - 1) * d],
            zero_row: vec![T::ZERO; d],
            attn: DiffAttn::new(d, heads, dh, idx + 1, n_read, rope_base),
            vres_g: vec![T::ZERO; 1],
            dvres_g: vec![T::ZERO; 1],
            wo: vec![T::ZERO; m * d],
            dwo: vec![T::ZERO; m * d],
            g2: vec![T::ONE; d],
            dg2: vec![T::ZERO; d],
            ffn: SwiGLU::new(d, ff, lora_r),
            wkw: vec![T::ZERO; d * m],
            dwkw: vec![T::ZERO; d * m],
            wvw: vec![T::ZERO; d * m],
            dwvw: vec![T::ZERO; d * m],
            gu: vec![T::ZERO; d],
            dgu: vec![T::ZERO; d],
            gb: vec![T::ZERO; 1],
            dgb: vec![T::ZERO; 1],
            mem: Mrm::new(mcfg),
            mgrads: MrmGrads {
                gk: vec![T::ZERO; mcfg.fine * mcfg.d],
                gv: vec![T::ZERO; mcfg.fine * mcfg.d],
                gck: vec![T::ZERO; mcfg.coarse * mcfg.d],
                gcv: vec![T::ZERO; mcfg.coarse * mcfg.d],
            },
            cache: Vec::new(),
        }
    }

    pub fn reset_state(&mut self) {
        self.mem.reset();
        self.conv.reset();
        for v in self.conv_carry.iter_mut() {
            *v = T::ZERO;
        }
        self.zero_mgrads();
        self.cache.clear();
    }

    fn zero_mgrads(&mut self) {
        for g in [
            self.mgrads.gk.as_mut_slice(),
            self.mgrads.gv.as_mut_slice(),
            self.mgrads.gck.as_mut_slice(),
            self.mgrads.gcv.as_mut_slice(),
        ] {
            for v in g.iter_mut() {
                *v = T::ZERO;
            }
        }
    }

    /// Pillars 1→2→3→4→5→6 (mix) →Wo→residual, 7, 8. Mutates h in place.
    /// Returns the attention output (pre-mix): stage 0's output is the value-residual base.
    pub fn fwd_step(&mut self, h: &mut [T], tidx: u32, base: Option<&[T]>, train: bool) -> Vec<T> {
        let d = self.d;
        // 1: affine RMSNorm (pre-attention)
        let mut n1 = vec![T::ZERO; d];
        let mut xhat1 = vec![T::ZERO; d];
        let s1 = rmsnorm_fwd(h, &self.g1, &mut n1, &mut xhat1);
        // 2: causal depthwise highway conv
        let mut c = vec![T::ZERO; d];
        self.conv.fwd_step(&n1, &mut c);
        // 3–5: QK-norm + adaptive RoPE + differential attention over memory slots
        let mut out64 = vec![T::ZERO; self.m];
        let acache = self.attn.fwd_step(&c, &self.mem, tidx, &mut out64);
        // 6: ResFormer value residual mix
        let out_mix = match base {
            Some(b) => {
                let w = T::ONE / (T::ONE + (-self.vres_g[0]).exp());
                let mut om = vec![T::ZERO; self.m];
                for i in 0..self.m {
                    om[i] = w * out64[i] + (T::ONE - w) * b[i];
                }
                om
            }
            None => out64.clone(),
        };
        // Wo projection + residual
        let mut wom = vec![T::ZERO; d];
        T::gemv(&self.wo, &out_mix, &mut wom, self.m, d);
        T::vadd(h, &wom);
        // 7: SwiGLU 6x FFN + parallel LoRA
        let mut xhat2 = vec![T::ZERO; d];
        let mut n2 = vec![T::ZERO; d];
        let s2 = rmsnorm_fwd(h, &self.g2, &mut n2, &mut xhat2);
        let mut ffo = vec![T::ZERO; d];
        let mut lora_a = vec![T::ZERO; self.ffn.r];
        self.ffn.fwd(&n2, &mut ffo, &mut lora_a);
        T::vadd(h, &ffo);
        // 8: MRM write (3-tier state machine + coarse EMA)
        let (hn, hnorm) = {
            let (s, inv) = safe_inv_norm(h);
            let mut n = vec![T::ZERO; d];
            for i in 0..d {
                n[i] = h[i] * inv;
            }
            (n, s)
        };
        let mut k_raw = vec![T::ZERO; self.m];
        let mut v_raw = vec![T::ZERO; self.m];
        T::gemv(&self.wkw, &hn, &mut k_raw, d, self.m);
        T::gemv(&self.wvw, &hn, &mut v_raw, d, self.m);
        let glin = self.gb[0] + T::dot(&self.gu, &hn);
        let snap = self.mem.fwd_write(&k_raw, &v_raw, glin, &out64, train);

        // O(1) inference: the backward cache (incl. the MRM snapshot tape) is only
        // accumulated in training mode; at inference the stage keeps no per-step state.
        let base_out = out64.clone();
        if train {
            self.cache.push(StageCache {
                n1,
                xhat1,
                s1,
                c,
                attn: Some(acache),
                out64,
                out_mix,
                xhat2,
                s2,
                lora_a,
                hn,
                hnorm,
                k_raw,
                v_raw,
                glin,
                snap,
            });
        }
        base_out
    }

    /// Exact backward of one step. `t` is the step index; `gh` holds dL/d(h_out) and
    /// receives dL/d(h_in). `base` = stage-0 attention output at this step (layers > 0),
    /// `base_sink` accumulates dL/d(base), `ext` = external dL/d(out64) for stage 0.
    #[allow(clippy::too_many_arguments)]
    pub fn bwd_step(
        &mut self,
        t: usize,
        gh: &mut [T],
        base: Option<&[T]>,
        mut base_sink: Option<&mut [T]>,
        ext: Option<&[T]>,
    ) {
        let d = self.d;
        // ---- pillar 8 backward FIRST: the write path reads h AFTER the FFN residual,
        // so its route into gh must be accumulated before pillar 7 uses gh (the FFN/g2/
        // LoRA param grads and the norm-2 input route need the total dL/dh_after_ffn).
        // Consumes the slot grads accumulated by step t+1's attention read, routes them
        // to S(t−1), restores the state, and produces this step's write-path grads. ----
        let (mut gk_raw, mut gv_raw, mut gglin) = (vec![T::ZERO; self.m], vec![T::ZERO; self.m], T::ZERO);
        {
            let cache = &self.cache[t];
            if let Some(snap) = &cache.snap {
                self.mem.bwd_write(
                    snap,
                    &mut self.mgrads,
                    &mut gk_raw,
                    &mut gv_raw,
                    &mut gglin,
                    &cache.k_raw,
                    &cache.v_raw,
                );
            }
        }
        {
            let cache = &self.cache[t];
            T::gemv_dw(&cache.hn, &gk_raw, &mut self.dwkw, d, self.m);
            let mut dhn = vec![T::ZERO; d];
            T::gemv_dx(&self.wkw, &gk_raw, &mut dhn, d, self.m);
            T::gemv_dw(&cache.hn, &gv_raw, &mut self.dwvw, d, self.m);
            let mut tmp = vec![T::ZERO; d];
            T::gemv_dx(&self.wvw, &gv_raw, &mut tmp, d, self.m);
            T::vadd(&mut dhn, &tmp);
            for i in 0..d {
                self.dgu[i] = self.dgu[i] + gglin * cache.hn[i];
                dhn[i] = dhn[i] + gglin * self.gu[i];
            }
            self.dgb[0] = self.dgb[0] + gglin;
            // hn = h/‖h‖, so dL/dh = (I − hn hnᵀ)·dhn / ‖h‖ — the full normalization
            // Jacobian, including the radial term.
            let inv = T::ONE / cache.hnorm;
            let dhn_hn = T::dot(&dhn, &cache.hn);
            for i in 0..d {
                gh[i] = gh[i] + (dhn[i] - cache.hn[i] * dhn_hn) * inv;
            }
        }
        // ---- pillar 7 backward (FFN + LoRA + RMSNorm-2), on the now-complete gh ----
        {
            let cache = &self.cache[t];
            let mut dx2 = vec![T::ZERO; d];
            self.ffn.bwd(&cache.xhat2, &cache.lora_a, gh, &mut dx2);
            let mut dn2 = vec![T::ZERO; d];
            rmsnorm_bwd(&dx2, &self.g2, &cache.xhat2, cache.s2, &mut dn2, &mut self.dg2);
            T::vadd(gh, &dn2);
        }
        // ---- pillars 5/6 backward: Wo, value residual, differential attention ----
        let mut dout_mix = vec![T::ZERO; self.m];
        {
            let cache = &self.cache[t];
            T::gemv_dx(&self.wo, gh, &mut dout_mix, self.m, d);
            T::gemv_dw(&cache.out_mix, gh, &mut self.dwo, self.m, d);
        }
        let mut dout64 = vec![T::ZERO; self.m];
        {
            let cache = &self.cache[t];
            match base {
                Some(b) => {
                    let w = T::ONE / (T::ONE + (-self.vres_g[0]).exp());
                    let sink = base_sink.take().unwrap();
                    for i in 0..self.m {
                        dout64[i] = w * dout_mix[i];
                        sink[i] = sink[i] + (T::ONE - w) * dout_mix[i];
                        let diff = cache.out64[i] - b[i];
                        self.dvres_g[0] = self.dvres_g[0] + dout_mix[i] * diff;
                    }
                }
                None => {
                    for i in 0..self.m {
                        dout64[i] = dout_mix[i];
                    }
                }
            }
        }
        if let Some(e) = ext {
            for i in 0..self.m {
                dout64[i] = dout64[i] + e[i];
            }
        }
        {
            let cache = &self.cache[t];
            let mut dc = vec![T::ZERO; d];
            if let Some(ac) = &cache.attn {
                if ac.n_alive > 0 {
                    self.attn
                        .bwd_step(&cache.c, &self.mem, t as u32, ac, &dout64, &mut self.mgrads, &mut dc);
                }
            }
            // ---- pillar 2 backward (conv), then pillar 1 backward (RMSNorm-1) ----
            let mut dn1 = vec![T::ZERO; d];
            {
                let mut xprev: Vec<&[T]> = Vec::with_capacity(self.conv.k - 1);
                for j in 1..self.conv.k {
                    if t >= j {
                        xprev.push(self.cache[t - j].n1.as_slice());
                    } else {
                        xprev.push(self.zero_row.as_slice());
                    }
                }
                self.conv
                    .bwd_step(&dc, &cache.n1, &xprev, &mut self.conv_carry, &mut dn1);
            }
            let mut dh0 = vec![T::ZERO; d];
            rmsnorm_bwd(&dn1, &self.g1, &cache.xhat1, cache.s1, &mut dh0, &mut self.dg1);
            T::vadd(gh, &dh0);
        }
    }

    pub fn zero_grad(&mut self) {
        for v in self.dg1.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dg2.iter_mut() {
            *v = T::ZERO;
        }
        self.conv.zero_grad();
        self.attn.zero_grad();
        for v in self.dvres_g.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dwo.iter_mut() {
            *v = T::ZERO;
        }
        self.ffn.zero_grad();
        for v in self.dwkw.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dwvw.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dgu.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dgb.iter_mut() {
            *v = T::ZERO;
        }
        self.zero_mgrads();
    }

    pub fn state_bytes(&self) -> usize {
        self.mem.state_bytes() + self.conv.hist.len() * std::mem::size_of::<T>() + 16
    }
}
