//! TESSERA-Q model: pillar 9 (soft-capped logits + dual-head MTP + auxiliary Z-loss)
//! and the orchestration of all stages, with streaming O(1)-state inference and
//! exact BPTT training.

use crate::layer::Stage;
use crate::mrm::MrmCfg;
use crate::norms::{rmsnorm_bwd, rmsnorm_fwd};
use crate::num::Num;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug)]
pub struct ModelCfg {
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub m: usize,
    pub ff: usize,
    pub lora_r: usize,
    pub conv_k: usize,
    pub fine: usize,
    pub coarse: usize,
    pub vocab: usize,
    pub mtp_alpha: f64,
    pub zloss_w: f64,
    pub cap: f64,
    pub rope_base: f64,
}

impl Default for ModelCfg {
    fn default() -> Self {
        Self {
            d: 128,
            layers: 2,
            heads: 2,
            m: 64,
            ff: 768,
            lora_r: 8,
            conv_k: 4,
            fine: 128,
            coarse: 16,
            vocab: 256,
            mtp_alpha: 0.3,
            zloss_w: 1e-4,
            cap: 30.0,
            rope_base: 10000.0,
        }
    }
}

pub struct MCache<T: Num> {
    pub token: usize,
    pub xhatn: Vec<T>,
    pub sn: T,
}

pub struct Model<T: Num> {
    pub cfg: ModelCfg,
    pub embed: Vec<T>, // [vocab][d]
    pub dembed: Vec<T>,
    pub stages: Vec<Stage<T>>,
    pub gf: Vec<T>,
    pub dgf: Vec<T>,
    pub head1: Vec<T>, // [d][vocab] next token
    pub dhead1: Vec<T>,
    pub head2: Vec<T>, // [d][vocab] next-next token (MTP)
    pub dhead2: Vec<T>,
    pub t: u64,
    pub cache: Vec<MCache<T>>,
    pub vres_base: Vec<Vec<T>>,
    pub vres_base_grad: Vec<Vec<T>>,
}

pub struct Pref<T> {
    pub w: *mut Vec<T>,
    pub g: *mut Vec<T>,
    pub wd: bool,
    pub name: &'static str,
}

impl<T: Num> Model<T> {
    pub fn new(cfg: ModelCfg, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let d = cfg.d;
        let dh = cfg.m / cfg.heads; // memory-attention head dim
        let mcfg = MrmCfg {
            d: cfg.m,
            fine: cfg.fine,
            coarse: cfg.coarse,
            ..MrmCfg::default()
        };
        let n_read = cfg.fine + cfg.coarse;
        let mut stages = Vec::new();
        for l in 0..cfg.layers {
            stages.push(Stage::new(l, d, cfg.heads, dh, cfg.ff, cfg.lora_r, cfg.conv_k, mcfg, n_read, cfg.rope_base));
        }
        let mut m = Self {
            embed: vec![T::ZERO; cfg.vocab * d],
            dembed: vec![T::ZERO; cfg.vocab * d],
            gf: vec![T::ONE; d],
            dgf: vec![T::ZERO; d],
            head1: vec![T::ZERO; d * cfg.vocab],
            dhead1: vec![T::ZERO; d * cfg.vocab],
            head2: vec![T::ZERO; d * cfg.vocab],
            dhead2: vec![T::ZERO; d * cfg.vocab],
            stages,
            cfg,
            t: 0,
            cache: Vec::new(),
            vres_base: Vec::new(),
            vres_base_grad: Vec::new(),
        };
        m.init(&mut rng);
        m
    }

    fn init(&mut self, rng: &mut Rng) {
        let c = &self.cfg;
        let d = c.d;
        let norm = |x: f64| x.sqrt();
        init_vec(rng, &mut self.embed, 0.02);
        for s in self.stages.iter_mut() {
            init_vec(rng, &mut s.attn.wq, 1.0 / norm(d as f64));
            init_vec(rng, &mut s.wo, 1.0 / norm(c.m as f64) / norm(2.0 * c.layers as f64));
            init_vec(rng, &mut s.ffn.wg, 1.0 / norm(d as f64));
            init_vec(rng, &mut s.ffn.wu, 1.0 / norm(d as f64));
            init_vec(rng, &mut s.ffn.wd, 0.5 / norm(c.ff as f64) / norm(2.0 * c.layers as f64));
            init_vec(rng, &mut s.ffn.la, 1.0 / norm(d as f64));
            // lb stays exactly zero (LoRA B init = 0)
            for j in 0..s.conv.k {
                for q in 0..d {
                    s.conv.w[j * d + q] = T::of(1.0 / s.conv.k as f64 + 0.02 * rng.normal());
                }
            }
            for q in 0..d {
                s.conv.gate[q] = T::of(0.5 * rng.normal());
            }
            init_vec(rng, &mut s.wkw, 1.0 / norm(d as f64));
            init_vec(rng, &mut s.wvw, 1.0 / norm(d as f64));
            init_vec(rng, &mut s.gu, 0.1 / norm(d as f64));
            s.gb[0] = T::of(2.0); // eager writes early in training
        }
        init_vec(rng, &mut self.head1, 0.02);
        init_vec(rng, &mut self.head2, 0.02);
    }

    pub fn reset_state(&mut self) {
        self.t = 0;
        self.cache.clear();
        self.vres_base.clear();
        self.vres_base_grad.clear();
        for s in self.stages.iter_mut() {
            s.reset_state();
        }
    }

    pub fn zero_grad(&mut self) {
        for v in self.dembed.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dgf.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dhead1.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dhead2.iter_mut() {
            *v = T::ZERO;
        }
        for s in self.stages.iter_mut() {
            s.zero_grad();
        }
    }

    /// One streaming step. Returns (z_main, z_aux) logits (pre-cap).
    pub fn fwd_step(&mut self, token: usize, train: bool) -> (Vec<T>, Vec<T>) {
        let d = self.cfg.d;
        let tidx = self.t as u32;
        self.t += 1;
        let mut h = self.embed[token * d..(token + 1) * d].to_vec();
        let base0 = self.stages[0].fwd_step(&mut h, tidx, None, train);
        if train {
            self.vres_base.push(base0.clone());
            self.vres_base_grad.push(vec![T::ZERO; self.cfg.m]);
        }
        for l in 1..self.cfg.layers {
            self.stages[l].fwd_step(&mut h, tidx, Some(&base0), train);
        }
        let mut xhatn = vec![T::ZERO; d];
        let mut hn = vec![T::ZERO; d];
        let sn = rmsnorm_fwd(&h, &self.gf, &mut hn, &mut xhatn);
        if train {
            self.cache.push(MCache { token, xhatn, sn });
        }
        let mut z1 = vec![T::ZERO; self.cfg.vocab];
        let mut z2 = vec![T::ZERO; self.cfg.vocab];
        T::gemv(&self.head1, &hn, &mut z1, d, self.cfg.vocab);
        T::gemv(&self.head2, &hn, &mut z2, d, self.cfg.vocab);
        (z1, z2)
    }

    /// Pillar 9 loss: masked CE(main, t+1) + α·CE(aux, t+2) + 1e−4·lse², on soft-capped logits.
    pub fn loss_step(&self, z1: &[T], z2: &[T], tgt1: i32, tgt2: i32, mask: f64) -> (f64, Vec<T>, Vec<T>) {
        let cap = self.cfg.cap;
        let v = self.cfg.vocab;
        let a = self.cfg.mtp_alpha;
        let mut c1 = vec![0.0f64; v];
        let mut c2 = vec![0.0f64; v];
        for i in 0..v {
            c1[i] = cap * (z1[i].to() / cap).tanh();
            c2[i] = cap * (z2[i].to() / cap).tanh();
        }
        let (p1, lp1) = softmax_ln(&c1, tgt1);
        let (p2, lp2) = softmax_ln(&c2, tgt2);
        let _ = (p1.clone(), p2.clone());
        let lse1 = lse(z1);
        let mut loss = mask * (-lp1) + mask * a * (-lp2) + self.cfg.zloss_w * lse1 * lse1;
        let _ = &mut loss;
        let mut dz1 = vec![T::ZERO; v];
        let mut dz2 = vec![T::ZERO; v];
        // p_raw = softmax(z) for the z-loss path
        let mut pz = vec![0.0f64; v];
        {
            let mut zmax = f64::NEG_INFINITY;
            for i in 0..v {
                zmax = zmax.max(z1[i].to());
            }
            let mut s = 0.0;
            for i in 0..v {
                pz[i] = (z1[i].to() - zmax).exp();
                s += pz[i];
            }
            for i in 0..v {
                pz[i] /= s;
            }
        }
        let tgt1_ok = tgt1 >= 0 && (tgt1 as usize) < v;
        let tgt2_ok = tgt2 >= 0 && (tgt2 as usize) < v;
        for i in 0..v {
            let t1 = if tgt1_ok && i as i32 == tgt1 { 1.0 } else { 0.0 };
            let t2 = if tgt2_ok && i as i32 == tgt2 { 1.0 } else { 0.0 };
            let j1 = 1.0 - (c1[i] / cap) * (c1[i] / cap);
            let j2 = 1.0 - (c2[i] / cap) * (c2[i] / cap);
            dz1[i] = T::of(mask * (p1[i] - t1) * j1 + 2.0 * self.cfg.zloss_w * lse1 * pz[i]);
            // masked tail: no target => no aux gradient (avoid pushing toward "nothing")
            dz2[i] = T::of(if tgt2_ok { mask * a * (p2[i] - t2) * j2 } else { 0.0 });
        }
        (loss, dz1, dz2)
    }

    /// Full-sequence forward (streams all tokens, recording training caches).
    pub fn fwd_seq(&mut self, toks: &[usize], train: bool) {
        self.reset_state();
        for &tk in toks {
            self.fwd_step(tk, train);
        }
    }

    /// Exact BPTT backward over the recorded caches. Targets/mask are aligned so that
    /// position t predicts toks[t+1] (main) and toks[t+2] (aux). Returns total loss.
    pub fn bwd_seq(&mut self, tgt1: &[i32], tgt2: &[i32], mask: &[f64]) -> f64 {
        self.bwd_seq_impl(tgt1, tgt2, mask, None, 0.0, &[])
    }

    /// Backward with an optional knowledge-distillation seed (`transplant`).
    /// `kd[t] = Some(probs)` holds the teacher's distribution over the
    /// retrieval-alphabet logits at position t. Adds, on the alphabet subset A
    /// of the CAPPED logits c1 (identical chain to loss_step's CE gradient):
    ///   dL/dc1_i = kd_w · (softmax(c1|A)_i − T_i),  dz1_i = that · sech²(z1_i/cap)
    /// and `kd_w · −Σ T log softmax(c1|A)` to the reported loss. Everything
    /// downstream (heads, norm, BPTT tape) reuses the same analytical path.
    pub fn bwd_seq_kd(
        &mut self,
        tgt1: &[i32],
        tgt2: &[i32],
        mask: &[f64],
        kd: &[Option<Vec<f64>>],
        kd_w: f64,
        alpha: &[usize],
    ) -> f64 {
        self.bwd_seq_impl(tgt1, tgt2, mask, Some(kd), kd_w, alpha)
    }

    fn bwd_seq_impl(
        &mut self,
        tgt1: &[i32],
        tgt2: &[i32],
        mask: &[f64],
        kd: Option<&[Option<Vec<f64>>]>,
        kd_w: f64,
        alpha: &[usize],
    ) -> f64 {
        let d = self.cfg.d;
        let v = self.cfg.vocab;
        let n = self.cache.len();
        let mut total = 0.0f64;
        for t in (0..n).rev() {
            // recompute logits from cached final hidden. The fwd applies the final
            // norm gain: hn = gf ⊙ xhatn — the recompute must match exactly.
            let (hn, xhatn) = {
                let xhatn = self.cache[t].xhatn.clone();
                let mut hn = vec![T::ZERO; d];
                for i in 0..d {
                    hn[i] = xhatn[i] * self.gf[i];
                }
                (hn, xhatn)
            };
            let (z1, z2) = {
                let mut z1 = vec![T::ZERO; v];
                let mut z2 = vec![T::ZERO; v];
                T::gemv(&self.head1, &hn, &mut z1, d, v);
                T::gemv(&self.head2, &hn, &mut z2, d, v);
                (z1, z2)
            };
            let (loss, mut dz1, dz2) = self.loss_step(&z1, &z2, tgt1[t], tgt2[t], mask[t]);
            total += loss;
            // knowledge-distillation seed (soft teacher targets on the alphabet)
            if let Some(kds) = kd {
                if let Some(tp) = &kds[t] {
                    if kd_w > 0.0 && !alpha.is_empty() {
                        let cap = self.cfg.cap;
                        // log-softmax of the alphabet subset of capped logits
                        let k = alpha.len();
                        let mut logp: Vec<f64> = Vec::with_capacity(k);
                        let mut zmax = f64::NEG_INFINITY;
                        for &i in alpha {
                            let c = cap * (z1[i].to() / cap).tanh();
                            logp.push(c);
                            zmax = zmax.max(c);
                        }
                        let mut s = 0.0f64;
                        for v in logp.iter_mut() {
                            *v = (*v - zmax).exp();
                            s += *v;
                        }
                        let ls = s.ln();
                        let mut lkd = 0.0f64;
                        for (kk, &i) in alpha.iter().enumerate() {
                            let lp = logp[kk].ln() - ls;
                            let tt = tp[kk].min(1.0);
                            lkd -= tt * lp;
                            let p = lp.exp();
                            // softcap chain: d(c)/d(z) = sech²(z/cap) = 1 − tanh²(z/cap)
                            let j = 1.0 - (z1[i].to() / cap) * (z1[i].to() / cap);
                            dz1[i] = dz1[i] + T::of(kd_w * (p - tt) * j);
                        }
                        total += kd_w * lkd;
                    }
                }
            }
            T::gemv_dw(&hn, &dz1, &mut self.dhead1, d, v);
            T::gemv_dw(&hn, &dz2, &mut self.dhead2, d, v);
            // dxn = dL/dhn ; rmsnorm_bwd routes it through the gain (dgf) to dh
            let mut dxn = vec![T::ZERO; d];
            {
                let mut t1 = vec![T::ZERO; d];
                T::gemv_dx(&self.head1, &dz1, &mut t1, d, v);
                let mut t2 = vec![T::ZERO; d];
                T::gemv_dx(&self.head2, &dz2, &mut t2, d, v);
                for i in 0..d {
                    dxn[i] = t1[i] + t2[i];
                }
            }
            let mut dh = vec![T::ZERO; d];
            rmsnorm_bwd(&dxn, &self.gf, &xhatn, self.cache[t].sn, &mut dh, &mut self.dgf);
            let token = self.cache[t].token;
            self.bwd_into_stages(t, &mut dh);
            for i in 0..d {
                self.dembed[token * d + i] = self.dembed[token * d + i] + dh[i];
            }
        }
        total
    }

    fn bwd_into_stages(&mut self, t: usize, dh: &mut [T]) {
        // split borrows: stages / vres_base / vres_base_grad are disjoint fields
        let vbase = &self.vres_base;
        let vgrad = &mut self.vres_base_grad;
        let stages = &mut self.stages;
        for l in (0..self.cfg.layers).rev() {
            let (base, ext, sink): (Option<&[T]>, Option<&[T]>, Option<&mut [T]>);
            if l > 0 {
                base = Some(vbase[t].as_slice());
                ext = None;
                sink = Some(vgrad[t].as_mut_slice());
            } else {
                base = None;
                ext = Some(vgrad[t].as_slice());
                sink = None;
            }
            stages[l].bwd_step(t, dh, base, sink, ext);
        }
    }

    pub fn params(&mut self) -> Vec<Pref<T>> {
        let mut out = Vec::with_capacity(128);
        out.push(Pref { w: &mut self.embed as *mut _, g: &mut self.dembed as *mut _, wd: true, name: "embed" });
        for (l, s) in self.stages.iter_mut().enumerate() {
            let lname: &'static str = if l == 0 { "s0" } else { "s1" };
            out.push(Pref { w: &mut s.g1 as *mut _, g: &mut s.dg1 as *mut _, wd: false, name: "g1" });
            out.push(Pref { w: &mut s.conv.w as *mut _, g: &mut s.conv.dw as *mut _, wd: false, name: "conv.w" });
            out.push(Pref { w: &mut s.conv.gate as *mut _, g: &mut s.conv.dgate as *mut _, wd: false, name: "conv.gate" });
            out.push(Pref { w: &mut s.conv.bg as *mut _, g: &mut s.conv.dbg as *mut _, wd: false, name: "conv.bg" });
            out.push(Pref { w: &mut s.attn.wq as *mut _, g: &mut s.attn.dwq as *mut _, wd: true, name: "attn.wq" });
            out.push(Pref { w: &mut s.attn.gq as *mut _, g: &mut s.attn.dgq as *mut _, wd: false, name: "attn.gq" });
            out.push(Pref { w: &mut s.attn.lam_a as *mut _, g: &mut s.attn.dlam_a as *mut _, wd: false, name: "attn.lam_a" });
            out.push(Pref { w: &mut s.attn.lam_b as *mut _, g: &mut s.attn.dlam_b as *mut _, wd: false, name: "attn.lam_b" });
            out.push(Pref { w: &mut s.attn.scale as *mut _, g: &mut s.attn.dscale as *mut _, wd: false, name: "attn.scale" });
            out.push(Pref { w: &mut s.attn.rope.eta as *mut _, g: &mut s.attn.rope.deta as *mut _, wd: false, name: "rope.eta" });
            out.push(Pref { w: &mut s.vres_g as *mut _, g: &mut s.dvres_g as *mut _, wd: false, name: "vres_g" });
            out.push(Pref { w: &mut s.wo as *mut _, g: &mut s.dwo as *mut _, wd: true, name: "wo" });
            out.push(Pref { w: &mut s.g2 as *mut _, g: &mut s.dg2 as *mut _, wd: false, name: "g2" });
            out.push(Pref { w: &mut s.ffn.wg as *mut _, g: &mut s.ffn.dwg as *mut _, wd: true, name: "ffn.wg" });
            out.push(Pref { w: &mut s.ffn.wu as *mut _, g: &mut s.ffn.dwu as *mut _, wd: true, name: "ffn.wu" });
            out.push(Pref { w: &mut s.ffn.wd as *mut _, g: &mut s.ffn.dwd as *mut _, wd: true, name: "ffn.wd" });
            out.push(Pref { w: &mut s.ffn.la as *mut _, g: &mut s.ffn.dla as *mut _, wd: false, name: "lora.a" });
            out.push(Pref { w: &mut s.ffn.lb as *mut _, g: &mut s.ffn.dlb as *mut _, wd: false, name: "lora.b" });
            out.push(Pref { w: &mut s.wkw as *mut _, g: &mut s.dwkw as *mut _, wd: true, name: "mem.wkw" });
            out.push(Pref { w: &mut s.wvw as *mut _, g: &mut s.dwvw as *mut _, wd: true, name: "mem.wvw" });
            out.push(Pref { w: &mut s.gu as *mut _, g: &mut s.dgu as *mut _, wd: false, name: "mem.gu" });
            out.push(Pref { w: &mut s.gb as *mut _, g: &mut s.dgb as *mut _, wd: false, name: "mem.gb" });
            let _ = lname;
        }
        out.push(Pref { w: &mut self.gf as *mut _, g: &mut self.dgf as *mut _, wd: false, name: "gf" });
        out.push(Pref { w: &mut self.head1 as *mut _, g: &mut self.dhead1 as *mut _, wd: true, name: "head1" });
        out.push(Pref { w: &mut self.head2 as *mut _, g: &mut self.dhead2 as *mut _, wd: true, name: "head2" });
        out
    }

    pub fn count_params(&mut self) -> usize {
        self.params().iter().map(|p| unsafe { (*p.w).len() }).sum()
    }

    /// Inference working-memory footprint (constant in T — the O(1) claim).
    pub fn state_bytes(&self) -> usize {
        let mut b = 16 + 8; // counters
        for s in &self.stages {
            b += s.state_bytes();
        }
        b
    }
    pub fn substrate_bytes(&self) -> usize {
        self.stages.iter().map(|s| s.mem.substrate_bytes()).sum()
    }

    // ---------- checkpointing ----------

    pub fn save(&mut self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let c = &self.cfg;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"TSS1")?;
        for x in [c.d, c.layers, c.heads, c.m, c.ff, c.lora_r, c.conv_k, c.fine, c.coarse, c.vocab] {
            f.write_all(&(x as u32).to_le_bytes())?;
        }
        let params = self.params();
        f.write_all(&(params.len() as u32).to_le_bytes())?;
        unsafe {
            for p in &params {
                let w = &*p.w;
                f.write_all(&(w.len() as u32).to_le_bytes())?;
                for x in w.iter() {
                    f.write_all(&(x.to() as f32).to_le_bytes())?; // true f32 payload
                }
            }
        }
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        assert_eq!(&magic, b"TSS1", "bad checkpoint magic");
        let mut hdr = [0u32; 10];
        let mut hb = [0u8; 40];
        f.read_exact(&mut hb)?;
        for i in 0..10 {
            hdr[i] = u32::from_le_bytes(hb[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let c = &mut self.cfg;
        c.d = hdr[0] as usize;
        c.layers = hdr[1] as usize;
        c.heads = hdr[2] as usize;
        c.m = hdr[3] as usize;
        c.ff = hdr[4] as usize;
        c.lora_r = hdr[5] as usize;
        c.conv_k = hdr[6] as usize;
        c.fine = hdr[7] as usize;
        c.coarse = hdr[8] as usize;
        c.vocab = hdr[9] as usize;
        let mut nb = [0u8; 4];
        f.read_exact(&mut nb)?;
        let nparam = u32::from_le_bytes(nb) as usize;
        let params = self.params();
        assert_eq!(nparam, params.len(), "checkpoint param-count mismatch");
        unsafe {
            for p in &params {
                f.read_exact(&mut nb)?;
                let len = u32::from_le_bytes(nb) as usize;
                let w = &mut *p.w;
                assert_eq!(w.len(), len, "checkpoint shape mismatch");
                let mut buf = vec![0u8; len * 4];
                f.read_exact(&mut buf)?;
                for (i, x) in w.iter_mut().enumerate() {
                    *x = T::of(f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap()) as f64);
                }
            }
        }
        Ok(())
    }
}

// ---- helpers --------------------------------------------------------------------------

fn softmax_ln(p: &[f64], tgt: i32) -> (Vec<f64>, f64) {
    let mut m = f64::NEG_INFINITY;
    for &x in p {
        m = m.max(x);
    }
    let mut e: Vec<f64> = p.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    for x in e.iter_mut() {
        *x /= s;
    }
    if tgt < 0 || tgt as usize >= p.len() {
        return (e, 0.0); // masked tail position: no CE term
    }
    let ln = ((p[tgt as usize] - m).exp() / s).ln();
    (e, ln)
}

fn lse<T: Num>(z: &[T]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in z {
        m = m.max(x.to());
    }
    let mut s = 0.0;
    for &x in z {
        s += (x.to() - m).exp();
    }
    m + s.ln()
}

fn init_vec<T: Num>(rng: &mut Rng, v: &mut [T], std: f64) {
    for x in v.iter_mut() {
        *x = T::of(rng.normal_mu(0.0, std));
    }
}


#[cfg(test)]
mod tests {
    // U24..U25 — substrate invariants + tiny-config checkpoint roundtrip
    use super::*;

    #[test]
    fn u24_param_count_and_memory_substrate() {
        let mut m = Model::<f32>::new(ModelCfg::default(), 0);
        assert_eq!(m.count_params(), 760_256);
        assert_eq!(m.state_bytes(), 153_960);
        assert_eq!(m.substrate_bytes(), 147_456); // 144 KiB (~145 KB spec)
    }

    #[test]
    fn u25_save_load_bitexact_small_cfg() {
        let cfg = ModelCfg { d: 16, m: 8, ff: 16, fine: 4, coarse: 2, layers: 1, ..ModelCfg::default() };
        let dir = std::env::temp_dir().join("tessera_u25");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny_rt.bin");
        let path = path.to_str().unwrap().to_string();

        let mut a = Model::<f32>::new(cfg.clone(), 31337);
        // run a couple of tokens so weights are exercised, then checkpoint
        for t in [3usize, 17, 200] {
            a.fwd_step(t, false);
        }
        a.save(&path).unwrap();
        let mut b = Model::<f32>::new(cfg, 0);
        b.load(&path).unwrap();
        // identical inference state: feed the SAME token stream to both from reset
        a.reset_state();
        b.reset_state();
        // every parameter must be bit-identical
        let pa = a.params();
        let pb = b.params();
        assert_eq!(pa.len(), pb.len());
        for (x, y) in pa.iter().zip(pb.iter()) {
            unsafe {
                let wa: &Vec<f32> = &*x.w;
                let wb: &Vec<f32> = &*y.w;
                assert_eq!(wa.len(), wb.len(), "param {} shape", x.name);
                for (i, (p, q)) in wa.iter().zip(wb.iter()).enumerate() {
                    assert_eq!(p.to_bits(), q.to_bits(), "param {} [{}]", x.name, i);
                }
            }
        }
        // inference must match bit-exactly after reload on a fresh stream
        for t in [11usize, 86, 42] {
            let (za, _) = a.fwd_step(t, false);
            let (zb, _) = b.fwd_step(t, false);
            for (x, y) in za.iter().zip(zb.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "logits diverge at token {t}");
            }
        }
    }
}
