//! Pillars 3–6 — Differential Attention over the working-memory slots.
//!
//!   A_k = softmax(scale · Q_k K_kᵀ / √dsub)           (k ∈ {1,2}, paired sub-heads)
//!   DiffAttn(Q,K,V) = (A₁ − λ_eff · A₂) V
//!   λ_init(l) = 0.8 − 0.6·e^(−0.3(l−1))
//!   λ_eff   = max(0, exp(aₚ) − exp(bₚ) + λ_init)
//!
//! Q and slot-keys pass through per-sub-head QK-RMSNorm (Pillar 3, bounds |q·k| ≤ dsub)
//! and adaptive RoPE (Pillar 4). Query position is t; a fine slot's key is rotated by
//! (τ_slot − t), its last-write time, giving an exact relative-position encoding with
//! O(1) state. Coarse centroids are matched purely by content. The attention output is
//! scaled by (1 − λ_init) for cross-layer magnitude parity (as in the Diff Transformer).

use crate::mrm::{Mrm, MrmGrads};
use crate::norms::{qknorm_bwd_chunk, qknorm_fwd};
use crate::num::Num;
use crate::rope::Rope;

pub struct DiffAttn<T: Num> {
    pub heads: usize,
    pub dh: usize,
    pub dsub: usize,
    pub d: usize,
    pub wq: Vec<T>, // [d][heads*dh]
    pub dwq: Vec<T>,
    pub gq: Vec<T>, // [dsub] shared QK gain
    pub dgq: Vec<T>,
    pub lam_a: Vec<T>, // [heads]
    pub dlam_a: Vec<T>,
    pub lam_b: Vec<T>,
    pub dlam_b: Vec<T>,
    pub scale: Vec<T>, // [heads] learnable logit scale, init 1
    pub dscale: Vec<T>,
    pub rope: Rope<T>,
    pub lam_init: f64,
    pub out_scale: f64,
    pub n_read: usize,
}

pub struct AttnCache<T: Num> {
    pub qraw: Vec<T>, // heads*dh, pre-norm pre-rope
    pub a1: Vec<T>,   // [heads][n_read]
    pub a2: Vec<T>,
    pub lam_eff: Vec<f64>,
    pub n_alive: usize,
}

impl<T: Num> DiffAttn<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(d: usize, heads: usize, dh: usize, layer_1based: usize, n_read: usize, rope_base: f64) -> Self {
        let dsub = dh / 2;
        let lam_init = 0.8 - 0.6 * (-0.3 * (layer_1based as f64 - 1.0)).exp();
        Self {
            heads,
            dh,
            dsub,
            d,
            wq: vec![T::ZERO; d * heads * dh],
            dwq: vec![T::ZERO; d * heads * dh],
            gq: vec![T::ONE; dsub],
            dgq: vec![T::ZERO; dsub],
            lam_a: vec![T::ZERO; heads],
            dlam_a: vec![T::ZERO; heads],
            lam_b: vec![T::ZERO; heads],
            dlam_b: vec![T::ZERO; heads],
            scale: vec![T::ONE; heads],
            dscale: vec![T::ZERO; heads],
            rope: Rope::new(dsub / 2, rope_base, 0.0),
            lam_init,
            out_scale: 1.0 - lam_init,
            n_read,
        }
    }
    pub fn zero_grad(&mut self) {
        for g in [
            &mut self.dwq,
            &mut self.dgq,
            &mut self.dlam_a,
            &mut self.dlam_b,
            &mut self.dscale,
        ] {
            for v in g.iter_mut() {
                *v = T::ZERO;
            }
        }
        self.rope.zero_grad();
    }
    #[inline]
    fn lam_eff(&self, hh: usize) -> f64 {
        let v = self.lam_a[hh].to().exp() - self.lam_b[hh].to().exp() + self.lam_init;
        if v > 0.0 {
            v
        } else {
            0.0
        }
    }

    /// Slot keys as seen by attention: content key -> QK-norm -> rotate by (τ − t).
    /// Coarse centroids skip rotation. Returns (rotated, per-slot times).
    fn build_krot(&self, mem: &Mrm<T>, tidx: u32) -> Vec<T> {
        let (_d, dh, dsub, fine) = (self.d, self.dh, self.dsub, mem.cfg.fine);
        let m = self.heads * dh; // slot row width
        let nsub = self.heads * 2;
        let mut krot = vec![T::ZERO; self.n_read * m];
        for i in 0..fine {
            let src = &mem.k[i * m..(i + 1) * m];
            let dst = &mut krot[i * m..(i + 1) * m];
            dst.copy_from_slice(src);
            for s in 0..nsub {
                qknorm_fwd(&mut dst[s * dsub..(s + 1) * dsub], &self.gq, 1, dsub);
            }
            if mem.alive[i] {
                let pos = mem.ts[i] as f64 - tidx as f64;
                self.rope.apply(dst, nsub, pos);
            }
        }
        for j in 0..mem.cfg.coarse {
            let r = fine + j;
            let src = &mem.ck[j * m..(j + 1) * m];
            let dst = &mut krot[r * m..(r + 1) * m];
            dst.copy_from_slice(src);
            for s in 0..nsub {
                qknorm_fwd(&mut dst[s * dsub..(s + 1) * dsub], &self.gq, 1, dsub);
            }
        }
        krot
    }

    pub fn fwd_step(&self, c: &[T], mem: &Mrm<T>, tidx: u32, out: &mut [T]) -> AttnCache<T> {
        let (heads, dh, dsub) = (self.heads, self.dh, self.dsub);
        let nsub = heads * 2;
        let n_read = self.n_read;
        let mut q = vec![T::ZERO; heads * dh];
        T::gemv(&self.wq, c, &mut q, self.d, heads * dh);
        let qraw = q.clone();
        for s in 0..nsub {
            qknorm_fwd(&mut q[s * dsub..(s + 1) * dsub], &self.gq, 1, dsub);
        }
        self.rope.apply(&mut q, nsub, tidx as f64);

        let krot = self.build_krot(mem, tidx);

        let mut n_alive = 0usize;
        for i in 0..mem.cfg.fine {
            if mem.alive[i] {
                n_alive += 1;
            }
        }
        for i in 0..mem.cfg.coarse {
            if mem.calive[i] {
                n_alive += 1;
            }
        }

        let mut a1 = vec![T::ZERO; heads * n_read];
        let mut a2 = vec![T::ZERO; heads * n_read];
        let mut lam_effs = vec![0.0f64; heads];

        if n_alive == 0 {
            for v in out.iter_mut() {
                *v = T::ZERO;
            }
            return AttnCache { qraw, a1, a2, lam_eff: lam_effs, n_alive };
        }

        let inv = 1.0 / (dsub as f64).sqrt();
        for hh in 0..heads {
            let lam_eff = self.lam_eff(hh);
            lam_effs[hh] = lam_eff;
            let qs1 = &q[hh * dh..hh * dh + dsub];
            let qs2 = &q[hh * dh + dsub..(hh + 1) * dh];
            let mut l1 = vec![T::ZERO; n_read];
            let mut l2 = vec![T::ZERO; n_read];
            for j in 0..n_read {
                let alive = if j < mem.cfg.fine { mem.alive[j] } else { mem.calive[j - mem.cfg.fine] };
                let base = j * heads * dh + hh * dh;
                if alive {
                    l1[j] = self.scale[hh] * T::dot(qs1, &krot[base..base + dsub]) * T::of(inv);
                    l2[j] = self.scale[hh] * T::dot(qs2, &krot[base + dsub..base + dh]) * T::of(inv);
                } else {
                    l1[j] = T::of(-1e4);
                    l2[j] = T::of(-1e4);
                }
            }
            T::softmax(&mut l1);
            T::softmax(&mut l2);
            for j in 0..n_read {
                a1[hh * n_read + j] = l1[j];
                a2[hh * n_read + j] = l2[j];
            }
            // out_hh = (A1 − λeff·A2)·V
            for qd in 0..dh {
                let mut acc = T::ZERO;
                for j in 0..n_read {
                    let vval = if j < mem.cfg.fine {
                        mem.v[j * heads * dh + qd]
                    } else {
                        mem.cv[(j - mem.cfg.fine) * heads * dh + qd]
                    };
                    let p = a1[hh * n_read + j] - T::of(lam_eff) * a2[hh * n_read + j];
                    acc = acc + p * vval;
                }
                out[hh * dh + qd] = acc * T::of(self.out_scale);
            }
        }
        AttnCache { qraw, a1, a2, lam_eff: lam_effs, n_alive }
    }

    /// Backward. `dout` = dL/d(out64) (after value-residual mixing). Accumulates slot
    /// gradients into `gmem` and input gradient into `dc`.
    pub fn bwd_step(
        &mut self,
        c: &[T],
        mem: &Mrm<T>,
        tidx: u32,
        cache: &AttnCache<T>,
        dout: &[T],
        gmem: &mut MrmGrads<T>,
        dc: &mut [T],
    ) {
        if cache.n_alive == 0 {
            return;
        }
        let (heads, dh, dsub) = (self.heads, self.dh, self.dsub);
        let m = heads * dh; // slot row width
        let nsub = heads * 2;
        let n_read = self.n_read;
        let fine = mem.cfg.fine;
        let inv = 1.0 / (dsub as f64).sqrt();

        // rebuild krot exactly as forward
        let krot = self.build_krot(mem, tidx);
        let mut gkrot = vec![T::ZERO; n_read * m]; // grads wrt rotated+normed keys
        let mut dq = vec![T::ZERO; heads * dh]; // grads wrt post-qknorm post-rope q

        for hh in 0..heads {
            let lam_eff = cache.lam_eff[hh];
            let do_hh = &dout[hh * dh..(hh + 1) * dh];
            let oscale = T::of(self.out_scale);
            // dP_j and value grads
            let mut dpv = vec![T::ZERO; n_read];
            let mut r2 = vec![T::ZERO; dh];
            for j in 0..n_read {
                let p = cache.a1[hh * n_read + j] - T::of(lam_eff) * cache.a2[hh * n_read + j];
                let vrow = if j < fine { &mem.v[j * m..(j + 1) * m] } else { &mem.cv[(j - fine) * m..(j - fine + 1) * m] };
                let mut dp = T::ZERO;
                for qd in 0..dh {
                    dp = dp + do_hh[qd] * vrow[qd];
                    gmem_slot_add(gmem, mem, j, qd, p * do_hh[qd] * oscale);
                    r2[qd] = r2[qd] + cache.a2[hh * n_read + j] * vrow[qd];
                }
                dpv[j] = dp * oscale;
            }
            // dL/dλ_eff = −(1−λ_init)·⟨dout_hh, Σ A2_j V_j⟩  (single out_scale factor)
            let mut dlam = 0.0f64;
            for qd in 0..dh {
                dlam -= do_hh[qd].to() * r2[qd].to();
            }
            dlam *= self.out_scale;
            if lam_eff > 0.0 {
                self.dlam_a[hh] = self.dlam_a[hh] + T::of(dlam * self.lam_a[hh].to().exp());
                self.dlam_b[hh] = self.dlam_b[hh] - T::of(dlam * self.lam_b[hh].to().exp());
            }
            // softmax backward
            let a1s = &cache.a1[hh * n_read..(hh + 1) * n_read];
            let a2s = &cache.a2[hh * n_read..(hh + 1) * n_read];
            let mut g1 = vec![T::ZERO; n_read];
            let mut g2 = vec![T::ZERO; n_read];
            let (m1, m2) = {
                let mut s1 = T::ZERO;
                let mut s2 = T::ZERO;
                for j in 0..n_read {
                    s1 = s1 + dpv[j] * a1s[j];
                    s2 = s2 + (-T::of(lam_eff) * dpv[j]) * a2s[j];
                }
                (s1, s2)
            };
            for j in 0..n_read {
                g1[j] = a1s[j] * (dpv[j] - m1);
                g2[j] = a2s[j] * (-T::of(lam_eff) * dpv[j] - m2);
            }
            // logit grads -> q/k grads; also learnable scale
            // recompute normed+roped q (cheap)
            let mut qpost = cache.qraw.clone();
            for s in 0..nsub {
                qknorm_fwd(&mut qpost[s * dsub..(s + 1) * dsub], &self.gq, 1, dsub);
            }
            self.rope.apply(&mut qpost, nsub, tidx as f64);
            let q1n = &qpost[hh * dh..hh * dh + dsub];
            let q2n = &qpost[hh * dh + dsub..(hh + 1) * dh];
            for j in 0..n_read {
                let base = j * m + hh * dh;
                let k1n = &krot[base..base + dsub];
                let k2n = &krot[base + dsub..base + dh];
                let sc = self.scale[hh] * T::of(inv);
                for qd in 0..dsub {
                    dq[hh * dh + qd] = dq[hh * dh + qd] + g1[j] * k1n[qd] * sc;
                    dq[hh * dh + dsub + qd] = dq[hh * dh + dsub + qd] + g2[j] * k2n[qd] * sc;
                    gkrot[base + qd] = gkrot[base + qd] + g1[j] * q1n[qd] * sc;
                    gkrot[base + dsub + qd] = gkrot[base + dsub + qd] + g2[j] * q2n[qd] * sc;
                }
                let s1c = T::dot(q1n, k1n);
                let s2c = T::dot(q2n, k2n);
                self.dscale[hh] = self.dscale[hh] + g1[j] * s1c * T::of(inv) + g2[j] * s2c * T::of(inv);
            }
        }

        // q path backward: rope then qknorm. vraw for the rotation chain is the
        // POST-NORM (pre-rope) q — recompute it.
        let mut qnorm = cache.qraw.clone();
        for s in 0..nsub {
            qknorm_fwd(&mut qnorm[s * dsub..(s + 1) * dsub], &self.gq, 1, dsub);
        }
        let mut dqraw = dq;
        self.rope.apply_bwd(&mut dqraw, &qnorm, nsub, tidx as f64);
        // dqraw currently holds dL/d(qknorm output); qknorm bwd per subhead:
        let mut dq_prenorm = vec![T::ZERO; heads * dh];
        for s in 0..nsub {
            qknorm_bwd_chunk(
                &dqraw[s * dsub..(s + 1) * dsub],
                &cache.qraw[s * dsub..(s + 1) * dsub],
                &self.gq,
                &mut dq_prenorm[s * dsub..(s + 1) * dsub],
                &mut self.dgq,
                dsub,
            );
        }
        T::gemv_dw(c, &dq_prenorm, &mut self.dwq, self.d, heads * dh);
        T::gemv_dx(&self.wq, &dq_prenorm, dc, self.d, heads * dh);

        // k path backward: un-rope then qknorm-bwd, route into memory grads
        for i in 0..fine {
            if !mem.alive[i] {
                continue;
            }
            let mut grow = vec![T::ZERO; m];
            grow.copy_from_slice(&gkrot[i * m..(i + 1) * m]);
            let mut raw_rot = vec![T::ZERO; m];
            raw_rot.copy_from_slice(&mem.k[i * m..(i + 1) * m]);
            for s in 0..nsub {
                qknorm_fwd(&mut raw_rot[s * dsub..(s + 1) * dsub], &self.gq, 1, dsub);
            }
            let pos = mem.ts[i] as f64 - tidx as f64;
            // norm-then-rope in fwd: bwd = un-rope then un-norm
            self.rope.apply_bwd(&mut grow, &raw_rot, nsub, pos);
            let mut gcontent = vec![T::ZERO; m];
            for s in 0..nsub {
                qknorm_bwd_chunk(
                    &grow[s * dsub..(s + 1) * dsub],
                    &mem.k[i * m + s * dsub..i * m + (s + 1) * dsub],
                    &self.gq,
                    &mut gcontent[s * dsub..(s + 1) * dsub],
                    &mut self.dgq,
                    dsub,
                );
            }
            for qd in 0..m {
                gmem.gk[i * m + qd] = gmem.gk[i * m + qd] + gcontent[qd];
            }
        }
        for jj in 0..mem.cfg.coarse {
            let r = fine + jj;
            let mut gcontent = vec![T::ZERO; m];
            for s in 0..nsub {
                qknorm_bwd_chunk(
                    &gkrot[r * m + s * dsub..r * m + (s + 1) * dsub],
                    &mem.ck[jj * m + s * dsub..jj * m + (s + 1) * dsub],
                    &self.gq,
                    &mut gcontent[s * dsub..(s + 1) * dsub],
                    &mut self.dgq,
                    dsub,
                );
            }
            for qd in 0..m {
                gmem.gck[jj * m + qd] = gmem.gck[jj * m + qd] + gcontent[qd];
            }
        }
    }
}

#[inline]
#[allow(dead_code)] // analytic λ reference kept for diagnostics
fn lam_eff_raw(x: f64) -> f64 {
    x.exp()
}

/// accumulate a value-grad contribution to the right slot family
#[inline]
fn gmem_slot_add<T: Num>(gmem: &mut MrmGrads<T>, mem: &Mrm<T>, j: usize, qd: usize, val: T) {
    if j < mem.cfg.fine {
        gmem.gv[j * mem.cfg.d + qd] = gmem.gv[j * mem.cfg.d + qd] + val;
    } else {
        let r = j - mem.cfg.fine;
        gmem.gcv[r * mem.cfg.d + qd] = gmem.gcv[r * mem.cfg.d + qd] + val;
    }
}

