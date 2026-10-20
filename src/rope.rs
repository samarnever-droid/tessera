//! Pillar 4 — Adaptive RoPE Banding.
//!
//!   theta_i = base^(-2i/d) * (2 * sigmoid(eta_i)),   eta in R^{d/2} learnable
//!
//! Rotation angle for position p: phi_i = theta_i * p (p may be negative/relative for
//! slot keys whose last-write time is tau: p = tau - t). Backward is exact, including
//! the gradient wrt eta through the sigmoid band scale.

use crate::num::Num;

pub struct Rope<T: Num> {
    pub eta: Vec<T>,
    pub deta: Vec<T>,
    pub pairs: usize, // = d/2 for the vectors it is applied to
    pub base: f64,
}

impl<T: Num> Rope<T> {
    pub fn new(pairs: usize, base: f64, seedlike: f64) -> Self {
        Self {
            eta: vec![T::of(seedlike); pairs],
            deta: vec![T::ZERO; pairs],
            pairs,
            base,
        }
    }
    #[inline]
    fn inv_base(&self, i: usize) -> f64 {
        self.base.powf(-(2.0 * i as f64) / (2.0 * self.pairs as f64))
    }
    pub fn thetas(&self) -> Vec<f64> {
        (0..self.pairs)
            .map(|i| self.inv_base(i) * 2.0 / (1.0 + (-self.eta[i].to()).exp()))
            .collect()
    }
    pub fn zero_grad(&mut self) {
        for v in self.deta.iter_mut() {
            *v = T::ZERO;
        }
    }

    /// Rotate `nchunks` chunks of 2*pairs dims in place at position `pos`.
    pub fn apply(&self, v: &mut [T], nchunks: usize, pos: f64) {
        let th = self.thetas();
        let twop = 2 * self.pairs;
        for c in 0..nchunks {
            let base = c * twop;
            for i in 0..self.pairs {
                let phi = th[i] * pos;
                let (sphi, cphi) = phi.sin_cos();
                let a = v[base + 2 * i];
                let b = v[base + 2 * i + 1];
                v[base + 2 * i] = a * T::of(cphi) - b * T::of(sphi);
                v[base + 2 * i + 1] = a * T::of(sphi) + b * T::of(cphi);
            }
        }
    }

    /// Backward of `apply`. `vraw` = pre-rotation values; `g` holds dL/d(rotated) and is
    /// overwritten with dL/d(raw). dL/deta accumulates into `self.deta`.
    pub fn apply_bwd(&mut self, g: &mut [T], vraw: &[T], nchunks: usize, pos: f64) {
        let th = self.thetas();
        let twop = 2 * self.pairs;
        for c in 0..nchunks {
            let base = c * twop;
            for i in 0..self.pairs {
                let phi = th[i] * pos;
                let (sphi, cphi) = phi.sin_cos();
                let a = vraw[base + 2 * i];
                let b = vraw[base + 2 * i + 1];
                let g0 = g[base + 2 * i];
                let g1 = g[base + 2 * i + 1];
                // transposed rotation
                g[base + 2 * i] = g0 * T::of(cphi) + g1 * T::of(sphi);
                g[base + 2 * i + 1] = -g0 * T::of(sphi) + g1 * T::of(cphi);
                // dL/dphi  (computed in f64 for the eta chain)
                let gphi = g0.to() * (-a.to() * sphi - b.to() * cphi)
                    + g1.to() * (a.to() * cphi - b.to() * sphi);
                // dL/dtheta = gphi * pos ; dtheta/deta = inv_base(i) * 2 sig(1-sig)
                let sg = 1.0 / (1.0 + (-self.eta[i].to()).exp());
                let dth_deta = self.inv_base(i) * 2.0 * sg * (1.0 - sg);
                self.deta[i] = self.deta[i] + T::of(gphi * pos * dth_deta);
            }
        }
    }
}
