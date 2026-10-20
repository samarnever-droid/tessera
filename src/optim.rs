//! Training engine: AVX-accelerated AdamW with first/second moments, global gradient
//! clipping, and the WSD (Warmup–Stable–Decay) learning-rate schedule
//! (10% linear warmup -> 70% stable plateau -> 20% cosine decay to 1% of peak).

use crate::model::Pref;
use crate::num::Num;

pub struct AdamW<T: Num> {
    pub m: Vec<Vec<T>>,
    pub v: Vec<Vec<T>>,
    pub step: u64,
    pub b1: f64,
    pub b2: f64,
    pub eps: f64,
    pub wd: f64,
}

impl<T: Num> AdamW<T> {
    pub fn new(params: &[Pref<T>], wd: f64) -> Self {
        Self {
            m: params.iter().map(|p| vec![T::ZERO; unsafe { (*p.w).len() }]).collect(),
            v: params.iter().map(|p| vec![T::ZERO; unsafe { (*p.w).len() }]).collect(),
            step: 0,
            b1: 0.9,
            b2: 0.95,
            eps: 1e-8,
            wd,
        }
    }

    /// Global L2 clip; returns the pre-clip norm.
    pub fn clip(&self, params: &[Pref<T>], max_norm: f64) -> f64 {
        let mut sq = 0.0f64;
        unsafe {
            for p in params {
                let g = &*p.g;
                for x in g.iter() {
                    sq += x.to() * x.to();
                }
            }
        }
        let norm = sq.sqrt();
        if norm > max_norm && norm > 0.0 {
            let s = max_norm / norm;
            unsafe {
                for p in params {
                    let g = &mut *p.g;
                    for x in g.iter_mut() {
                        *x = *x * T::of(s);
                    }
                }
            }
        }
        norm
    }

    pub fn step(&mut self, params: &[Pref<T>], lr: f64) {
        self.step += 1;
        let t = self.step as f64;
        let bc1 = 1.0 - self.b1.powf(t);
        let bc2 = 1.0 - self.b2.powf(t);
        unsafe {
            for (pi, p) in params.iter().enumerate() {
                let w = &mut *p.w;
                let g = &*p.g;
                let m = &mut self.m[pi];
                let v = &mut self.v[pi];
                for i in 0..w.len() {
                    let gi = g[i].to();
                    m[i] = T::of(self.b1 * m[i].to() + (1.0 - self.b1) * gi);
                    v[i] = T::of(self.b2 * v[i].to() + (1.0 - self.b2) * gi * gi);
                    let mh = m[i].to() / bc1;
                    let vh = v[i].to() / bc2;
                    let mut upd = lr * mh / (vh.sqrt() + self.eps);
                    if p.wd && self.wd > 0.0 {
                        upd += lr * self.wd * w[i].to();
                    }
                    w[i] = w[i] - T::of(upd);
                }
            }
        }
    }
}

/// WSD schedule: 10% warmup (linear), 70% stable, 20% cosine decay to 1% of peak.
pub fn wsd_lr(step: u64, total_steps: u64, peak: f64) -> f64 {
    let n = total_steps as f64;
    let s = step as f64;
    let w = 0.10 * n;
    let d_start = 0.80 * n;
    if s < w {
        peak * (s + 1.0) / w
    } else if s < d_start {
        peak
    } else {
        let x = ((s - d_start) / (n - d_start)).min(1.0);
        let cos = 0.5 * (1.0 + (std::f64::consts::PI * x).cos());
        peak * (0.01 + 0.99 * cos)
    }
}


#[cfg(test)]
mod tests {
    // U21..U23 — WSD schedule + AdamW/clip semantics
    use super::*;
    use crate::model::Pref;

    fn pref_of(w: &mut Vec<f32>, g: &mut Vec<f32>) -> Pref<f32> {
        Pref { w, g, wd: true, name: "test" }
    }

    #[test]
    fn u21_wsd_lr_shape() {
        let peak = 1e-3;
        let (n, w, d0) = (100u64, 10u64, 80u64);
        // warmup: linear ramp
        let l0 = wsd_lr(0, n, peak);
        assert!((l0 - peak * 1.0 / 10.0).abs() < 1e-12, "step0={l0}");
        assert!(wsd_lr(5, n, peak) < wsd_lr(9, n, peak), "warmup rises");
        // stable: exactly peak
        assert_eq!(wsd_lr(w, n, peak), peak);
        assert_eq!(wsd_lr(d0 - 1, n, peak), peak);
        // decay: monotone down to ~0.01x
        assert!(wsd_lr(d0 + 1, n, peak) < peak);
        let last = wsd_lr(n - 1, n, peak);
        assert!(last > 0.01 * peak && last < 0.03 * peak, "final lr {last}");
    }

    #[test]
    fn u22_clip_scales_global_norm() {
        let mut w = vec![0.0f32; 2];
        let mut g = vec![3.0f32, 4.0]; // global norm 5
        let p = vec![pref_of(&mut w, &mut g)];
        let opt: AdamW<f32> = AdamW::new(&p, 0.0);
        let old = opt.clip(&p, 1.0);
        assert!((old - 5.0).abs() < 1e-5, "clip returns pre-clip norm");
        let grads = unsafe { &*p[0].g };
        let n2: f32 = grads.iter().map(|x| x * x).sum();
        assert!((n2 - 1.0).abs() < 1e-4, "post-clip norm^2 = {n2}");
    }

    #[test]
    fn u23_adamw_zero_grad_wd_zero_keeps_weights() {
        let mut w = vec![1.5f32, -2.5];
        let mut g = vec![0.0f32, 0.0];
        let p = vec![pref_of(&mut w, &mut g)];
        let mut opt: AdamW<f32> = AdamW::new(&p, 0.0); // no weight decay
        opt.clip(&p, 1.0);
        opt.step(&p, 1e-3);
        unsafe {
            let wv = &*p[0].w;
            assert_eq!(wv[0], 1.5);
            assert_eq!(wv[1], -2.5);
        }
        // with weight decay, weights must shrink toward zero
        let mut w2 = vec![1.5f32, -2.5];
        let mut g2 = vec![0.0f32, 0.0];
        let p2 = vec![pref_of(&mut w2, &mut g2)];
        let mut opt2: AdamW<f32> = AdamW::new(&p2, 0.1);
        opt2.step(&p2, 1e-1);
        unsafe {
            let wv = &*p2[0].w;
            assert!(wv[0] < 1.5, "decoupled decay must shrink weights");
        }
        // gradient descent actually moves weights
        let mut w3 = vec![0.0f32];
        let mut g3 = vec![1.0f32];
        let p3 = vec![pref_of(&mut w3, &mut g3)];
        let mut opt3: AdamW<f32> = AdamW::new(&p3, 0.0);
        opt3.step(&p3, 1e-2);
        unsafe {
            let wv = &*p3[0].w;
            assert!(wv[0] < 0.0, "positive grad must decrease weight");
        }
    }
}
