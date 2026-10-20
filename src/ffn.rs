//! Pillar 7 — SwiGLU FFN with 6x expansion, plus a parallel low-rank adapter (LoRA, r=8):
//!
//!   ff(x)  = W_down ( silu(W_gate x) ⊙ (W_up x) )
//!   lora(x)= W_b (W_a x),   W_b init = 0
//!   y      = ff(x) + lora(x)

use crate::num::Num;

pub struct SwiGLU<T: Num> {
    pub d: usize,
    pub f: usize,
    pub r: usize,
    pub wg: Vec<T>, // [d][f]
    pub dwg: Vec<T>,
    pub wu: Vec<T>, // [d][f]
    pub dwu: Vec<T>,
    pub wd: Vec<T>, // [f][d]
    pub dwd: Vec<T>,
    pub la: Vec<T>, // [d][r]
    pub dla: Vec<T>,
    pub lb: Vec<T>, // [r][d]
    pub dlb: Vec<T>,
}

impl<T: Num> SwiGLU<T> {
    pub fn new(d: usize, f: usize, r: usize) -> Self {
        Self {
            d,
            f,
            r,
            wg: vec![T::ZERO; d * f],
            dwg: vec![T::ZERO; d * f],
            wu: vec![T::ZERO; d * f],
            dwu: vec![T::ZERO; d * f],
            wd: vec![T::ZERO; f * d],
            dwd: vec![T::ZERO; f * d],
            la: vec![T::ZERO; d * r],
            dla: vec![T::ZERO; d * r],
            lb: vec![T::ZERO; r * d],
            dlb: vec![T::ZERO; r * d],
        }
    }
    pub fn zero_grad(&mut self) {
        for g in [&mut self.dwg, &mut self.dwu, &mut self.dwd, &mut self.dla, &mut self.dlb] {
            for v in g.iter_mut() {
                *v = T::ZERO;
            }
        }
    }

    /// y = ff(x) + lora(x). Caches the LoRA activation `a = la x` (r floats).
    pub fn fwd(&self, x: &[T], y: &mut [T], a_out: &mut [T]) {
        let (d, f, r) = (self.d, self.f, self.r);
        let mut hg = vec![T::ZERO; f];
        let mut hu = vec![T::ZERO; f];
        T::gemv(&self.wg, x, &mut hg, d, f);
        T::gemv(&self.wu, x, &mut hu, d, f);
        // silu + gate
        for j in 0..f {
            let sg = T::ONE / (T::ONE + (-hg[j]).exp());
            hg[j] = hg[j] * sg * hu[j];
        }
        T::gemv(&self.wd, &hg, y, f, d);
        T::gemv(&self.la, x, a_out, d, r);
        let mut lo = vec![T::ZERO; d];
        T::gemv(&self.lb, a_out, &mut lo, r, d);
        for i in 0..d {
            y[i] = y[i] + lo[i];
        }
    }

    /// Backward: given dL/dy, accumulate parameter grads and write dL/dx into `dx`.
    /// `a` = cached LoRA activation from forward.
    pub fn bwd(&mut self, x: &[T], a: &[T], dy: &[T], dx: &mut [T]) {
        let (d, f, r) = (self.d, self.f, self.r);
        // recompute activations (cheap: 2 gemv)
        let mut hg = vec![T::ZERO; f];
        let mut hu = vec![T::ZERO; f];
        T::gemv(&self.wg, x, &mut hg, d, f);
        T::gemv(&self.wu, x, &mut hu, d, f);
        // dL/d(hg·σ(hg)⊙hu) chain
        // dh[j] = sum_o wd[j][o] dy[o]
        let mut dh = vec![T::ZERO; f];
        T::gemv_dx(&self.wd, dy, &mut dh, f, d);
        // dWd[j][o] += s[j]*hu[j] * dy[o]
        let mut s = vec![T::ZERO; f];
        for j in 0..f {
            let sg = T::ONE / (T::ONE + (-hg[j]).exp());
            s[j] = hg[j] * sg;
        }
        {
            let mut shu = vec![T::ZERO; f];
            for j in 0..f {
                shu[j] = s[j] * hu[j];
            }
            T::gemv_dw(&shu, dy, &mut self.dwd, f, d);
        }
        // h = silu(hg) ⊙ hu  ⇒  ds = dh⊙hu ; dhu = dh⊙silu(hg) ; dhg = ds⊙silu'(hg)
        let mut ds = vec![T::ZERO; f];
        for j in 0..f {
            ds[j] = dh[j] * hu[j];
        }
        let mut dhg = vec![T::ZERO; f];
        let mut dhu = vec![T::ZERO; f];
        for j in 0..f {
            let sg = T::ONE / (T::ONE + (-hg[j]).exp());
            let silu_p = sg * (T::ONE + hg[j] * (T::ONE - sg));
            dhg[j] = ds[j] * silu_p;
            dhu[j] = dh[j] * s[j];
        }
        T::gemv_dw(x, &dhg, &mut self.dwg, d, f);
        T::gemv_dw(x, &dhu, &mut self.dwu, d, f);
        T::gemv_dx(&self.wg, &dhg, dx, d, f);
        let mut dx2 = vec![T::ZERO; d];
        T::gemv_dx(&self.wu, &dhu, &mut dx2, d, f);
        T::vadd(dx, &dx2);
        // LoRA branch
        let mut dl2 = vec![T::ZERO; r];
        T::gemv_dx(&self.lb, dy, &mut dl2, r, d);
        T::gemv_dw(a, dy, &mut self.dlb, r, d);
        T::gemv_dw(x, &dl2, &mut self.dla, d, r);
        let mut dx3 = vec![T::ZERO; d];
        T::gemv_dx(&self.la, &dl2, &mut dx3, d, r);
        T::vadd(dx, &dx3);
    }
}
