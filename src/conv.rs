//! Pillar 2 — 1D causal depthwise highway convolution (kernel k=4) with a gated
//! temporal unit:
//!
//!   ypre_t = sum_{j=0..k-1} w_j ⊙ x_{t-j}          (depthwise, causal)
//!   h_t    = ypre_t ⊙ sigmoid(gate ⊙ x_t + bg)
//!
//! Inference state is O((k-1)·d): a small ring of recent inputs.

use crate::num::Num;

pub struct HwyConv<T: Num> {
    pub k: usize,
    pub d: usize,
    pub w: Vec<T>,    // [k][d]
    pub dw: Vec<T>,
    pub gate: Vec<T>, // [d]
    pub dgate: Vec<T>,
    pub bg: Vec<T>, // [d]
    pub dbg: Vec<T>,
    pub hist: Vec<T>, // [(k-1)][d]; hist[0..d] = x_{t-1}, hist[d..2d] = x_{t-2}, ...
}

impl<T: Num> HwyConv<T> {
    pub fn new(k: usize, d: usize) -> Self {
        Self {
            k,
            d,
            w: vec![T::ZERO; k * d],
            dw: vec![T::ZERO; k * d],
            gate: vec![T::ZERO; d],
            dgate: vec![T::ZERO; d],
            bg: vec![T::ZERO; d],
            dbg: vec![T::ZERO; d],
            hist: vec![T::ZERO; (k - 1) * d],
        }
    }
    pub fn reset(&mut self) {
        for v in self.hist.iter_mut() {
            *v = T::ZERO;
        }
    }
    #[inline]
    pub fn gate_sig(&self, x: &[T], c: usize) -> T {
        let z = self.gate[c] * x[c] + self.bg[c];
        T::ONE / (T::ONE + (-z).exp())
    }
    /// One causal step. `x` is the current input; output written to `out`.
    pub fn fwd_step(&mut self, x: &[T], out: &mut [T]) {
        let d = self.d;
        for c in 0..d {
            let mut acc = self.w[c] * x[c];
            for j in 1..self.k {
                acc = acc + self.w[j * d + c] * self.hist[(j - 1) * d + c];
            }
            let g = self.gate_sig(x, c);
            out[c] = acc * g;
        }
        // shift history: [x_{t-1},x_{t-2},..] -> [x_t, x_{t-1}, ..]
        self.hist.copy_within(0..(self.k - 2) * d, d);
        self.hist[0..d].copy_from_slice(x);
    }
    pub fn zero_grad(&mut self) {
        for v in self.dw.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dgate.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dbg.iter_mut() {
            *v = T::ZERO;
        }
    }

    /// Backward of one step given dL/d(out_t). `xt` = this step's input, `xprev[j-1]` =
    /// input at t-j (zeros for t<0). `carry[j-1]` holds dL/d(ypre_{t+j}) accumulated from
    /// already-processed later steps; must be zeroed at sequence start. Updates carry and
    /// writes dL/dx_t into `dx`.
    pub fn bwd_step(
        &mut self,
        dout: &[T],
        xt: &[T],
        xprev: &[&[T]],
        carry: &mut [T], // (k-1)*d
        dx: &mut [T],
    ) {
        let d = self.d;
        let mut dypre_t = vec![T::ZERO; d];
        for c in 0..d {
            let g = self.gate_sig(xt, c);
            let ypre = {
                let mut acc = self.w[c] * xt[c];
                for j in 1..self.k {
                    acc = acc + self.w[j * d + c] * xprev[j - 1][c];
                }
                acc
            };
            let dypre = dout[c] * g;
            dypre_t[c] = dypre;
            let dlogit = dout[c] * ypre * g * (T::ONE - g);
            // dx_t: direct (W_0·dypre_t), gate path, and contributions from later
            // steps' ypre_{t+j} (via W_j) carried in. Reads the OLD carry.
            let mut dxv = self.w[c] * dypre + dlogit * self.gate[c];
            for j in 1..self.k {
                dxv = dxv + self.w[j * d + c] * carry[(j - 1) * d + c];
            }
            dx[c] = dxv;
            // param grads
            self.dgate[c] = self.dgate[c] + xt[c] * dlogit;
            self.dbg[c] = self.dbg[c] + dlogit;
            self.dw[c] = self.dw[c] + xt[c] * dypre;
            for j in 1..self.k {
                self.dw[j * d + c] = self.dw[j * d + c] + xprev[j - 1][c] * dypre;
            }
        }
        // shift the old carry up, then push this step's dypre:
        // carry = [dypre_t, dypre_{t+1}, dypre_{t+2}] for the next (earlier) step
        for j in (1..self.k - 1).rev() {
            let (src, dst) = carry.split_at_mut(j * d);
            dst[0..d].copy_from_slice(&src[(j - 1) * d..j * d]);
        }
        carry[0..d].copy_from_slice(&dypre_t);
    }
}
