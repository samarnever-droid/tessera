//! Pillar 1 — Affine RMSNorm with safe, continuous, non-branching normalization:
//!     s = max(||v||_2, 1e-8),   xhat = sqrt(d) * v / s,   y = gamma ⊙ xhat
//! Pillar 3 — Per-head QK-RMSNorm on dsub-dim chunks (bounds |q·k| <= dsub).

use crate::num::Num;

#[inline]
pub fn safe_inv_norm<T: Num>(v: &[T]) -> (T, T) {
    let n = T::dot(v, v).sqrt();
    let s = n.mx(T::of(1e-8));
    (s, T::ONE / s)
}

/// y = gamma ⊙ (sqrt(d) x / max(||x||,1e-8)); also emits xhat and s for backward.
pub fn rmsnorm_fwd<T: Num>(x: &[T], gamma: &[T], out: &mut [T], xhat: &mut [T]) -> T {
    let d = x.len();
    let (s, inv) = safe_inv_norm(x);
    let sq = T::of((d as f64).sqrt()) * inv;
    for i in 0..d {
        xhat[i] = x[i] * sq;
        out[i] = xhat[i] * gamma[i];
    }
    s
}

/// dx = (sqrt(d) (gamma⊙g) - xhat <gamma⊙g, xhat>) / s ;  dgamma += g ⊙ xhat
pub fn rmsnorm_bwd<T: Num>(g: &[T], gamma: &[T], xhat: &[T], s: T, dx: &mut [T], dgamma: &mut [T]) {
    let d = xhat.len();
    let mut c = vec![T::ZERO; d];
    for i in 0..d {
        c[i] = g[i] * gamma[i];
    }
    let cd = T::dot(&c, xhat);
    let sq = T::of((d as f64).sqrt());
    for i in 0..d {
        // dx = (sqrt(d)·c − x̂·⟨c,x̂⟩/sqrt(d)) / s
        dx[i] = (sq * c[i] - xhat[i] * cd / sq) / s;
        dgamma[i] = dgamma[i] + g[i] * xhat[i];
    }
}

/// Per-chunk RMSNorm in place over `nchunks` chunks of `dsub` dims, shared gamma.
pub fn qknorm_fwd<T: Num>(v: &mut [T], gamma: &[T], nchunks: usize, dsub: usize) {
    for c in 0..nchunks {
        let (s, inv) = safe_inv_norm(&v[c * dsub..(c + 1) * dsub]);
        let sq = T::of((dsub as f64).sqrt()) * inv;
        for i in 0..dsub {
            let j = c * dsub + i;
            v[j] = v[j] * sq * gamma[i];
        }
        let _ = s;
    }
}

/// Backward of `qknorm_fwd` for one chunk: given g (grad wrt normed chunk) and the raw
/// (pre-norm) chunk, produce d(raw) and accumulate dgamma. Returns d(raw) in `dx`.
pub fn qknorm_bwd_chunk<T: Num>(g: &[T], raw: &[T], gamma: &[T], dx: &mut [T], dgamma: &mut [T], dsub: usize) {
    // xhat_i = sqrt(n) raw_i / ||raw|| ; dx = (sqrt(n) (gamma⊙g) - xhat <gamma⊙g, xhat>)/s
    let (s, _inv) = safe_inv_norm(raw);
    let sq = T::of((dsub as f64).sqrt());
    let mut xhat = vec![T::ZERO; dsub];
    let inv = T::ONE / s;
    for i in 0..dsub {
        xhat[i] = raw[i] * sq * inv;
    }
    rmsnorm_bwd(g, gamma, &xhat, s, dx, dgamma);
}


#[cfg(test)]
mod tests {
    // U19..U20 — norm floors and unit-norm behavior
    use super::*;

    #[test]
    fn u19_rmsnorm_zero_vector_stays_finite() {
        let d = 8;
        let x = vec![0.0f32; d];
        let gamma = vec![1.0f32; d];
        let mut out = vec![0.0f32; d];
        let mut xhat = vec![0.0f32; d];
        let s = rmsnorm_fwd(&x, &gamma, &mut out, &mut xhat);
        assert!(s.is_finite());
        assert!(out.iter().all(|v| v.is_finite() && *v == 0.0), "zero in -> zero out");
        // large vector does not overflow
        let big = vec![1e18f32; d];
        let mut out2 = vec![0.0f32; d];
        let mut xhat2 = vec![0.0f32; d];
        let s2 = rmsnorm_fwd(&big, &gamma, &mut out2, &mut xhat2);
        assert!(s2.is_finite() && out2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn u20_qknorm_unit_norm_per_chunk() {
        let nchunks = 3;
        let dsub = 4;
        let mut v: Vec<f32> = (0..nchunks * dsub).map(|i| (i as f32) * 0.37 - 2.0).collect();
        let gamma = vec![1.0f32; dsub];
        qknorm_fwd(&mut v, &gamma, nchunks, dsub);
        for c in 0..nchunks {
            // qknorm scales to RMS units: ||chunk|| = sqrt(dsub) -> |q.k| <= dsub
            let n2: f32 = v[c * dsub..(c + 1) * dsub].iter().map(|x| x * x).sum();
            assert!((n2 - dsub as f32).abs() < 1e-2, "chunk {c} norm^2 = {n2}");
        }
    }
}
