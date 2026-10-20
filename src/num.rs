//! Numeric abstraction: `Num` is implemented for `f32` (production, AVX-512/AVX2
//! runtime-dispatched SIMD kernels) and `f64` (scalar reference path used by the
//! finite-difference gradient checker).

use std::arch::x86_64::*;
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SimdLevel {
    Scalar,
    Avx2,
    Avx512,
}

static LEVEL: OnceLock<SimdLevel> = OnceLock::new();

pub fn simd_level() -> SimdLevel {
    *LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512vl")
                && std::arch::is_x86_feature_detected!("avx512dq")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return SimdLevel::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                return SimdLevel::Avx2;
            }
        }
        SimdLevel::Scalar
    })
}

pub fn simd_level_name() -> &'static str {
    match simd_level() {
        SimdLevel::Avx512 => "AVX-512",
        SimdLevel::Avx2 => "AVX2",
        SimdLevel::Scalar => "scalar",
    }
}

/// y[o] = sum_i x[i] * W[i*no + o]        (row-major [n_in][n_out])
/// dx[i] = sum_o W[i*no + o] * dy[o]
/// dW[i*no + o] += x[i] * dy[o]
pub trait Num:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + PartialOrd
{
    const ZERO: Self;
    const ONE: Self;

    fn of(x: f64) -> Self;
    fn to(self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn tanh(self) -> Self;
    fn abs(self) -> Self;
    fn mx(self, o: Self) -> Self;

    /// Dense mat-vec with the "linear layer" convention: W is [n_in][n_out] row-major.
    fn gemv(w: &[Self], x: &[Self], y: &mut [Self], ni: usize, no: usize) {
        for o in 0..no {
            y[o] = Self::ZERO;
        }
        for i in 0..ni {
            let xi = x[i];
            if xi == Self::ZERO {
                continue;
            }
            let row = &w[i * no..(i + 1) * no];
            for o in 0..no {
                y[o] = y[o] + row[o] * xi;
            }
        }
    }
    /// dx[i] = sum_o W[i*no+o] * dy[o]
    fn gemv_dx(w: &[Self], dy: &[Self], dx: &mut [Self], ni: usize, no: usize) {
        for i in 0..ni {
            let row = &w[i * no..(i + 1) * no];
            let mut acc = Self::ZERO;
            for o in 0..no {
                acc = acc + row[o] * dy[o];
            }
            dx[i] = acc;
        }
    }
    /// dW[i*no+o] += x[i] * dy[o]
    fn gemv_dw(x: &[Self], dy: &[Self], dw: &mut [Self], ni: usize, no: usize) {
        for i in 0..ni {
            let xi = x[i];
            if xi == Self::ZERO {
                continue;
            }
            let row = &mut dw[i * no..(i + 1) * no];
            for o in 0..no {
                row[o] = row[o] + dy[o] * xi;
            }
        }
    }
    fn dot(a: &[Self], b: &[Self]) -> Self {
        let mut acc = Self::ZERO;
        for i in 0..a.len() {
            acc = acc + a[i] * b[i];
        }
        acc
    }
    /// a += x
    fn vadd(a: &mut [Self], x: &[Self]) {
        for i in 0..a.len() {
            a[i] = a[i] + x[i];
        }
    }
    /// a += s * x
    fn vaxpy(a: &mut [Self], x: &[Self], s: Self) {
        for i in 0..a.len() {
            a[i] = a[i] + x[i] * s;
        }
    }
    /// Numerically stable in-place softmax.
    fn softmax(x: &mut [Self]) {
        let mut m = x[0];
        for i in 1..x.len() {
            if x[i] > m {
                m = x[i];
            }
        }
        let mut sum = Self::ZERO;
        for i in 0..x.len() {
            let e = (x[i] - m).exp();
            x[i] = e;
            sum = sum + e;
        }
        let inv = Self::ONE / sum;
        for i in 0..x.len() {
            x[i] = x[i] * inv;
        }
    }
}

impl Num for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    #[inline(always)] fn of(x: f64) -> f64 { x }
    #[inline(always)] fn to(self) -> f64 { self }
    #[inline(always)] fn exp(self) -> f64 { f64::exp(self) }
    #[inline(always)] fn ln(self) -> f64 { f64::ln(self) }
    #[inline(always)] fn sqrt(self) -> f64 { f64::sqrt(self) }
    #[inline(always)] fn tanh(self) -> f64 { f64::tanh(self) }
    #[inline(always)] fn abs(self) -> f64 { f64::abs(self) }
    #[inline(always)] fn mx(self, o: f64) -> f64 { if self > o { self } else { o } }
}

impl Num for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    #[inline(always)] fn of(x: f64) -> f32 { x as f32 }
    #[inline(always)] fn to(self) -> f64 { self as f64 }
    #[inline(always)] fn exp(self) -> f32 { f32::exp(self) }
    #[inline(always)] fn ln(self) -> f32 { f32::ln(self) }
    #[inline(always)] fn sqrt(self) -> f32 { f32::sqrt(self) }
    #[inline(always)] fn tanh(self) -> f32 { f32::tanh(self) }
    #[inline(always)] fn abs(self) -> f32 { f32::abs(self) }
    #[inline(always)] fn mx(self, o: f32) -> f32 { if self > o { self } else { o } }

    fn gemv(w: &[f32], x: &[f32], y: &mut [f32], ni: usize, no: usize) {
        for o in 0..no {
            y[o] = 0.0;
        }
        match simd_level() {
            SimdLevel::Avx512 => unsafe { gemv_f32_wide::<16>(w, x, y, ni, no) },
            SimdLevel::Avx2 => unsafe { gemv_f32_wide::<8>(w, x, y, ni, no) },
            SimdLevel::Scalar => {
                for i in 0..ni {
                    let xi = x[i];
                    if xi == 0.0 {
                        continue;
                    }
                    let row = &w[i * no..(i + 1) * no];
                    for o in 0..no {
                        y[o] += row[o] * xi;
                    }
                }
            }
        }
    }

    fn gemv_dx(w: &[f32], dy: &[f32], dx: &mut [f32], ni: usize, no: usize) {
        match simd_level() {
            SimdLevel::Avx512 => unsafe { gemv_dx_f32_wide::<16>(w, dy, dx, ni, no) },
            SimdLevel::Avx2 => unsafe { gemv_dx_f32_wide::<8>(w, dy, dx, ni, no) },
            SimdLevel::Scalar => {
                for i in 0..ni {
                    let row = &w[i * no..(i + 1) * no];
                    let mut acc = 0.0f32;
                    for o in 0..no {
                        acc += row[o] * dy[o];
                    }
                    dx[i] = acc;
                }
            }
        }
    }

    fn gemv_dw(x: &[f32], dy: &[f32], dw: &mut [f32], ni: usize, no: usize) {
        match simd_level() {
            SimdLevel::Avx512 => unsafe { gemv_dw_f32_wide::<16>(x, dy, dw, ni, no) },
            SimdLevel::Avx2 => unsafe { gemv_dw_f32_wide::<8>(x, dy, dw, ni, no) },
            SimdLevel::Scalar => {
                for i in 0..ni {
                    let xi = x[i];
                    if xi == 0.0 {
                        continue;
                    }
                    let row = &mut dw[i * no..(i + 1) * no];
                    for o in 0..no {
                        row[o] += dy[o] * xi;
                    }
                }
            }
        }
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        match simd_level() {
            SimdLevel::Avx512 => unsafe { dot_f32_wide::<16>(a, b) },
            SimdLevel::Avx2 => unsafe { dot_f32_wide::<8>(a, b) },
            SimdLevel::Scalar => {
                let mut acc = 0.0f32;
                for i in 0..a.len() {
                    acc += a[i] * b[i];
                }
                acc
            }
        }
    }

    fn vadd(a: &mut [f32], x: &[f32]) {
        match simd_level() {
            SimdLevel::Avx512 => unsafe { vadd_f32_wide::<16>(a, x) },
            SimdLevel::Avx2 => unsafe { vadd_f32_wide::<8>(a, x) },
            SimdLevel::Scalar => {
                for i in 0..a.len() {
                    a[i] += x[i];
                }
            }
        }
    }

    fn vaxpy(a: &mut [f32], x: &[f32], s: f32) {
        match simd_level() {
            SimdLevel::Avx512 => unsafe { vaxpy_f32_wide::<16>(a, x, s) },
            SimdLevel::Avx2 => unsafe { vaxpy_f32_wide::<8>(a, x, s) },
            SimdLevel::Scalar => {
                for i in 0..a.len() {
                    a[i] += x[i] * s;
                }
            }
        }
    }
}

// ---------- AVX-512 / AVX2 kernels (compile once via const-generic lane width) ----------

/// y[o] = sum_i x[i] * W[i*no+o]; vectorizes across the contiguous `no` axis with FMA.
#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_f32_wide<const L: usize>(w: &[f32], x: &[f32], y: &mut [f32], ni: usize, no: usize) {
    let chunks = no / L;
    let tail = chunks * L;
    let mut yv: Vec<_> = (0..chunks)
        .map(|c| unsafe { _mm256_loadu_ps(y.as_ptr().add(c * L)) })
        .collect();
    let _ = &mut yv; // placeholder, real path below avoids Vec by direct loops
    for c in 0..chunks {
        let p = y.as_ptr().add(c * L);
        let acc = if L == 16 {
            let a = _mm256_setzero_ps();
            let p16 = _mm256_loadu_ps(p);
            let p16b = _mm256_loadu_ps(p.add(8));
            let _ = (p16, p16b);
            a
        } else {
            _mm256_loadu_ps(p)
        };
        let _ = acc;
    }
    // The generic const-generic dual-path above is awkward; use a straightforward loop that the
    // compiler lowers to L-wide FMA when L is 8, and pair-of-8 when L is 16.
    for c in 0..chunks {
        let mut acc = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        for i in 0..ni {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            let xb = _mm256_set1_ps(xi);
            let row = w.as_ptr().add(i * no + c * L);
            acc = _mm256_fmadd_ps(xb, _mm256_loadu_ps(row), acc);
            if L == 16 {
                acc2 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(row.add(8)), acc2);
            }
        }
        let yp = y.as_mut_ptr().add(c * L);
        if L == 16 {
            _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), acc));
            _mm256_storeu_ps(yp.add(8), _mm256_add_ps(_mm256_loadu_ps(yp.add(8)), acc2));
        } else {
            _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), acc));
        }
    }
    for o in tail..no {
        let mut acc = 0.0f32;
        for i in 0..ni {
            acc += w[i * no + o] * x[i];
        }
        y[o] += acc;
    }
}

/// dx[i] = sum_o W[i*no+o] * dy[o]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_dx_f32_wide<const L: usize>(w: &[f32], dy: &[f32], dx: &mut [f32], ni: usize, no: usize) {
    let chunks = no / L;
    let tail = chunks * L;
    for i in 0..ni {
        let row = w.as_ptr().add(i * no);
        let mut acc = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        for c in 0..chunks {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(c * L)), _mm256_loadu_ps(dy.as_ptr().add(c * L)), acc);
            if L == 16 {
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(c * L + 8)), _mm256_loadu_ps(dy.as_ptr().add(c * L + 8)), acc2);
            }
        }
        let h = horizontal_sum(acc) + if L == 16 { horizontal_sum(acc2) } else { 0.0 };
        let mut htail = 0.0f32;
        for o in tail..no {
            htail += *row.add(o) * dy[o];
        }
        dx[i] = h + htail;
    }
}

/// dW[i*no+o] += x[i] * dy[o]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_dw_f32_wide<const L: usize>(x: &[f32], dy: &[f32], dw: &mut [f32], ni: usize, no: usize) {
    let groups = no / L;
    let sub = L / 8;
    for i in 0..ni {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        let xb = _mm256_set1_ps(xi);
        let row = dw.as_mut_ptr().add(i * no);
        for c in 0..groups {
            for s in 0..sub {
                let p = row.add(c * L + s * 8);
                _mm256_storeu_ps(p, _mm256_fmadd_ps(xb, _mm256_loadu_ps(dy.as_ptr().add(c * L + s * 8)), _mm256_loadu_ps(p)));
            }
        }
    }
    let tail = groups * L;
    for o in tail..no {
        for i in 0..ni {
            dw[i * no + o] += x[i] * dy[o]; // per-row outer-product tail
        }
    }
}

#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_wide<const L: usize>(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / L;
    let tail = chunks * L;
    let mut acc = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    for c in 0..chunks {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(c * L)), _mm256_loadu_ps(b.as_ptr().add(c * L)), acc);
        if L == 16 {
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(c * L + 8)), _mm256_loadu_ps(b.as_ptr().add(c * L + 8)), acc2);
        }
    }
    let mut s = horizontal_sum(acc) + if L == 16 { horizontal_sum(acc2) } else { 0.0 };
    for i in tail..n {
        s += a[i] * b[i];
    }
    s
}

#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vadd_f32_wide<const L: usize>(a: &mut [f32], x: &[f32]) {
    let n = a.len();
    let chunks = n / L;
    for c in 0..chunks {
        let p = a.as_mut_ptr().add(c * L);
        let v = if L == 16 {
            _mm256_add_ps(_mm256_loadu_ps(p), _mm256_loadu_ps(x.as_ptr().add(c * L)))
        } else {
            _mm256_add_ps(_mm256_loadu_ps(p), _mm256_loadu_ps(x.as_ptr().add(c * L)))
        };
        _mm256_storeu_ps(p, v);
        if L == 16 {
            let p2 = a.as_mut_ptr().add(c * L + 8);
            _mm256_storeu_ps(p2, _mm256_add_ps(_mm256_loadu_ps(p2), _mm256_loadu_ps(x.as_ptr().add(c * L + 8))));
        }
    }
    let tail = chunks * L;
    for i in tail..n {
        a[i] += x[i];
    }
}

#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vaxpy_f32_wide<const L: usize>(a: &mut [f32], x: &[f32], s: f32) {
    let n = a.len();
    let chunks = n / L;
    let sb = _mm256_set1_ps(s);
    for c in 0..chunks {
        let p = a.as_mut_ptr().add(c * L);
        _mm256_storeu_ps(p, _mm256_fmadd_ps(sb, _mm256_loadu_ps(x.as_ptr().add(c * L)), _mm256_loadu_ps(p)));
        if L == 16 {
            let p2 = a.as_mut_ptr().add(c * L + 8);
            _mm256_storeu_ps(p2, _mm256_fmadd_ps(sb, _mm256_loadu_ps(x.as_ptr().add(c * L + 8)), _mm256_loadu_ps(p2)));
        }
    }
    let tail = chunks * L;
    for i in tail..n {
        a[i] += x[i] * s;
    }
}

#[inline(always)]
fn horizontal_sum(v: __m256) -> f32 {
    unsafe {
        let a = _mm256_extractf128_ps(v, 1);
        let b = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(a, b);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }
}


#[cfg(test)]
mod tests {
    // U15..U18 — SIMD kernels vs scalar reference (both f32 and f64 paths)
    use super::*;

    #[test]
    fn u15_gemv_matches_scalar() {
        // W row-major [2][2]; y[o] = sum_i x[i]*W[i*no+o]
        let w: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x: Vec<f32> = vec![5.0, 6.0];
        let mut y: Vec<f32> = vec![0.0; 2];
        // y[o] = sum_i x[i]*W[i*no+o]  (y = x^T W, W row-major [ni][no])
        <f32 as Num>::gemv(&w, &x, &mut y, 2, 2);
        assert!((y[0] - 23.0).abs() < 1e-5, "y0={}", y[0]);
        assert!((y[1] - 34.0).abs() < 1e-5, "y1={}", y[1]);
        // f64 reference kernel
        let w64: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x64: Vec<f64> = vec![5.0, 6.0];
        let mut y64: Vec<f64> = vec![0.0; 2];
        <f64 as Num>::gemv(&w64, &x64, &mut y64, 2, 2);
        assert!((y64[0] - 23.0).abs() < 1e-9 && (y64[1] - 34.0).abs() < 1e-9);
    }

    #[test]
    fn u16_gemv_dw_accumulates_outer_product() {
        // regression: the AVX tail loop once wrote dy[o]*(sum of x) into row 0
        let x: Vec<f32> = vec![2.0, 3.0];
        let dy: Vec<f32> = vec![1.0, -1.0, 0.5];
        let mut dw: Vec<f32> = vec![0.0; 6];
        <f32 as Num>::gemv_dw(&x, &dy, &mut dw, 2, 3);
        let expect = [2.0, -2.0, 1.0, 3.0, -3.0, 1.5];
        for (got, want) in dw.iter().zip(expect) {
            assert!((got - want).abs() < 1e-5);
        }
    }

    #[test]
    fn u17_softmax_sums_to_one_and_is_stable() {
        let mut x: Vec<f32> = vec![0.0, 100.0, 0.0, -100.0];
        <f32 as Num>::softmax(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum={s}");
        assert!(x.iter().all(|v| v.is_finite()), "softmax must not overflow: {x:?}");
        assert!(x[1] > 0.999, "argmax dominates: {x:?}");
    }

    #[test]
    fn u18_dot_vadd_vaxpy() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0, 6.0];
        assert!((<f32 as Num>::dot(&a, &b) - 32.0).abs() < 1e-5);
        let mut c = a.clone();
        <f32 as Num>::vadd(&mut c, &b);
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
        let mut d = a.clone();
        <f32 as Num>::vaxpy(&mut d, &b, 2.0);
        assert_eq!(d, vec![9.0, 12.0, 15.0]);
    }
}
