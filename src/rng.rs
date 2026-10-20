//! Deterministic xoshiro256++ RNG with normal sampling (no external deps).

#[derive(Clone)]
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        Self {
            s: [
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
            ],
            spare: None,
        }
    }
    #[inline]
    pub fn u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
    /// uniform [0,1)
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    /// uniform in [a,b)
    #[inline]
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f64()
    }
    /// standard normal via Box-Muller
    pub fn normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        self.spare = Some(r * th.sin());
        r * th.cos()
    }
    pub fn normal_mu(&mut self, mu: f64, sigma: f64) -> f64 {
        mu + sigma * self.normal()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.u64() % n as u64) as usize
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(0x7E55E5)
    }
}


#[cfg(test)]
mod tests {
    // U11..U14 — Rng sanity
    use super::*;

    #[test]
    fn u11_below_in_bounds_and_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            let x = a.below(7);
            assert!(x < 7);
            assert_eq!(x, b.below(7));
        }
    }

    #[test]
    fn u12_below_covers_small_range() {
        let mut r = Rng::new(1);
        let mut seen = [false; 5];
        for _ in 0..500 {
            seen[r.below(5)] = true;
        }
        assert!(seen.iter().all(|&s| s), "all of 0..5 should appear in 500 draws");
    }

    #[test]
    fn u13_f64_in_unit_interval() {
        let mut r = Rng::new(9);
        for _ in 0..1000 {
            let x = r.f64();
            assert!((0.0..1.0).contains(&x), "{x}");
        }
    }

    #[test]
    fn u14_normal_bounded_and_deterministic() {
        let mut r = Rng::new(5);
        for _ in 0..2000 {
            assert!(r.normal().abs() < 10.0, "6-sigma excursion beyond 10.0");
        }
        let mut a = Rng::new(77);
        let mut b = Rng::new(77);
        assert_eq!(a.normal(), b.normal());
    }
}
