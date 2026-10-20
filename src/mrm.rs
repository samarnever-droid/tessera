//! Pillar 8 — Multi-Resolution Working Memory (MRM-v2): a continuous Modern Hopfield
//! substrate with 128 fine token slots + 16 coarse centroids, dimension-adaptive
//! overwrite/merge thresholds, a 3-tier state machine, Riemannian (Poincaré-ball)
//! soft merges, utility-based eviction, and exact BPTT via per-step `MrmSnapshot`s.
//!
//! Thresholds (d = key dim):
//!   t_overwrite(d) = min(0.9999, 8.014/√d)
//!   t_merge(d)     = min(t_overwrite − 1e−4, 6.50/√d)
//!
//! Tier 1: s ≥ t_overwrite  -> hard overwrite (exact entity update)
//! Tier 2: t_merge ≤ s < t_overwrite -> soft merge with Poincaré Riemannian EMA
//!         ∇_R = ((1−‖c‖²)²/4)·∇_E ;  c ← c + (1−γ)·∇_R   (∇_E = k̂_new − c)
//! Tier 3: s < t_merge -> new slot insert; on full memory evict argmin Utility(i)
//!         Utility(i) = 2·hits[i] + S_i,  S_i = write surprise
//!
//! Keys are kept inside the open unit ball (projected to norm `ball` = 0.999) so the
//! Poincaré conformal factor (1−‖c‖²)²/4 is always well-defined. Control decisions
//! (argmax / tier choice / eviction) are discrete; gradients flow through the recorded
//! continuous ops exactly, in reverse, using the snapshot tape.

use crate::num::Num;

#[derive(Clone, Copy, Debug)]
pub struct MrmCfg {
    pub d: usize,
    pub fine: usize,
    pub coarse: usize,
    pub gamma_merge: f64,
    pub gamma_coarse: f64,
    pub ball: f64,
    pub gate_threshold: f64,
}

impl Default for MrmCfg {
    fn default() -> Self {
        Self { d: 64, fine: 128, coarse: 16, gamma_merge: 0.5, gamma_coarse: 0.3, ball: 0.999, gate_threshold: 0.5 }
    }
}

impl MrmCfg {
    #[inline]
    pub fn thresholds(&self) -> (f64, f64) {
        let sq = (self.d as f64).sqrt();
        let t_ovw = 0.9999_f64.min(8.014 / sq);
        let t_mrg = (t_ovw - 1e-4).min(6.50 / sq);
        (t_ovw, t_mrg)
    }
}

/// Discrete op chosen at a write. op: 0 = overwrite, 1 = merge, 2 = insert/evict.
#[derive(Clone)]
pub struct MrmSnapshot<T: Num> {
    pub op: u8,
    pub i: usize,
    pub gate: f64, // post-sigmoid gate (cached for exact backward)
    pub prev_k: Vec<T>,
    pub prev_v: Vec<T>,
    pub prev_ts: u32,
    pub prev_hits: u32,
    pub prev_sur: T,
    pub prev_alive: bool,
    pub cidx: usize,
    pub prev_ck: Vec<T>,
    pub prev_cv: Vec<T>,
    pub prev_calive: bool,
}

/// Gradient buffers routed through the tape (dL/d slot-state at the current time).
pub struct MrmGrads<T: Num> {
    pub gk: Vec<T>,  // fine × d
    pub gv: Vec<T>,
    pub gck: Vec<T>, // coarse × d
    pub gcv: Vec<T>,
}

pub struct Mrm<T: Num> {
    pub cfg: MrmCfg,
    pub k: Vec<T>,
    pub v: Vec<T>,
    pub hits: Vec<u32>,
    pub sur: Vec<T>,
    pub ts: Vec<u32>,
    pub alive: Vec<bool>,
    pub ck: Vec<T>,
    pub cv: Vec<T>,
    pub calive: Vec<bool>,
    pub step: u64,
}

#[inline]
fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl<T: Num> Mrm<T> {
    pub fn new(cfg: MrmCfg) -> Self {
        Self {
            k: vec![T::ZERO; cfg.fine * cfg.d],
            v: vec![T::ZERO; cfg.fine * cfg.d],
            hits: vec![0; cfg.fine],
            sur: vec![T::ZERO; cfg.fine],
            ts: vec![0; cfg.fine],
            alive: vec![false; cfg.fine],
            ck: vec![T::ZERO; cfg.coarse * cfg.d],
            cv: vec![T::ZERO; cfg.coarse * cfg.d],
            calive: vec![false; cfg.coarse],
            cfg,
            step: 0,
        }
    }
    pub fn reset(&mut self) {
        let c = &self.cfg;
        self.k = vec![T::ZERO; c.fine * c.d];
        self.v = vec![T::ZERO; c.fine * c.d];
        self.hits = vec![0; c.fine];
        self.sur = vec![T::ZERO; c.fine];
        self.ts = vec![0; c.fine];
        self.alive = vec![false; c.fine];
        self.ck = vec![T::ZERO; c.coarse * c.d];
        self.cv = vec![T::ZERO; c.coarse * c.d];
        self.calive = vec![false; c.coarse];
        self.step = 0;
    }
    pub fn n_read(&self) -> usize {
        self.cfg.fine + self.cfg.coarse
    }
    /// Slots substrate bytes (the "~145 KB" constant): K_fine + V_fine + K_coarse + V_coarse.
    pub fn substrate_bytes(&self) -> usize {
        let c = &self.cfg;
        (c.fine + c.coarse) * c.d * 2 * std::mem::size_of::<T>()
    }
    /// Substrate + all bookkeeping (hits/surprise/timestamps/alive + counters).
    pub fn state_bytes(&self) -> usize {
        let c = &self.cfg;
        self.substrate_bytes()
            + c.fine * (4 + std::mem::size_of::<T>() + 4 + 1)
            + c.coarse * 1
            + 8
    }

    /// Continuous key direction for control + attention content key of slot `i`.
    #[inline]
    fn cos_to(&self, kh: &[T], i: usize) -> f64 {
        let d = self.cfg.d;
        let kn = T::dot(&self.k[i * d..(i + 1) * d], &self.k[i * d..(i + 1) * d]).to().sqrt();
        T::dot(kh, &self.k[i * d..(i + 1) * d]).to() / kn.max(1e-8)
    }

    /// One causal write. Returns a snapshot when `train` and the write happened.
    /// `k_raw`, `v_raw` are the proposed key/value; `glin` the gate logit; `r_ctx` the
    /// current readout (used as the surprise reference on fresh inserts).
    pub fn fwd_write(
        &mut self,
        k_raw: &[T],
        v_raw: &[T],
        glin: T,
        r_ctx: &[T],
        train: bool,
    ) -> Option<MrmSnapshot<T>> {
        let c = self.cfg;
        let d = c.d;
        self.step += 1;
        let t = self.step as u32;
        let gate = sig(glin.to());
        if gate <= c.gate_threshold {
            return None; // no write: state unchanged, backward is pure copy-through
        }
        let kn = T::dot(k_raw, k_raw).to().sqrt();
        if kn < 1e-6 {
            return None; // degenerate (near-zero) key: skip write, keep grads finite
        }
        let kh: Vec<T> = k_raw.iter().map(|x| *x * T::of(1.0 / kn)).collect();
        let v_w: Vec<T> = v_raw.iter().map(|x| *x * T::of(gate)).collect();

        // ---- similarity over alive fine slots (content keys, no positional rotation) ----
        let mut best = -2.0f64;
        let mut bi = 0usize;
        let mut any_alive = false;
        for i in 0..c.fine {
            if self.alive[i] {
                any_alive = true;
                let s = self.cos_to(&kh, i);
                if s > best {
                    best = s;
                    bi = i;
                }
            }
        }
        let (t_ovw, t_mrg) = c.thresholds();
        let op = if !any_alive {
            2u8
        } else if best >= t_ovw {
            0u8
        } else if best >= t_mrg {
            1u8
        } else {
            2u8
        };

        let mut snap = MrmSnapshot {
            op,
            i: bi,
            gate,
            prev_k: vec![T::ZERO; d],
            prev_v: vec![T::ZERO; d],
            prev_ts: 0,
            prev_hits: 0,
            prev_sur: T::ZERO,
            prev_alive: false,
            cidx: 0,
            prev_ck: vec![T::ZERO; d],
            prev_cv: vec![T::ZERO; d],
            prev_calive: false,
        };

        let kstore: Vec<T> = kh.iter().map(|x| *x * T::of(c.ball)).collect();

        match op {
            0 => {
                // Tier 1: hard overwrite (exact entity update)
                let o = bi * d;
                snap.prev_k.copy_from_slice(&self.k[o..o + d]);
                snap.prev_v.copy_from_slice(&self.v[o..o + d]);
                snap.prev_ts = self.ts[bi];
                snap.prev_hits = self.hits[bi];
                snap.prev_sur = self.sur[bi];
                snap.prev_alive = self.alive[bi];
                self.sur[bi] = {
                    let mut ds = T::ZERO;
                    for j in 0..d {
                        let e = v_w[j] - snap.prev_v[j];
                        ds = ds + e * e;
                    }
                    T::of(ds.to().sqrt())
                };
                self.k[o..o + d].copy_from_slice(&kstore);
                self.v[o..o + d].copy_from_slice(&v_w);
                self.hits[bi] += 1;
                self.ts[bi] = t;
            }
            1 => {
                // Tier 2: soft merge — Poincaré ball Riemannian EMA
                let o = bi * d;
                snap.prev_k.copy_from_slice(&self.k[o..o + d]);
                snap.prev_v.copy_from_slice(&self.v[o..o + d]);
                snap.prev_ts = self.ts[bi];
                snap.prev_hits = self.hits[bi];
                snap.prev_sur = self.sur[bi];
                snap.prev_alive = self.alive[bi];
                let rho2 = T::dot(&snap.prev_k, &snap.prev_k).to();
                let alpha = (1.0 - rho2) * (1.0 - rho2) / 4.0;
                let beta = (1.0 - c.gamma_merge) * alpha;
                let mut m = vec![T::ZERO; d];
                for j in 0..d {
                    m[j] = snap.prev_k[j] * T::of(1.0 - beta) + kh[j] * T::of(beta);
                }
                let mn = T::dot(&m, &m).to().sqrt();
                if mn > c.ball {
                    let s = c.ball / mn;
                    for x in m.iter_mut() {
                        *x = *x * T::of(s);
                    }
                }
                self.sur[bi] = {
                    let mut ds = T::ZERO;
                    for j in 0..d {
                        let e = v_w[j] - snap.prev_v[j];
                        ds = ds + e * e;
                    }
                    T::of(ds.to().sqrt())
                };
                self.k[o..o + d].copy_from_slice(&m);
                self.v[o..o + d].copy_from_slice(&v_w);
                self.hits[bi] += 1;
                self.ts[bi] = t;
            }
            _ => {
                // Tier 3: new slot insert; evict argmin utility if full
                let mut j = None;
                for i in 0..c.fine {
                    if !self.alive[i] {
                        j = Some(i);
                        break;
                    }
                }
                let j = match j {
                    Some(j) => j,
                    None => {
                        let mut bu = f64::INFINITY;
                        let mut bidx = 0;
                        for i in 0..c.fine {
                            let u = 2.0 * self.hits[i] as f64 + self.sur[i].to();
                            if u < bu {
                                bu = u;
                                bidx = i;
                            }
                        }
                        bidx
                    }
                };
                snap.i = j;
                let o = j * d;
                snap.prev_k.copy_from_slice(&self.k[o..o + d]);
                snap.prev_v.copy_from_slice(&self.v[o..o + d]);
                snap.prev_ts = self.ts[j];
                snap.prev_hits = self.hits[j];
                snap.prev_sur = self.sur[j];
                snap.prev_alive = self.alive[j];
                let mut ds = T::ZERO;
                for q in 0..d {
                    let e = v_w[q] - r_ctx[q];
                    ds = ds + e * e;
                }
                self.sur[j] = T::of(ds.to().sqrt());
                self.k[o..o + d].copy_from_slice(&kstore);
                self.v[o..o + d].copy_from_slice(&v_w);
                self.hits[j] = 1;
                self.ts[j] = t;
                self.alive[j] = true;
            }
        }

        // ---- coarse resolution: Riemannian-normalized EMA toward the new pattern ----
        let mut any_c = false;
        let mut bcs = -2.0f64;
        let mut bci = 0usize;
        for i in 0..c.coarse {
            if !self.calive[i] {
                bci = i;
                any_c = false;
                break;
            }
        }
        if self.calive[bci] {
            any_c = true;
            for i in 0..c.coarse {
                if self.calive[i] {
                    let kn = T::dot(&self.ck[i * d..(i + 1) * d], &self.ck[i * d..(i + 1) * d]).to().sqrt();
                    let s = T::dot(&kh, &self.ck[i * d..(i + 1) * d]).to() / kn.max(1e-8);
                    if s > bcs {
                        bcs = s;
                        bci = i;
                    }
                }
            }
        }
        let ci = bci;
        snap.cidx = ci;
        snap.prev_ck.copy_from_slice(&self.ck[ci * d..(ci + 1) * d]);
        snap.prev_cv.copy_from_slice(&self.cv[ci * d..(ci + 1) * d]);
        snap.prev_calive = self.calive[ci];
        let _ = any_c;
        if snap.prev_calive {
            let mut x = vec![T::ZERO; d];
            for j in 0..d {
                x[j] = snap.prev_ck[j] * T::of(1.0 - c.gamma_coarse) + kh[j] * T::of(c.gamma_coarse);
            }
            let xn = T::dot(&x, &x).to().sqrt().max(1e-8);
            for j in 0..d {
                self.ck[ci * d + j] = x[j] * T::of(1.0 / xn);
                self.cv[ci * d + j] = snap.prev_cv[j] * T::of(1.0 - c.gamma_coarse) + v_w[j] * T::of(c.gamma_coarse);
            }
        } else {
            self.ck[ci * d..(ci + 1) * d].copy_from_slice(&kh);
            self.cv[ci * d..(ci + 1) * d].copy_from_slice(&v_w);
        }
        self.calive[ci] = true;

        if train {
            Some(snap)
        } else {
            None
        }
    }

    /// Reverse of one write: routes dL/dS(t) -> dL/dS(t−1) in `g` (in place), produces the
    /// write-path gradients (gk_raw, gv_raw, gglin), and restores the state to S(t−1).
    pub fn bwd_write(
        &mut self,
        snap: &MrmSnapshot<T>,
        g: &mut MrmGrads<T>,
        gk_raw: &mut [T],
        gv_raw: &mut [T],
        gglin: &mut T,
        k_raw: &[T],
        v_raw: &[T],
    ) {
        let c = self.cfg;
        let d = c.d;
        let kn = T::dot(k_raw, k_raw).to().sqrt();
        let kh: Vec<T> = k_raw.iter().map(|x| *x * T::of(1.0 / kn.max(1e-12))).collect();

        let mut gv_w = vec![T::ZERO; d];
        let mut gkhat = vec![T::ZERO; d];
        let i = snap.i;
        let o = i * d;

        match snap.op {
            0 => {
                // overwrite: K_i = ball·k̂, V_i = v_w
                let gkk = T::dot(&g.gk[o..o + d], &kh);
                for j in 0..d {
                    gv_w[j] = g.gv[o + j];
                    let gjk = g.gk[o + j];
                    gkhat[j] = gkhat[j] + T::of(c.ball) * (gjk - kh[j] * gkk);
                }
                for x in g.gv[o..o + d].iter_mut() {
                    *x = T::ZERO;
                }
                for x in g.gk[o..o + d].iter_mut() {
                    *x = T::ZERO;
                }
            }
            1 => {
                // merge: m = (1−β)c + βk̂ (β depends on ‖c‖), then project to ball
                for j in 0..d {
                    gv_w[j] = g.gv[o + j];
                }
                let rho2 = T::dot(&snap.prev_k, &snap.prev_k).to();
                let alpha = (1.0 - rho2) * (1.0 - rho2) / 4.0;
                let beta = (1.0 - c.gamma_merge) * alpha;
                let mut m = vec![T::ZERO; d];
                for j in 0..d {
                    m[j] = snap.prev_k[j] * T::of(1.0 - beta) + kh[j] * T::of(beta);
                }
                let mn = T::dot(&m, &m).to().sqrt();
                let mut gm = vec![T::ZERO; d];
                if mn > c.ball {
                    let b = c.ball / mn;
                    let gmd = T::dot(&g.gk[o..o + d], &m).to();
                    for j in 0..d {
                        gm[j] = T::of(b) * (g.gk[o + j] - m[j] * T::of(gmd / (mn * mn)));
                    }
                } else {
                    gm.copy_from_slice(&g.gk[o..o + d]);
                }
                // dβ/dρ² = (1−γ)·d((1−ρ²)²/4)/dρ² = −(1−γ)(1−ρ²)/2 ; dβ/dc = −(1−γ)(1−ρ²)·c
                let bg = (T::dot(&gm, &kh) - T::dot(&gm, &snap.prev_k)).to(); // <gm, kh − c>
                let bcoef = (1.0 - c.gamma_merge) * (1.0 - rho2);
                for j in 0..d {
                    gkhat[j] = gkhat[j] + gm[j] * T::of(beta);
                    g.gk[o + j] = gm[j] * T::of(1.0 - beta) - snap.prev_k[j] * T::of(bg * bcoef);
                }
                for x in g.gv[o..o + d].iter_mut() {
                    *x = T::ZERO;
                }
            }
            _ => {
                // insert/evict: K_j = ball·k̂, V_j = v_w; previous content dropped
                let gkk = T::dot(&g.gk[o..o + d], &kh);
                for j in 0..d {
                    gv_w[j] = g.gv[o + j];
                    let gjk = g.gk[o + j];
                    gkhat[j] = gkhat[j] + T::of(c.ball) * (gjk - kh[j] * gkk);
                }
                for x in g.gv[o..o + d].iter_mut() {
                    *x = T::ZERO;
                }
                for x in g.gk[o..o + d].iter_mut() {
                    *x = T::ZERO;
                }
            }
        }

        // coarse EMA chains
        let ci = snap.cidx;
        let co = ci * d;
        if snap.prev_calive {
            for j in 0..d {
                gv_w[j] = gv_w[j] + T::of(c.gamma_coarse) * g.gcv[co + j];
            }
            let mut x = vec![T::ZERO; d];
            for j in 0..d {
                x[j] = snap.prev_ck[j] * T::of(1.0 - c.gamma_coarse) + kh[j] * T::of(c.gamma_coarse);
            }
            let xn = T::dot(&x, &x).to().sqrt().max(1e-12);
            let xnn = T::dot(&x, &x).to().sqrt();
            let ck_new_norm = if xnn > 0.0 { xnn } else { 1.0 };
            let _ = ck_new_norm;
            // ck_new = x/‖x‖ ; gx = (g − ck_new⟨g, ck_new⟩)/‖x‖
            let mut gckn = 0.0f64;
            for j in 0..d {
                gckn += g.gck[co + j].to() * (x[j].to() / xn);
            }
            let mut gx = vec![T::ZERO; d];
            for j in 0..d {
                gx[j] = (g.gck[co + j] - x[j] * T::of(gckn / xn)) * T::of(1.0 / xn);
            }
            for j in 0..d {
                g.gck[co + j] = gx[j] * T::of(1.0 - c.gamma_coarse);
                gkhat[j] = gkhat[j] + gx[j] * T::of(c.gamma_coarse);
            }
        } else {
            // fresh coarse insert in fwd: cv = v_w, ck = kh (factor 1, no EMA scaling)
            for j in 0..d {
                gv_w[j] = gv_w[j] + g.gcv[co + j];
                gkhat[j] = gkhat[j] + g.gck[co + j];
            }
            for x in g.gck[co..co + d].iter_mut() {
                *x = T::ZERO;
            }
        }
        for j in 0..d {
            g.gcv[co + j] = g.gcv[co + j] * T::of(1.0 - c.gamma_coarse);
        }

        // restore coarse state
        self.ck[co..co + d].copy_from_slice(&snap.prev_ck);
        self.cv[co..co + d].copy_from_slice(&snap.prev_cv);
        self.calive[ci] = snap.prev_calive;

        // restore fine slot state
        self.k[o..o + d].copy_from_slice(&snap.prev_k);
        self.v[o..o + d].copy_from_slice(&snap.prev_v);
        self.ts[i] = snap.prev_ts;
        self.hits[i] = snap.prev_hits;
        self.sur[i] = snap.prev_sur;
        self.alive[i] = snap.prev_alive;

        // gate split: v_w = gate·v_raw ; k̂ = k_raw/‖k_raw‖.
        // The full gkhat is grad wrt k̂, so the key-normalization Jacobian
        // (I − k̂k̂ᵀ)/‖k_raw‖ is applied here. Projectors are idempotent, so the
        // already-projected fine-branch contribution is unaffected.
        let gate = snap.gate;
        let gg = T::dot(&gv_w, v_raw) * T::of(gate * (1.0 - gate));
        *gglin = *gglin + gg;
        let gkk = T::dot(&gkhat, &kh);
        for j in 0..d {
            gv_raw[j] = gv_raw[j] + gv_w[j] * T::of(gate);
            gk_raw[j] = gk_raw[j] + (gkhat[j] - kh[j] * gkk) * T::of(1.0 / kn.max(1e-12));
        }
    }
}
