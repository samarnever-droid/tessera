//! Standard causal Transformer baseline (pre-RMSNorm, fixed RoPE, SwiGLU FFN, multi-head
//! attention with an O(T) KV cache) for parameter-matched quality/speed comparisons.

use crate::ffn::SwiGLU;
use crate::model::Pref;
use crate::norms::{rmsnorm_bwd, rmsnorm_fwd};
use crate::num::Num;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug)]
pub struct BaselineCfg {
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub ff: usize,
    pub vocab: usize,
}

impl Default for BaselineCfg {
    fn default() -> Self {
        Self { d: 128, layers: 2, heads: 4, ff: 768, vocab: 256 }
    }
}

pub struct BLayer<T: Num> {
    pub wq: Vec<T>,
    pub dwq: Vec<T>,
    pub wk: Vec<T>,
    pub dwk: Vec<T>,
    pub wv: Vec<T>,
    pub dwv: Vec<T>,
    pub wo: Vec<T>, // [d][d]
    pub dwo: Vec<T>,
    pub g1: Vec<T>,
    pub dg1: Vec<T>,
    pub g2: Vec<T>,
    pub dg2: Vec<T>,
    pub ffn: SwiGLU<T>,
}

struct BLayerCache<T: Num> {
    xhat1: Vec<T>,    // T×d
    xhat2: Vec<T>,    // T×d
    q: Vec<T>,        // T×d (normed, rope'd, per-head split)
    k: Vec<T>,
    v: Vec<T>,
    attn_out: Vec<T>, // T×d (concat heads, pre-wo)
}

pub struct Baseline<T: Num> {
    pub cfg: BaselineCfg,
    pub embed: Vec<T>,
    pub dembed: Vec<T>,
    pub layers: Vec<BLayer<T>>,
    lcache: Vec<BLayerCache<T>>,
    pub gf: Vec<T>,
    pub dgf: Vec<T>,
    pub head: Vec<T>,
    pub dhead: Vec<T>,
}

fn rope_rotate<T: Num>(v: &mut [T], t: usize, pairs: usize, base: f64) {
    for i in 0..pairs {
        let th = base.powf(-(2.0 * i as f64) / (2.0 * pairs as f64));
        let phi = th * t as f64;
        let (s, c) = phi.sin_cos();
        let a = v[2 * i];
        let b = v[2 * i + 1];
        v[2 * i] = a * T::of(c) - b * T::of(s);
        v[2 * i + 1] = a * T::of(s) + b * T::of(c);
    }
}

/// Transposed rotation (gradient of rope_rotate wrt its input).
fn rope_unrotate<T: Num>(g: &mut [T], t: usize, pairs: usize, base: f64) {
    for i in 0..pairs {
        let th = base.powf(-(2.0 * i as f64) / (2.0 * pairs as f64));
        let phi = th * t as f64;
        let (s, c) = phi.sin_cos();
        let g0 = g[2 * i];
        let g1 = g[2 * i + 1];
        g[2 * i] = g0 * T::of(c) + g1 * T::of(s);
        g[2 * i + 1] = -g0 * T::of(s) + g1 * T::of(c);
    }
}

/// s of a normalization whose xhat is cached: xhat = sqrt(d)·x/s and ||x|| = s
/// (above the eps floor) => s = sqrt(d)/||xhat||.
fn s_from_xhat<T: Num>(xhat: &[T]) -> T {
    let n = T::dot(xhat, xhat).to().sqrt().max(1e-8);
    T::of((xhat.len() as f64).sqrt()) / T::of(n)
}

impl<T: Num> Baseline<T> {
    pub fn new(cfg: BaselineCfg, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let d = cfg.d;
        let mut layers = Vec::new();
        for _ in 0..cfg.layers {
            layers.push(BLayer {
                wq: vec![T::ZERO; d * d],
                dwq: vec![T::ZERO; d * d],
                wk: vec![T::ZERO; d * d],
                dwk: vec![T::ZERO; d * d],
                wv: vec![T::ZERO; d * d],
                dwv: vec![T::ZERO; d * d],
                wo: vec![T::ZERO; d * d],
                dwo: vec![T::ZERO; d * d],
                g1: vec![T::ONE; d],
                dg1: vec![T::ZERO; d],
                g2: vec![T::ONE; d],
                dg2: vec![T::ZERO; d],
                ffn: SwiGLU::new(d, cfg.ff, 0),
            });
        }
        let mut b = Self {
            embed: vec![T::ZERO; cfg.vocab * d],
            dembed: vec![T::ZERO; cfg.vocab * d],
            gf: vec![T::ONE; d],
            dgf: vec![T::ZERO; d],
            head: vec![T::ZERO; d * cfg.vocab],
            dhead: vec![T::ZERO; d * cfg.vocab],
            layers,
            lcache: Vec::new(),
            cfg,
        };
        let n = (d as f64).sqrt();
        for x in b.embed.iter_mut() {
            *x = T::of(rng.normal_mu(0.0, 0.02));
        }
        for l in b.layers.iter_mut() {
            for w in [&mut l.wq, &mut l.wk, &mut l.wv] {
                for x in w.iter_mut() {
                    *x = T::of(rng.normal_mu(0.0, 1.0 / n));
                }
            }
            for x in l.wo.iter_mut() {
                *x = T::of(rng.normal_mu(0.0, 1.0 / n / (2.0 * cfg.layers as f64).sqrt()));
            }
            for x in l.ffn.wg.iter_mut() {
                *x = T::of(rng.normal_mu(0.0, 1.0 / n));
            }
            for x in l.ffn.wu.iter_mut() {
                *x = T::of(rng.normal_mu(0.0, 1.0 / n));
            }
            for x in l.ffn.wd.iter_mut() {
                *x = T::of(rng.normal_mu(0.0, 0.5 / (cfg.ff as f64).sqrt() / (2.0 * cfg.layers as f64).sqrt()));
            }
        }
        for x in b.head.iter_mut() {
            *x = T::of(rng.normal_mu(0.0, 0.02));
        }
        b
    }

    pub fn count_params(&mut self) -> usize {
        self.params().iter().map(|p| unsafe { (*p.w).len() }).sum()
    }

    pub fn zero_grad(&mut self) {
        for v in self.dembed.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dgf.iter_mut() {
            *v = T::ZERO;
        }
        for v in self.dhead.iter_mut() {
            *v = T::ZERO;
        }
        for l in self.layers.iter_mut() {
            for g in [&mut l.dwq, &mut l.dwk, &mut l.dwv, &mut l.dwo, &mut l.dg1, &mut l.dg2] {
                for v in g.iter_mut() {
                    *v = T::ZERO;
                }
            }
            l.ffn.zero_grad();
        }
    }

    pub fn params(&mut self) -> Vec<Pref<T>> {
        let mut out = Vec::new();
        out.push(Pref { w: &mut self.embed as *mut _, g: &mut self.dembed as *mut _, wd: true, name: "embed" });
        for l in self.layers.iter_mut() {
            out.push(Pref { w: &mut l.g1 as *mut _, g: &mut l.dg1 as *mut _, wd: false, name: "g1" });
            out.push(Pref { w: &mut l.wq as *mut _, g: &mut l.dwq as *mut _, wd: true, name: "wq" });
            out.push(Pref { w: &mut l.wk as *mut _, g: &mut l.dwk as *mut _, wd: true, name: "wk" });
            out.push(Pref { w: &mut l.wv as *mut _, g: &mut l.dwv as *mut _, wd: true, name: "wv" });
            out.push(Pref { w: &mut l.wo as *mut _, g: &mut l.dwo as *mut _, wd: true, name: "wo" });
            out.push(Pref { w: &mut l.g2 as *mut _, g: &mut l.dg2 as *mut _, wd: false, name: "g2" });
            out.push(Pref { w: &mut l.ffn.wg as *mut _, g: &mut l.ffn.dwg as *mut _, wd: true, name: "ffn.wg" });
            out.push(Pref { w: &mut l.ffn.wu as *mut _, g: &mut l.ffn.dwu as *mut _, wd: true, name: "ffn.wu" });
            out.push(Pref { w: &mut l.ffn.wd as *mut _, g: &mut l.ffn.dwd as *mut _, wd: true, name: "ffn.wd" });
        }
        out.push(Pref { w: &mut self.gf as *mut _, g: &mut self.dgf as *mut _, wd: false, name: "gf" });
        out.push(Pref { w: &mut self.head as *mut _, g: &mut self.dhead as *mut _, wd: true, name: "head" });
        out
    }

    /// Forward + backward for one sequence (teacher forcing). Returns total masked CE.
    /// Held-out evaluation: mean CE (nats/token) on one chunk via a full forward+backward
    /// pass with unit mask, then gradients zeroed so training is unaffected.
    pub fn eval_ce(&mut self, toks: &[usize], tgt1: &[i32]) -> f64 {
        let mask = vec![1.0f64; toks.len()];
        let loss = self.fwd_bwd(toks, tgt1, &mask);
        self.zero_grad();
        loss
    }

    pub fn fwd_bwd(&mut self, toks: &[usize], tgt1: &[i32], mask: &[f64]) -> f64 {
        let (d, t_len, heads) = (self.cfg.d, toks.len(), self.cfg.heads);
        let dh = d / heads;
        let pairs = dh / 2;
        let v = self.cfg.vocab;
        let inv = 1.0 / (dh as f64).sqrt();

        // ================= forward =================
        let mut x = vec![T::ZERO; t_len * d];
        for (i, &tk) in toks.iter().enumerate() {
            x[i * d..(i + 1) * d].copy_from_slice(&self.embed[tk * d..(tk + 1) * d]);
        }
        self.lcache.clear();
        for l in self.layers.iter() {
            let mut lc = BLayerCache {
                xhat1: vec![T::ZERO; t_len * d],
                xhat2: vec![T::ZERO; t_len * d],
                q: vec![T::ZERO; t_len * d],
                k: vec![T::ZERO; t_len * d],
                v: vec![T::ZERO; t_len * d],
                attn_out: vec![T::ZERO; t_len * d],
            };
            for r in 0..t_len {
                let mut o = vec![T::ZERO; d];
                let mut xh = vec![T::ZERO; d];
                rmsnorm_fwd(&x[r * d..(r + 1) * d], &l.g1, &mut o, &mut xh);
                lc.xhat1[r * d..(r + 1) * d].copy_from_slice(&xh);
                T::gemv(&l.wq, &o, &mut lc.q[r * d..(r + 1) * d], d, d);
                T::gemv(&l.wk, &o, &mut lc.k[r * d..(r + 1) * d], d, d);
                T::gemv(&l.wv, &o, &mut lc.v[r * d..(r + 1) * d], d, d);
                for h in 0..heads {
                    rope_rotate(&mut lc.q[r * d + h * dh..r * d + (h + 1) * dh], r, pairs, 10000.0);
                    rope_rotate(&mut lc.k[r * d + h * dh..r * d + (h + 1) * dh], r, pairs, 10000.0);
                }
            }
            for h in 0..heads {
                for a in 0..t_len {
                    let mut logits = vec![T::ZERO; t_len];
                    for bb in 0..=a {
                        logits[bb] = T::dot(
                            &lc.q[a * d + h * dh..a * d + (h + 1) * dh],
                            &lc.k[bb * d + h * dh..bb * d + (h + 1) * dh],
                        ) * T::of(inv);
                    }
                    for bb in (a + 1)..t_len {
                        logits[bb] = T::of(-1e4);
                    }
                    T::softmax(&mut logits);
                    for bb in 0..=a {
                        let p = logits[bb];
                        if p.to() > 0.0 {
                            for qd in 0..dh {
                                lc.attn_out[a * d + h * dh + qd] =
                                    lc.attn_out[a * d + h * dh + qd] + p * lc.v[bb * d + h * dh + qd];
                            }
                        }
                    }
                }
            }
            for a in 0..t_len {
                let mut o = vec![T::ZERO; d];
                T::gemv(&l.wo, &lc.attn_out[a * d..(a + 1) * d], &mut o, d, d);
                for i in 0..d {
                    x[a * d + i] = x[a * d + i] + o[i];
                }
            }
            for a in 0..t_len {
                let mut o = vec![T::ZERO; d];
                let mut xh = vec![T::ZERO; d];
                rmsnorm_fwd(&x[a * d..(a + 1) * d], &l.g2, &mut o, &mut xh);
                lc.xhat2[a * d..(a + 1) * d].copy_from_slice(&xh);
                let mut fo = vec![T::ZERO; d];
                let mut la = vec![T::ZERO; 0];
                l.ffn.fwd(&o, &mut fo, &mut la);
                for i in 0..d {
                    x[a * d + i] = x[a * d + i] + fo[i];
                }
            }
            self.lcache.push(lc);
        }
        // head + loss
        let mut loss = 0.0f64;
        let mut dz = vec![T::ZERO; t_len * v];
        let mut xhatn_all = vec![T::ZERO; t_len * d];
        for a in 0..t_len {
            let mut hn = vec![T::ZERO; d];
            let mut xh = vec![T::ZERO; d];
            rmsnorm_fwd(&x[a * d..(a + 1) * d], &self.gf, &mut hn, &mut xh);
            xhatn_all[a * d..(a + 1) * d].copy_from_slice(&xh);
            let mut z = vec![T::ZERO; v];
            T::gemv(&self.head, &hn, &mut z, d, v);
            if mask[a] > 0.0 && tgt1[a] >= 0 {
                let mut c = z.clone();
                T::softmax(&mut c);
                loss += -c[tgt1[a] as usize].to().ln();
                for i in 0..v {
                    dz[a * v + i] = T::of(mask[a] * (c[i].to() - if i == tgt1[a] as usize { 1.0 } else { 0.0 }));
                }
            }
        }
        // ================= backward =================
        let mut dx = vec![T::ZERO; t_len * d];
        for a in 0..t_len {
            if mask[a] <= 0.0 {
                continue;
            }
            let dza = dz[a * v..(a + 1) * v].to_vec();
            let xh = xhatn_all[a * d..(a + 1) * d].to_vec();
            // hn = gf ⊙ xhat
            let mut hn = vec![T::ZERO; d];
            for i in 0..d {
                hn[i] = xh[i] * self.gf[i];
            }
            T::gemv_dw(&hn, &dza, &mut self.dhead, d, v);
            let mut dh_ = vec![T::ZERO; d];
            T::gemv_dx(&self.head, &dza, &mut dh_, d, v);
            let mut dxn = vec![T::ZERO; d];
            rmsnorm_bwd(&dh_, &self.gf, &xh, s_from_xhat(&xh), &mut dxn, &mut self.dgf);
            for i in 0..d {
                dx[a * d + i] = dx[a * d + i] + dxn[i];
            }
        }
        for li in (0..self.cfg.layers).rev() {
            let lc = &self.lcache[li];
            let (lc_xhat1, lc_xhat2) = (lc.xhat1.clone(), lc.xhat2.clone());
            let lc_attn = lc.attn_out.clone();
            let l = &mut self.layers[li];
            // ---- FFN block ----
            for a in 0..t_len {
                let dfo = dx[a * d..(a + 1) * d].to_vec();
                let mut dnorm_in = vec![T::ZERO; d];
                l.ffn.bwd(&lc_xhat2[a * d..(a + 1) * d], &[], &dfo, &mut dnorm_in);
                let xh = lc_xhat2[a * d..(a + 1) * d].to_vec();
                let mut dxn = vec![T::ZERO; d];
                rmsnorm_bwd(&dnorm_in, &l.g2, &xh, s_from_xhat(&xh), &mut dxn, &mut l.dg2);
                for i in 0..d {
                    dx[a * d + i] = dx[a * d + i] + dxn[i];
                }
            }
            // ---- attention block ----
            let mut dq = vec![T::ZERO; t_len * d];
            let mut dk = vec![T::ZERO; t_len * d];
            let mut dv = vec![T::ZERO; t_len * d];
            // causal attention backward with flash-style recomputation: the softmax
            // rows are rebuilt per query block from the cached q/k/v, keeping peak
            // memory O(QB * T) instead of O(heads * T^2).
            const QB: usize = 512;
            for h in 0..heads {
                for a0 in (0..t_len).step_by(QB) {
                    let a1 = (a0 + QB).min(t_len);
                    for a in a0..a1 {
                        let mut logits = vec![T::ZERO; a + 1];
                        for bb in 0..=a {
                            logits[bb] = T::dot(
                                &lc.q[a * d + h * dh..a * d + (h + 1) * dh],
                                &lc.k[bb * d + h * dh..bb * d + (h + 1) * dh],
                            ) * T::of(inv);
                        }
                        T::softmax(&mut logits);
                        let d_out = &dx[a * d + h * dh..a * d + (h + 1) * dh];
                        let mut dp = vec![T::ZERO; a + 1];
                        for bb in 0..=a {
                            let p = logits[bb];
                            let mut dpab = T::ZERO;
                            for qd in 0..dh {
                                dpab = dpab + d_out[qd] * lc.v[bb * d + h * dh + qd];
                                dv[bb * d + h * dh + qd] = dv[bb * d + h * dh + qd] + p * d_out[qd];
                            }
                            dp[bb] = dpab;
                        }
                        let mut dot = T::ZERO;
                        for bb in 0..=a {
                            dot = dot + logits[bb] * dp[bb];
                        }
                        for bb in 0..=a {
                            let dl = logits[bb] * (dp[bb] - dot) * T::of(inv);
                            if dl.to() != 0.0 {
                                for qd in 0..dh {
                                    dq[a * d + h * dh + qd] = dq[a * d + h * dh + qd] + dl * lc.k[bb * d + h * dh + qd];
                                    dk[bb * d + h * dh + qd] = dk[bb * d + h * dh + qd] + dl * lc.q[a * d + h * dh + qd];
                                }
                            }
                        }
                    }
                }
            }
            for r in 0..t_len {
                for h in 0..heads {
                    rope_unrotate(&mut dq[r * d + h * dh..r * d + (h + 1) * dh], r, pairs, 10000.0);
                    rope_unrotate(&mut dk[r * d + h * dh..r * d + (h + 1) * dh], r, pairs, 10000.0);
                }
            }
            // wo + qkv projections + norms
            for a in 0..t_len {
                let mut dcat = vec![T::ZERO; d];
                T::gemv_dx(&l.wo, &dx[a * d..(a + 1) * d], &mut dcat, d, d);
                T::gemv_dw(&lc_attn[a * d..(a + 1) * d], &dx[a * d..(a + 1) * d], &mut l.dwo, d, d);
                let mut dn1 = vec![T::ZERO; d];
                T::gemv_dx(&l.wq, &dq[a * d..(a + 1) * d], &mut dn1, d, d);
                let mut t2 = vec![T::ZERO; d];
                T::gemv_dx(&l.wk, &dk[a * d..(a + 1) * d], &mut t2, d, d);
                for i in 0..d {
                    dn1[i] = dn1[i] + t2[i];
                }
                T::gemv_dx(&l.wv, &dv[a * d..(a + 1) * d], &mut t2, d, d);
                for i in 0..d {
                    dn1[i] = dn1[i] + t2[i];
                }
                // n1 = g1 ⊙ xhat1
                let xh = lc_xhat1[a * d..(a + 1) * d].to_vec();
                let mut n1 = vec![T::ZERO; d];
                for i in 0..d {
                    n1[i] = xh[i] * l.g1[i];
                }
                T::gemv_dw(&n1, &dq[a * d..(a + 1) * d], &mut l.dwq, d, d);
                T::gemv_dw(&n1, &dk[a * d..(a + 1) * d], &mut l.dwk, d, d);
                T::gemv_dw(&n1, &dv[a * d..(a + 1) * d], &mut l.dwv, d, d);
                let mut dxn = vec![T::ZERO; d];
                rmsnorm_bwd(&dn1, &l.g1, &xh, s_from_xhat(&xh), &mut dxn, &mut l.dg1);
                for i in 0..d {
                    dx[a * d + i] = dx[a * d + i] + dxn[i];
                }
            }
        }
        // embedding
        for a in 0..t_len {
            let tk = toks[a];
            for i in 0..d {
                self.dembed[tk * d + i] = self.dembed[tk * d + i] + dx[a * d + i];
            }
        }
        loss
    }
}
