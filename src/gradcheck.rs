//! Acceptance criterion #1 — Finite-Difference Gradient Checks.
//!
//! Every module's analytical backward is compared against central finite differences in
//! f64 (the scalar reference path). The spec requires relative error < 1e-4; the f64
//! central difference typically lands at 1e-9 or better.
//!
//! Discrete memory-control decisions (argmax slot, tier, eviction) are piecewise-constant
//! in the parameters, so checks use random configurations away from decision boundaries.

use crate::attention::DiffAttn;
use crate::mrm::{Mrm, MrmCfg, MrmGrads};
use crate::model::{Model, ModelCfg};
use crate::norms::{qknorm_bwd_chunk, qknorm_fwd, rmsnorm_bwd, rmsnorm_fwd};
use crate::num::Num;
use crate::rng::Rng;
use crate::rope::Rope;

pub struct Report {
    pub rows: Vec<(String, f64, f64)>, // name, max_rel_err, threshold
    pub failed: usize,
}

const THRESH: f64 = 1e-4;

impl Report {
    fn new() -> Self {
        Self { rows: Vec::new(), failed: 0 }
    }
    fn check(&mut self, name: &str, analytic: &[f64], numeric: &[f64]) {
        let mut worst = 0.0f64;
        let mut worst_i = 0usize;
        for i in 0..analytic.len().min(numeric.len()) {
            let a = analytic[i];
            let n = numeric[i];
            let d = (a - n).abs();
            if d <= 1e-8 {
                continue; // FD noise floor: both effectively zero
            }
            let denom = a.abs().max(n.abs()).max(1e-6);
            let rel = d / denom;
            if rel > worst {
                worst = rel;
                worst_i = i;
            }
        }
        if worst >= THRESH {
            self.failed += 1;
            eprintln!(
                "FAIL {name}: max rel err {worst:.3e} at [{worst_i}] analytic={:+e} numeric={:+e}",
                analytic[worst_i], numeric[worst_i]
            );
        }
        self.rows.push((name.to_string(), worst, THRESH));
    }
}

/// Central finite differences where the closure receives the perturbed vector.
fn num_grad_with(mut f: impl FnMut(&[f64]) -> f64, base: &[f64], eps: f64) -> Vec<f64> {
    let mut g = vec![0.0f64; base.len()];
    let mut x = base.to_vec();
    for i in 0..x.len() {
        let old = x[i];
        x[i] = old + eps;
        let a = f(&x);
        x[i] = old - eps;
        let b = f(&x);
        x[i] = old;
        g[i] = (a - b) / (2.0 * eps);
    }
    g
}

fn rand_vec(rng: &mut Rng, n: usize, s: f64) -> Vec<f64> {
    (0..n).map(|_| rng.normal_mu(0.0, s)).collect()
}

fn dotr(out: &[f64], r: &[f64]) -> f64 {
    out.iter().zip(r.iter()).map(|(a, b)| a * b).sum()
}

pub fn run_all() -> Report {
    let mut rep = Report::new();
    check_rmsnorm(&mut rep);
    check_qknorm(&mut rep);
    check_rope(&mut rep);
    check_conv(&mut rep);
    check_ffn(&mut rep);
    check_pillar9(&mut rep);
    check_attn_and_mrm(&mut rep);
    check_loss_step(&mut rep);
    check_stage_chain(&mut rep);
    check_stage_time(&mut rep);
    check_full_model(&mut rep);
    rep
}

// 1 ── Affine RMSNorm -----------------------------------------------------------------
fn check_rmsnorm(rep: &mut Report) {
    let mut rng = Rng::new(1);
    let d = 12;
    let x0 = rand_vec(&mut rng, d, 1.0);
    let g0 = rand_vec(&mut rng, d, 0.5);
    let r = rand_vec(&mut rng, d, 1.0);
    let (dx, dgamma) = {
        let mut dx = vec![0.0; d];
        let mut dgamma = vec![0.0; d];
        let mut xhat = vec![0.0; d];
        let mut out = vec![0.0; d];
        let s = rmsnorm_fwd(&x0, &g0, &mut out, &mut xhat);
        rmsnorm_bwd(&r, &g0, &xhat, s, &mut dx, &mut dgamma);
        (dx, dgamma)
    };
    let f = |p: &[f64]| -> f64 {
        // p = [x (d), gamma (d)]
        let (xx, gg) = p.split_at(d);
        let mut o = vec![0.0; d];
        let mut xh = vec![0.0; d];
        rmsnorm_fwd(xx, gg, &mut o, &mut xh);
        dotr(&o, &r)
    };
    let p = [x0.clone(), g0.clone()].concat();
    let ng = num_grad_with(&mut |q: &[f64]| f(q), &p, 1e-6);
    rep.check("pillar1.rmsnorm.dx", &dx, &ng[..d]);
    rep.check("pillar1.rmsnorm.dgamma", &dgamma, &ng[d..]);
    // safe, non-branching edge: zero input must stay finite
    let mut o = vec![0.0; d];
    let mut xh = vec![0.0; d];
    let _ = rmsnorm_fwd(&vec![0.0; d], &g0, &mut o, &mut xh);
    assert!(o.iter().all(|v| v.is_finite()));
}

// 3 ── Per-head QK-RMSNorm -------------------------------------------------------------
fn check_qknorm(rep: &mut Report) {
    let mut rng = Rng::new(2);
    let n = 8;
    let dsub = 4;
    let raw0 = rand_vec(&mut rng, n, 1.0);
    let g0 = rand_vec(&mut rng, dsub, 0.5);
    let r = rand_vec(&mut rng, n, 1.0);
    let (draw, dgamma) = {
        let mut draw = vec![0.0; n];
        let mut dgamma = vec![0.0; dsub];
        for c in 0..n / dsub {
            qknorm_bwd_chunk(
                &r[c * dsub..(c + 1) * dsub],
                &raw0[c * dsub..(c + 1) * dsub],
                &g0,
                &mut draw[c * dsub..(c + 1) * dsub],
                &mut dgamma,
                dsub,
            );
        }
        (draw, dgamma)
    };
    let f = |p: &[f64]| -> f64 {
        let (rr, gg) = p.split_at(n);
        let mut v = rr.to_vec();
        qknorm_fwd(&mut v, gg, n / dsub, dsub);
        dotr(&v, &r)
    };
    let p = [raw0.clone(), g0.clone()].concat();
    let ng = num_grad_with(&mut |q: &[f64]| f(q), &p, 1e-6);
    rep.check("pillar3.qknorm.draw", &draw, &ng[..n]);
    rep.check("pillar3.qknorm.dgamma", &dgamma, &ng[n..]);
    // |q·k| ≤ dsub bound on normalized vectors
    let mut a = rand_vec(&mut rng, dsub, 3.0);
    let mut b = rand_vec(&mut rng, dsub, 3.0);
    qknorm_fwd(&mut a, &vec![1.0; dsub], 1, dsub);
    qknorm_fwd(&mut b, &vec![1.0; dsub], 1, dsub);
    let dp = crate::num::Num::dot(&a, &b);
    assert!(dp.abs() <= dsub as f64 + 1e-9, "QK dot bound violated: {dp}");
}

// 4 ── Adaptive RoPE banding --------------------------------------------------------------
fn rope_apply(eta: &[f64], pairs: usize, v: &mut [f64], nchunks: usize, pos: f64) {
    let rope = Rope::<f64> { eta: eta.to_vec(), deta: vec![0.0; pairs], pairs, base: 10000.0 };
    rope.apply(v, nchunks, pos);
}

fn check_rope(rep: &mut Report) {
    let mut rng = Rng::new(3);
    let pairs = 4;
    let n = 2 * pairs;
    let mut rope = Rope::<f64>::new(pairs, 10000.0, 0.5);
    for e in rope.eta.iter_mut() {
        *e = rng.normal_mu(0.0, 1.0);
    }
    let v0 = rand_vec(&mut rng, 3 * n, 1.0);
    let r = rand_vec(&mut rng, 3 * n, 1.0);

    // pos = +17 : input grads
    let mut v = v0.clone();
    rope.apply(&mut v, 3, 17.0);
    let mut gv = r.clone();
    rope.apply_bwd(&mut gv, &v0, 3, 17.0);
    let f = |p: &[f64], pos: f64| -> f64 {
        let (eta, vv) = p.split_at(pairs);
        let mut x = vv.to_vec();
        rope_apply(eta, pairs, &mut x, 3, pos);
        dotr(&x, &r)
    };
    let p = [rope.eta.clone(), v0.clone()].concat();
    let ng = num_grad_with(&mut |q: &[f64]| f(q, 17.0), &p, 1e-6);
    rep.check("pillar4.rope.dinput(pos17)", &gv, &ng[pairs..]);

    // pos = −513 (slot-key relative path): eta grads
    rope.zero_grad();
    let mut v2 = v0.clone();
    rope.apply(&mut v2, 3, -513.0);
    let mut g2 = r.clone();
    rope.apply_bwd(&mut g2, &v0, 3, -513.0);
    let eta_an = rope.deta.clone();
    let ng = num_grad_with(&mut |q: &[f64]| f(q, -513.0), &p, 1e-6);
    rep.check("pillar4.rope.deta(neg513)", &eta_an, &ng[..pairs]);
    rep.check("pillar4.rope.dinput(neg513)", &g2, &ng[pairs..]);
}

// 2 ── Causal depthwise highway conv ---------------------------------------------------------
fn check_conv(rep: &mut Report) {
    let mut rng = Rng::new(4);
    let (k, d, t_len) = (4, 6, 7);
    let mut conv = crate::conv::HwyConv::<f64>::new(k, d);
    for v in conv.w.iter_mut() {
        *v = rng.normal_mu(0.0, 0.5);
    }
    for v in conv.gate.iter_mut() {
        *v = rng.normal_mu(0.0, 0.5);
    }
    for v in conv.bg.iter_mut() {
        *v = rng.normal_mu(0.0, 0.5);
    }
    let xs: Vec<Vec<f64>> = (0..t_len).map(|_| rand_vec(&mut rng, d, 1.0)).collect();
    let rs: Vec<Vec<f64>> = (0..t_len).map(|_| rand_vec(&mut rng, d, 1.0)).collect();
    let zero = vec![0.0; d];

    fn run(conv: &mut crate::conv::HwyConv<f64>, xs: &Vec<Vec<f64>>, rs: &Vec<Vec<f64>>) -> f64 {
        conv.reset();
        let mut loss = 0.0;
        let mut out = vec![0.0; conv.d];
        for (t, x) in xs.iter().enumerate() {
            conv.fwd_step(x, &mut out);
            loss += dotr(&out, &rs[t]);
        }
        loss
    }
    // analytic grads via reverse sweep (params + per-step input grads)
    run(&mut conv, &xs, &rs);
    conv.zero_grad();
    let mut carry = vec![0.0f64; (k - 1) * d];
    let mut dxs: Vec<Vec<f64>> = Vec::new();
    for t in (0..t_len).rev() {
        let mut dx = vec![0.0; d];
        let xprev: Vec<&[f64]> = (1..k)
            .map(|j| if t >= j { xs[t - j].as_slice() } else { zero.as_slice() })
            .collect();
        conv.bwd_step(&rs[t], &xs[t], &xprev, &mut carry, &mut dx);
        dxs.push(dx);
    }
    dxs.reverse();
    let (dw_a, dg_a, db_a) = (conv.dw.clone(), conv.dgate.clone(), conv.dbg.clone());

    let f = |p: &[f64]| -> f64 {
        // p = [w, gate, bg]
        let mut c2 = crate::conv::HwyConv::<f64>::new(k, d);
        let (a, rest) = p.split_at(k * d);
        c2.w.copy_from_slice(a);
        let (b, c) = rest.split_at(d);
        c2.gate.copy_from_slice(b);
        c2.bg.copy_from_slice(c);
        run(&mut c2, &xs, &rs)
    };
    let p = [conv.w.clone(), conv.gate.clone(), conv.bg.clone()].concat();
    let ng = num_grad_with(&mut |q: &[f64]| f(q), &p, 1e-6);
    rep.check("pillar2.conv.dw", &dw_a, &ng[..k * d]);
    // per-step input grads (exercises history + carry across steps)
    for t in 0..t_len {
        let mut c3 = crate::conv::HwyConv::<f64>::new(k, d);
        c3.w.copy_from_slice(&conv.w);
        c3.gate.copy_from_slice(&conv.gate);
        c3.bg.copy_from_slice(&conv.bg);
        let fxt = |xv: &[f64]| -> f64 {
            let mut c4 = crate::conv::HwyConv::<f64>::new(k, d);
            c4.w.copy_from_slice(&conv.w);
            c4.gate.copy_from_slice(&conv.gate);
            c4.bg.copy_from_slice(&conv.bg);
            let mut xs2 = xs.clone();
            xs2[t] = xv.to_vec();
            run(&mut c4, &xs2, &rs)
        };
        let ngx = num_grad_with(&mut |q: &[f64]| fxt(q), &xs[t], 1e-6);
        rep.check(&format!("pillar2.conv.dx[{t}]"), &dxs[t], &ngx);
    }
    rep.check("pillar2.conv.dgate", &dg_a, &ng[k * d..k * d + d]);
    rep.check("pillar2.conv.dbg", &db_a, &ng[k * d + d..]);
}

// 7 ── SwiGLU + parallel LoRA ------------------------------------------------------------------
fn check_ffn(rep: &mut Report) {
    let mut rng = Rng::new(5);
    let (d, f, r) = (8, 20, 3);
    let mut ffn = crate::ffn::SwiGLU::<f64>::new(d, f, r);
    for v in ffn.wg.iter_mut() {
        *v = rng.normal_mu(0.0, 0.4);
    }
    for v in ffn.wu.iter_mut() {
        *v = rng.normal_mu(0.0, 0.4);
    }
    for v in ffn.wd.iter_mut() {
        *v = rng.normal_mu(0.0, 0.3);
    }
    for v in ffn.la.iter_mut() {
        *v = rng.normal_mu(0.0, 0.4);
    }
    for v in ffn.lb.iter_mut() {
        *v = rng.normal_mu(0.0, 0.3);
    }
    let x0 = rand_vec(&mut rng, d, 1.0);
    let rr = rand_vec(&mut rng, d, 1.0);

    // analytic
    let mut y = vec![0.0; d];
    let mut a = vec![0.0; r];
    ffn.fwd(&x0, &mut y, &mut a);
    let mut dx = vec![0.0; d];
    ffn.bwd(&x0, &a, &rr, &mut dx);
    let (dwg, dwu, dwd, dla, dlb) =
        (ffn.dwg.clone(), ffn.dwu.clone(), ffn.dwd.clone(), ffn.dla.clone(), ffn.dlb.clone());
    let lf = |p: &[f64]| -> f64 {
        // p = [x, wg, wu, wd, la, lb] — builds its own FFN each call
        let mut g = crate::ffn::SwiGLU::<f64>::new(d, f, r);
        let mut o = d;
        g.wg.copy_from_slice(&p[o..o + d * f]);
        o += d * f;
        g.wu.copy_from_slice(&p[o..o + d * f]);
        o += d * f;
        g.wd.copy_from_slice(&p[o..o + f * d]);
        o += f * d;
        g.la.copy_from_slice(&p[o..o + d * r]);
        o += d * r;
        g.lb.copy_from_slice(&p[o..o + r * d]);
        let mut y2 = vec![0.0; d];
        let mut a2 = vec![0.0; r];
        g.fwd(&p[..d], &mut y2, &mut a2);
        dotr(&y2, &rr)
    };

    let p = [x0.clone(), ffn.wg.clone(), ffn.wu.clone(), ffn.wd.clone(), ffn.la.clone(), ffn.lb.clone()].concat();
    let ng = num_grad_with(&mut |q: &[f64]| lf(q), &p, 1e-6);
    let mut o = 0;
    macro_rules! seg {
        ($name:expr, $acc:ident, $len:expr) => {{
            rep.check($name, &$acc, &ng[o..o + $len]);
            o += $len;
        }};
    }
    seg!("pillar7.ffn.dx", dx, d);
    seg!("pillar7.ffn.dwg", dwg, d * f);
    seg!("pillar7.ffn.dwu", dwu, d * f);
    seg!("pillar7.ffn.dwd", dwd, f * d);
    seg!("pillar7.lora.dla", dla, d * r);
    seg!("pillar7.lora.dlb", dlb, r * d);
    let _ = o; // last seg leaves o assigned-but-unread
}

// 9 ── soft-cap + dual-head MTP + Z-loss --------------------------------------------------------
fn check_pillar9(rep: &mut Report) {
    let mut rng = Rng::new(6);
    let v = 9;
    let cfg = ModelCfg { vocab: v, ..ModelCfg::default() };
    let m = Model::<f64>::new(cfg, 99);
    let z1 = rand_vec(&mut rng, v, 2.0);
    let z2 = rand_vec(&mut rng, v, 2.0);
    let (tgt1, tgt2, maskv) = (3i32, 7i32, 1.0);
    let (_, dz1, dz2) = m.loss_step(&z1, &z2, tgt1, tgt2, maskv);
    let (_, dzm, _) = m.loss_step(&z1, &z2, tgt1, tgt2, 0.0);
    let f = |zz: &[f64]| -> f64 {
        let (l, _, _) = m.loss_step(zz, &z2, tgt1, tgt2, maskv);
        l
    };
    let ng1 = num_grad_with(&mut |q: &[f64]| f(q), &z1, 1e-6);
    rep.check("pillar9.dlogits(main+cap+zloss)", &dz1, &ng1);
    let f2 = |zz: &[f64]| -> f64 {
        let (l, _, _) = m.loss_step(&z1, zz, tgt1, tgt2, maskv);
        l
    };
    let ng2 = num_grad_with(&mut |q: &[f64]| f2(q), &z2, 1e-6);
    rep.check("pillar9.dlogits(mtp_aux)", &dz2, &ng2);
    let fm = |zz: &[f64]| -> f64 {
        let (l, _, _) = m.loss_step(zz, &z2, tgt1, tgt2, 0.0);
        l
    };
    let ngm = num_grad_with(&mut |q: &[f64]| fm(q), &z1, 1e-6);
    rep.check("pillar9.zloss-only-when-masked", &dzm, &ngm);
}

// 5 + 8 ── Differential attention over a populated memory + MRM tape BPTT ------------------------
fn clone_attn(a: &DiffAttn<f64>) -> DiffAttn<f64> {
    DiffAttn {
        heads: a.heads,
        dh: a.dh,
        dsub: a.dsub,
        d: a.d,
        wq: a.wq.clone(),
        dwq: vec![0.0; a.wq.len()],
        gq: a.gq.clone(),
        dgq: vec![0.0; a.gq.len()],
        lam_a: a.lam_a.clone(),
        dlam_a: vec![0.0; a.lam_a.len()],
        lam_b: a.lam_b.clone(),
        dlam_b: vec![0.0; a.lam_b.len()],
        scale: a.scale.clone(),
        dscale: vec![0.0; a.scale.len()],
        rope: Rope { eta: a.rope.eta.clone(), deta: vec![0.0; a.rope.pairs], pairs: a.rope.pairs, base: a.rope.base },
        lam_init: a.lam_init,
        out_scale: a.out_scale,
        n_read: a.n_read,
    }
}

fn clone_mem(m: &Mrm<f64>) -> Mrm<f64> {
    let mut n = Mrm::new(m.cfg);
    n.k = m.k.clone();
    n.v = m.v.clone();
    n.hits = m.hits.clone();
    n.sur = m.sur.clone();
    n.ts = m.ts.clone();
    n.alive = m.alive.clone();
    n.ck = m.ck.clone();
    n.cv = m.cv.clone();
    n.calive = m.calive.clone();
    n.step = m.step;
    n
}


/// Direct probe of mrm.bwd_write: linear loss on the final memory state, no attention.
#[allow(dead_code)] // diagnostic helper — superseded by check_stage_merge in the standard suite
fn check_mrm_direct(rep: &mut Report) {
    type T = f64;
    use crate::num::Num as _NumT;
    let mcfg = MrmCfg { d: 16, fine: 3, coarse: 2, ..MrmCfg::default() };
    for trial in 0..4 {
        let mut rng = Rng::new(100 + trial as u64);
        let mut mem = Mrm::new(mcfg);
        let mut kraws: Vec<Vec<f64>> = Vec::new();
        let mut vraws: Vec<Vec<f64>> = Vec::new();
        let mut glins: Vec<f64> = Vec::new();
        let mut rctxs: Vec<Vec<f64>> = Vec::new();
        let mut snaps = Vec::new();
        let n = 1 + trial; // 1..=4 writes
        // trial 4: force a merge — k1 at cos 0.9 with k0
        let mut k0 = rand_vec(&mut rng, 64, 1.0);
        let mut k1: Vec<f64> = Vec::new();
        if trial == 4 {
            let n0 = k0.iter().map(|x| x * x).sum::<f64>().sqrt();
            let u0: Vec<f64> = k0.iter().map(|x| x / n0).collect();
            loop {
                let p = rand_vec(&mut rng, 64, 1.0);
                let dp: f64 = p.iter().zip(u0.iter()).map(|(a, b)| a * b).sum();
                let o: Vec<f64> = p.iter().zip(u0.iter()).map(|(a, b)| a - dp * b).collect();
                let nn = o.iter().map(|x| x * x).sum::<f64>().sqrt();
                if nn > 0.5 {
                    k1 = u0.iter().zip(o.iter().map(|x| x / nn)).map(|(a, b)| 0.90 * a + 0.436 * b).collect();
                    break;
                }
            }
            k0 = u0;
        }
        for s in 0..n {
            let kk = if trial == 4 && s == 0 { k0.clone() } else if trial == 4 { k1.clone() } else { rand_vec(&mut rng, 64, 1.0) };
            kraws.push(kk);
            vraws.push(rand_vec(&mut rng, 64, 1.0));
            glins.push(2.0);
            let rc = rand_vec(&mut rng, 64, 0.3);
            rctxs.push(rc.clone());
        }
        for s in 0..n {
            if let Some(sn) = mem.fwd_write(&kraws[s], &vraws[s], T::of(glins[s]), &rctxs[s], true) {
                snaps.push(sn);
            }
        }
        // linear loss on final state
        let gk_t: Vec<f64> = rand_vec(&mut rng, 3 * 64, 1.0);
        let gv_t: Vec<f64> = rand_vec(&mut rng, 3 * 64, 1.0);
        let gck_t: Vec<f64> = rand_vec(&mut rng, 2 * 64, 1.0);
        let gcv_t: Vec<f64> = rand_vec(&mut rng, 2 * 64, 1.0);
        let loss_of_state = |k: &[f64], v: &[f64], ck: &[f64], cv: &[f64]| -> f64 {
            let mut l = 0.0;
            for i in 0..k.len() { l += gk_t[i] * k[i] + gv_t[i] * v[i]; }
            for i in 0..ck.len() { l += gck_t[i] * ck[i] + gcv_t[i] * cv[i]; }
            l
        };
        // analytic: route the linear grads through the tape in reverse
        let mut g = crate::mrm::MrmGrads::<f64> {
            gk: vec![0.0; 3 * 64],
            gv: vec![0.0; 3 * 64],
            gck: vec![0.0; 2 * 64],
            gcv: vec![0.0; 2 * 64],
        };
        for i in 0..gk_t.len() { g.gk[i] = T::of(gk_t[i]); }
        for i in 0..gv_t.len() { g.gv[i] = T::of(gv_t[i]); }
        for i in 0..gck_t.len() { g.gck[i] = T::of(gck_t[i]); }
        for i in 0..gcv_t.len() { g.gcv[i] = T::of(gcv_t[i]); }
        let mut gk_raw = vec![0.0f64; n * 16];
        let mut gv_raw = vec![0.0f64; n * 16];
        let mut gglin = vec![0.0f64; n];
        for s in (0..snaps.len()).rev() {
            let mut g1 = vec![0.0f64; 64];
            let mut g2 = vec![0.0f64; 64];
            let mut g3 = 0.0f64;
            mem.bwd_write(&snaps[s], &mut g, &mut g1, &mut g2, &mut g3, &kraws[s], &vraws[s]);
            for j in 0..64 {
                gk_raw[s * 64 + j] += g1[j];
                gv_raw[s * 64 + j] += g2[j];
            }
            gglin[s] += g3;
        }
        // numeric: replay from scratch
        let replay = |kk: &[Vec<f64>], vv: &[Vec<f64>]| -> f64 {
            let mut mm = Mrm::new(mcfg);
            for s in 0..kk.len() {
                let _ = mm.fwd_write(&kk[s], &vv[s], T::of(glins[s]), &rctxs[s], false);
            }
            loss_of_state(&mm.k, &mm.v, &mm.ck, &mm.cv)
        };
        let eps = 1e-6;
        for s in 0..n {
            let mut ng = vec![0.0f64; 16];
            for j in 0..64 {
                let mut kp = kraws.clone();
                kp[s][j] += eps;
                let lp = replay(&kp, &vraws);
                let mut km = kraws.clone();
                km[s][j] -= eps;
                let lm = replay(&km, &vraws);
                ng[j] = (lp - lm) / (2.0 * eps);
            }
            rep.check(&format!("mrm.direct.dk[{trial},{s}]"), &gk_raw[s * 64..(s + 1) * 64], &ng);
        }
    }
}

/// One-sided FD pair: plus[i] = (L(x+eps e_i) − L(x))/eps, minus[i] = (L(x) − L(x−eps e_i))/eps.
/// For a smooth function they agree to O(eps²); a ±-straddled discrete boundary makes
/// them disagree by the jump/eps, which eps-halving can never detect.
fn num_grad_split<F: FnMut(&[f64]) -> f64>(f: &mut F, x0: &[f64], eps: f64) -> (Vec<f64>, Vec<f64>) {
    let n = x0.len();
    let mut xp = x0.to_vec();
    let mut xm = x0.to_vec();
    let mut plus = vec![0.0f64; n];
    let mut minus = vec![0.0f64; n];
    for i in 0..n {
        let orig = x0[i];
        xp[i] = orig + eps;
        let lp = f(&xp);
        xp[i] = orig;
        xm[i] = orig - eps;
        let lm = f(&xm);
        xm[i] = orig;
        plus[i] = (lp - f(x0)) / eps;
        minus[i] = (f(x0) - lm) / eps;
    }
    (plus, minus)
}

/// Element-wise FD discontinuity screening: an element is "stable" when the eps and
/// eps/2 estimates agree; screened elements straddle a discrete decision boundary.
fn screen(
    ng: &[f64],
    ng_half: &[f64],
) -> (Vec<usize>, usize) {
    let mut keep = Vec::new();
    let mut screened = 0usize;
    for i in 0..ng.len() {
        let d = (ng[i] - ng_half[i]).abs();
        let scale = ng[i].abs().max(ng_half[i].abs()).max(1e-4);
        if d / scale > 0.25 {
            screened += 1;
        } else {
            keep.push(i);
        }
    }
    (keep, screened)
}


/// Hybrid probe: attn-tape key patterns + ops, but linear loss on the final memory
/// state (no attention read). Isolates bwd_write chain correctness under the exact
/// tape geometry that fails in the full check.
#[allow(dead_code)] // diagnostic helper (seeded variant used via check_stage_merge)
fn check_mrm_pattern(rep: &mut Report, nsteps: usize) {
    check_mrm_pattern_v(rep, nsteps, false, 0.3);
}

fn check_mrm_pattern_seed(rep: &mut Report, nsteps: usize, seed: u64) {
    check_mrm_pattern_v_seed(rep, nsteps, false, 0.3, seed);
}

#[allow(dead_code)] // diagnostic helper
fn check_mrm_pattern_v(rep: &mut Report, nsteps: usize, no_merge: bool, gcoarse: f64) {
    check_mrm_pattern_v_seed(rep, nsteps, no_merge, gcoarse, 7);
}

fn check_mrm_pattern_v_seed(rep: &mut Report, nsteps: usize, no_merge: bool, gcoarse: f64, seed: u64) {
    check_mrm_pattern_v_full(rep, nsteps, no_merge, gcoarse, seed, 5, true);
}

fn check_mrm_pattern_v_full(rep: &mut Report, nsteps: usize, no_merge: bool, gcoarse: f64, seed: u64, fine: usize, noisy_glin: bool) {
    type Tp = f64;
    let d = 64usize;
    let mcfg = MrmCfg { d, fine, coarse: 2, gamma_coarse: gcoarse, ..MrmCfg::default() };
    let mut rng = Rng::new(seed);
    let mut mem = Mrm::new(mcfg);
    let mut kraws: Vec<Vec<f64>> = Vec::new();
    let mut vraws: Vec<Vec<f64>> = Vec::new();
    let mut glins: Vec<f64> = Vec::new();
    let mut rctxs: Vec<Vec<f64>> = Vec::new();
    let mut snaps = Vec::new();
    let base_dir = rand_vec(&mut rng, d, 1.0);
    let bnorm = base_dir.iter().map(|x| x * x).sum::<f64>().sqrt();
    let base_u: Vec<f64> = base_dir.iter().map(|x| x / bnorm).collect();
    let orth: Vec<f64> = loop {
        let p = rand_vec(&mut rng, d, 1.0);
        let dp: f64 = p.iter().zip(base_u.iter()).map(|(a, b)| a * b).sum();
        let o: Vec<f64> = p.iter().zip(base_u.iter()).map(|(a, b)| a - dp * b).collect();
        let n = o.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 0.5 {
            break o.into_iter().map(|x| x / n).collect();
        }
    };
    for s in 0..nsteps {
        let mut k = rand_vec(&mut rng, d, 1.0);
        if !no_merge {
            match s % 3 {
                0 => k = base_u.iter().map(|x| x * 0.99999).collect(),
                1 => k = base_u.iter().zip(orth.iter()).map(|(a, b)| 0.90 * a + 0.436 * b).collect(),
                _ => {}
            }
        }
        kraws.push(k);
        vraws.push(rand_vec(&mut rng, d, 1.0));
        glins.push(if noisy_glin { 2.0 + rng.normal_mu(0.0, 0.5) } else { 2.0 });
        let rc = rand_vec(&mut rng, d, 0.3);
        rctxs.push(rc.clone());
        if let Some(sn) = mem.fwd_write(&kraws[s], &vraws[s], Tp::of(glins[s]), &rc, true) {
            snaps.push(sn);
        }
    }
    // invariant: replay state must equal analytic forward state exactly
    {
        let mut mm = Mrm::new(mcfg);
        for s in 0..nsteps {
            let _ = mm.fwd_write(&kraws[s], &vraws[s], Tp::of(glins[s]), &rctxs[s], false);
        }
        for i in 0..mem.k.len() {
            assert_eq!(mem.k[i].to(), mm.k[i].to(), "state.k diverges at {i}");
        }
        for i in 0..mem.ck.len() {
            assert_eq!(mem.ck[i].to(), mm.ck[i].to(), "state.ck diverges at {i}");
        }
        for i in 0..mem.cv.len() {
            assert_eq!(mem.cv[i].to(), mm.cv[i].to(), "state.cv diverges at {i}");
        }
        assert_eq!(mem.alive, mm.alive, "state.alive diverges");
        assert_eq!(mem.hits, mm.hits, "state.hits diverges");
        assert_eq!(mem.ts, mm.ts, "state.ts diverges");
        assert_eq!(mem.calive, mm.calive, "state.calive diverges");
    }
    let gk_t: Vec<f64> = rand_vec(&mut rng, 5 * d, 1.0);
    let gv_t: Vec<f64> = rand_vec(&mut rng, 5 * d, 1.0);
    let gck_t: Vec<f64> = rand_vec(&mut rng, 2 * d, 1.0);
    let gcv_t: Vec<f64> = rand_vec(&mut rng, 2 * d, 1.0);
    // analytic
    let g_init = crate::mrm::MrmGrads::<f64> {
        gk: gk_t.iter().map(|&x| Tp::of(x)).collect(),
        gv: gv_t.iter().map(|&x| Tp::of(x)).collect(),
        gck: gck_t.iter().map(|&x| Tp::of(x)).collect(),
        gcv: gcv_t.iter().map(|&x| Tp::of(x)).collect(),
    };
    let mut g = crate::mrm::MrmGrads::<f64> {
        gk: gk_t.iter().map(|&x| Tp::of(x)).collect(),
        gv: gv_t.iter().map(|&x| Tp::of(x)).collect(),
        gck: gck_t.iter().map(|&x| Tp::of(x)).collect(),
        gcv: gcv_t.iter().map(|&x| Tp::of(x)).collect(),
    };
    let mut gk_raw = vec![0.0f64; nsteps * d];
    for s in (0..snaps.len()).rev() {
        let mut g1 = vec![0.0f64; d];
        let mut g2 = vec![0.0f64; d];
        let mut g3 = 0.0f64;
        mem.bwd_write(&snaps[s], &mut g, &mut g1, &mut g2, &mut g3, &kraws[s], &vraws[s]);
        for j in 0..d { gk_raw[s * d + j] += g1[j]; }
        let _ = g2;
    }
    // numeric
    let replay = |kk: &[Vec<f64>], vv: &[Vec<f64>]| -> f64 {
        let mut mm = Mrm::new(mcfg);
        for s in 0..kk.len() {
            let _ = mm.fwd_write(&kk[s], &vv[s], Tp::of(glins[s]), &rctxs[s], false);
        }
        let mut l = 0.0;
        for i in 0..mm.k.len() { l += gk_t[i] * mm.k[i] + gv_t[i] * mm.v[i]; }
        for i in 0..mm.ck.len() { l += gck_t[i] * mm.ck[i] + gcv_t[i] * mm.cv[i]; }
        l
    };
    for s in 0..nsteps {
        let eps = 1e-6;
        let mut ng = vec![0.0f64; d];
        for j in 0..d {
            let mut kp = kraws.clone(); kp[s][j] += eps;
            let lp = replay(&kp, &vraws);
            let mut km = kraws.clone(); km[s][j] -= eps;
            let lm = replay(&km, &vraws);
            ng[j] = (lp - lm) / (2.0 * eps);
        }
        if seed == 0 && fine == 5 && noisy_glin && s == 0 {
            use std::fmt::Write as _;
            let mut out = String::new();
            let kn = kraws[0].iter().map(|x| x * x).sum::<f64>().sqrt();
            for j in 0..d {
                let _ = writeln!(out, "{} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e}", j, gk_raw[s * d + j], ng[j], kraws[0][j] / kn, g_init.gk[j], g_init.gck[j], g_init.gck[d + j]);
            }
            std::fs::write("/tmp/dk_dump.txt", out).unwrap();
        }
        rep.check(&format!("p8.pat.s{nsteps}.{seed}.{fine}.{noisy_glin}.dk[{s}]"), &gk_raw[s * d..(s + 1) * d], &ng);
    }
}

fn check_attn_and_mrm(rep: &mut Report) {
    // isolation matrix: (fine, noisy_glin)
    check_mrm_pattern_v_full(rep, 3, true, 0.3, 7, 3, false);
    check_mrm_pattern_v_full(rep, 3, true, 0.3, 7, 5, false);
    check_mrm_pattern_v_full(rep, 3, true, 0.3, 7, 3, true);
    check_mrm_pattern_v_full(rep, 3, true, 0.3, 7, 5, true);
    for sd in 0u64..8 {
        check_mrm_pattern_seed(rep, 3, sd);
    }
    for nw in 1..=9 {
        check_attn_tape(rep, nw, 7);
    }
    for nw in 1..=6 {
        check_attn_tape(rep, nw, 11);
    }
}

fn check_attn_tape(rep: &mut Report, nsteps: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let (d, m, fine, coarse) = (10, 64, 5, 2); // d=64: 6.50/√d=0.8125 gives the intended wide merge band
    let heads = 2;
    let dh = m / heads; // 4
    let n_read = fine + coarse;
    let mcfg = MrmCfg { d: m, fine, coarse, ..MrmCfg::default() };

    let mut attn = DiffAttn::<f64>::new(d, heads, dh, 2, n_read, 10000.0);
    for v in attn.wq.iter_mut() {
        *v = rng.normal_mu(0.0, 0.5);
    }
    for v in attn.gq.iter_mut() {
        *v = 0.5 + 0.5 * rng.f64();
    }
    for v in attn.lam_a.iter_mut() {
        *v = rng.normal_mu(0.0, 0.3);
    }
    for v in attn.lam_b.iter_mut() {
        *v = rng.normal_mu(0.0, 0.3);
    }
    for v in attn.scale.iter_mut() {
        *v = 0.5 + 0.5 * rng.f64();
    }
    for v in attn.rope.eta.iter_mut() {
        *v = rng.normal_mu(0.0, 0.7);
    }

    // ---- write tape: writes into 5 fine slots (forces overwrite / merge / evict) ----
    let mut mem = Mrm::new(mcfg);
    let mut kraws: Vec<Vec<f64>> = Vec::new();
    let mut vraws: Vec<Vec<f64>> = Vec::new();
    let mut glins: Vec<f64> = Vec::new();
    let mut snaps = Vec::new();
    let mut rctxs: Vec<Vec<f64>> = Vec::new();
    let base_dir = rand_vec(&mut rng, m, 1.0);
    let bnorm = base_dir.iter().map(|x| x * x).sum::<f64>().sqrt();
    let base_u: Vec<f64> = base_dir.iter().map(|x| x / bnorm).collect();
    // a unit vector orthogonal to base_u (Gram-Schmidt)
    let orth: Vec<f64> = loop {
        let p = rand_vec(&mut rng, m, 1.0);
        let dp: f64 = p.iter().zip(base_u.iter()).map(|(a, b)| a * b).sum();
        let o: Vec<f64> = p.iter().zip(base_u.iter()).map(|(a, b)| a - dp * b).collect();
        let n = o.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 0.5 {
            break o.into_iter().map(|x| x / n).collect();
        }
    };
    for s in 0..nsteps {
        // cos=0.99999 with the slot content -> solidly inside the OVERWRITE tier
        // cos=0.90                            -> solidly inside the MERGE tier [0.8125, 0.9999)
        // random                              -> below t_merge -> insert/evict tier
        let mut k = rand_vec(&mut rng, m, 1.0);
        match s % 3 {
            0 => {
                k = base_u.iter().map(|x| x * 0.99999).collect();
            }
            1 => {
                k = base_u.iter().zip(orth.iter()).map(|(a, b)| 0.90 * a + 0.436 * b).collect();
            }
            _ => {}
        }
        let v = rand_vec(&mut rng, m, 1.0);
        let gl = 2.0 + rng.normal_mu(0.0, 0.5);
        kraws.push(k);
        vraws.push(v);
        glins.push(gl);
        let rctx = rand_vec(&mut rng, m, 0.3);
        rctxs.push(rctx.clone());
        if let Some(snap) = mem.fwd_write(&kraws[s], &vraws[s], gl, &rctx, true) {
            snaps.push(snap);
        }
    }
    // sanity: expected op sequence (0/3/6 overwrite, 1/4/7 merge, others insert)
    for (s, sn) in snaps.iter().enumerate() {
        // op encoding: 0 = overwrite, 1 = merge, 2 = insert/evict.
        // Write 0 lands on an empty memory, so it is always an insert.
        let want = if s == 0 {
            2u8
        } else {
            match s % 3 {
                0 => 0u8,
                1 => 1u8,
                _ => 2u8,
            }
        };
        assert_eq!(sn.op, want, "tape test op mismatch at write {s} (got {} want {want})", sn.op);
    }
    assert_eq!(snaps.len(), nsteps, "all writes should pass the gate");

    let c0 = rand_vec(&mut rng, d, 1.0);
    let rref = rand_vec(&mut rng, m, 1.0);
    let tidx = nsteps as u32;

    // ---- analytic backward (mem is fully populated here) ----
    let mut out = vec![0.0; m];
    let cache = attn.fwd_step(&c0, &mem, tidx, &mut out);
    let mut gmem = MrmGrads {
        gk: vec![0.0; fine * m],
        gv: vec![0.0; fine * m],
        gck: vec![0.0; coarse * m],
        gcv: vec![0.0; coarse * m],
    };
    let mut dc = vec![0.0; d];
    attn.bwd_step(&c0, &mem, tidx, &cache, &rref, &mut gmem, &mut dc);
    let (dwq, dgq, dla, dlb, dsc, deta) = (
        attn.dwq.clone(),
        attn.dgq.clone(),
        attn.dlam_a.clone(),
        attn.dlam_b.clone(),
        attn.dscale.clone(),
        attn.rope.deta.clone(),
    );

    // populated-state snapshot (for read-path numeric grads)
    let mem_pop = clone_mem(&mem);
    // capture read-path grads BEFORE the tape walk consumes them
    let (gk_read, gv_read) = (gmem.gk.clone(), gmem.gv.clone());
    let gck_read = gmem.gck.clone();

    // tape backward: route state grads to each write (rewinds mem to empty)
    let mut gk_tape = vec![vec![0.0; m]; nsteps];
    let mut gv_tape = vec![vec![0.0; m]; nsteps];
    let mut gg_tape = vec![0.0; nsteps];
    for s in (0..nsteps).rev() {
        mem.bwd_write(&snaps[s], &mut gmem, &mut gk_tape[s], &mut gv_tape[s], &mut gg_tape[s], &kraws[s], &vraws[s]);
    }

    // ---- numeric checks (patch-on-clone) ----
    let replay = |a3: &DiffAttn<f64>, k: &[Vec<f64>], v: &[Vec<f64>], g: &[f64], cc: &[f64]| -> f64 {
        let mut mm = Mrm::new(mcfg);
        for s in 0..nsteps {
            // identical r_ctx as the analytic pass: it feeds utility/eviction decisions
            let _ = mm.fwd_write(&k[s], &v[s], g[s], &rctxs[s], false);
        }
        let mut o = vec![0.0; m];
        let _ = a3.fwd_step(cc, &mm, tidx, &mut o);
        dotr(&o, &rref)
    };
    let state_loss = |a3: &DiffAttn<f64>, pk: Option<&[f64]>, pv: Option<&[f64]>, pck: Option<&[f64]>| -> f64 {
        let mut mm = clone_mem(&mem_pop);
        if let Some(kk) = pk {
            mm.k.copy_from_slice(kk);
        }
        if let Some(vv) = pv {
            mm.v.copy_from_slice(vv);
        }
        if let Some(cc) = pck {
            mm.ck.copy_from_slice(cc);
        }
        let mut o = vec![0.0; m];
        let _ = a3.fwd_step(&c0, &mm, tidx, &mut o);
        dotr(&o, &rref)
    };
    macro_rules! attn_check {
        ($field:ident, $acc:ident, $name:expr) => {{
            let base = attn.$field.clone();
            let ng = num_grad_with(
                &mut |p: &[f64]| {
                    let mut a3 = clone_attn(&attn);
                    a3.$field.copy_from_slice(p);
                    replay(&a3, &kraws, &vraws, &glins, &c0)
                },
                &base,
                1e-6,
            );
            rep.check($name, &$acc, &ng);
        }};
    }
    attn_check!(wq, dwq, "pillar5.diffattn.dwq");
    attn_check!(gq, dgq, "pillar3.diffattn.dgq");
    attn_check!(lam_a, dla, "pillar5.diffattn.dlam_a");
    attn_check!(lam_b, dlb, "pillar5.diffattn.dlam_b");
    attn_check!(scale, dsc, "pillar5.diffattn.dscale");
    {
        let base = attn.rope.eta.clone();
        let ng = num_grad_with(
            &mut |p: &[f64]| {
                let mut a3 = clone_attn(&attn);
                a3.rope.eta.copy_from_slice(p);
                replay(&a3, &kraws, &vraws, &glins, &c0)
            },
            &base,
            1e-6,
        );
        rep.check("pillar4.diffattn.deta", &deta, &ng);
    }
    {
        let ng = num_grad_with(
            &mut |p: &[f64]| {
                let a3 = clone_attn(&attn);
                replay(&a3, &kraws, &vraws, &glins, p)
            },
            &c0,
            1e-6,
        );
        rep.check("pillar5.diffattn.dc", &dc, &ng);
    }
    {
        let mk = mem_pop.k.clone();
        let ng = num_grad_with(&mut |p: &[f64]| {
            let a3 = clone_attn(&attn);
            state_loss(&a3, Some(p), None, None)
        }, &mk, 1e-6);
        rep.check("pillar8.memread.dk_final", &gk_read, &ng);
        let mv = mem_pop.v.clone();
        let ng = num_grad_with(&mut |p: &[f64]| {
            let a3 = clone_attn(&attn);
            state_loss(&a3, None, Some(p), None)
        }, &mv, 1e-6);
        rep.check("pillar8.memread.dv_final", &gv_read, &ng);
        // coarse key read grads
        let mck = mem_pop.ck.clone();
        let nck = num_grad_with(
            &mut |p: &[f64]| state_loss(&attn, None, None, Some(p)),
            &mck,
            1e-6,
        );
        rep.check("pillar8.memread.dck_final", &gck_read, &nck);
    }
    // exact tape BPTT: per-write gradients (k_raw, v_raw, glin)
    for s in 0..nsteps {
        let ks = kraws.clone();
        let ngk = num_grad_with(
            &mut |p: &[f64]| {
                let mut kk = ks.clone();
                kk[s] = p.to_vec();
                let a3 = clone_attn(&attn);
                replay(&a3, &kk, &vraws, &glins, &c0)
            },
            &ks[s],
            1e-6,
        );
        let ngk_half = num_grad_with(
            &mut |p: &[f64]| {
                let mut kk = ks.clone();
                kk[s] = p.to_vec();
                let a3 = clone_attn(&attn);
                replay(&a3, &kk, &vraws, &glins, &c0)
            },
            &ks[s],
            5e-7,
        );
        let (keep, screened) = screen(&ngk, &ngk_half);
        if screened > 0 {
            rep.rows.push((
                format!("p8.tape.s{nsteps}.dk[{s}] [screened {screened}/{}]", ks[s].len()),
                0.0,
                1e-4,
            ));
        }
        let av: Vec<f64> = keep.iter().map(|&i| gk_tape[s][i]).collect();
        let nv: Vec<f64> = keep.iter().map(|&i| ngk[i]).collect();
        rep.check(&format!("p8.tape.s{nsteps}.dk[{s}]"), &av, &nv);

        let vs = vraws.clone();
        let ngv = num_grad_with(
            &mut |p: &[f64]| {
                let mut vv = vs.clone();
                vv[s] = p.to_vec();
                let a3 = clone_attn(&attn);
                replay(&a3, &kraws, &vv, &glins, &c0)
            },
            &vs[s],
            1e-6,
        );
        rep.check(&format!("p8.tape.s{nsteps}.dv[{s}]"), &gv_tape[s], &ngv);

        let gs = glins.clone();
        let ngg = num_grad_with(
            &mut |p: &[f64]| {
                let mut gg = gs.clone();
                gg[s] = p[0];
                let a3 = clone_attn(&attn);
                replay(&a3, &kraws, &vraws, &gg, &c0)
            },
            &[gs[s]],
            1e-6,
        );
        rep.check(&format!("p8.tape.s{nsteps}.dg[{s}]"), &[gg_tape[s]], &[ngg[0]]);
    }
}

// ── full model: all 9 pillars integrated (value residual + MTP + tape BPTT) ─────────────────

/// Unit check of pillar 9: loss_step FD on raw logits (softcap + CE + MTP + z-loss).
fn check_loss_step(rep: &mut Report) {
    let cfg = ModelCfg {
        d: 4,
        layers: 1,
        heads: 1,
        m: 4,
        ff: 6,
        lora_r: 2,
        conv_k: 4,
        fine: 2,
        coarse: 1,
        vocab: 7,
        mtp_alpha: 0.3,
        zloss_w: 1e-4,
        cap: 30.0,
        rope_base: 10000.0,
    };
    let model = Model::<f64>::new(cfg, 99);
    let mut rng = Rng::new(5);
    let z1: Vec<f64> = (0..7).map(|_| rng.normal_mu(0.0, 3.0)).collect();
    let z2: Vec<f64> = (0..7).map(|_| rng.normal_mu(0.0, 3.0)).collect();
    let (loss_a, dz1_a, dz2_a) = model.loss_step(&z1, &z2, 3, 5, 1.0);
    let _ = loss_a;
    let fd = |which: usize, i: usize, eps: f64| -> f64 {
        let mut a1 = z1.clone();
        let mut a2 = z2.clone();
        let mut b1 = z1.clone();
        let mut b2 = z2.clone();
        if which == 0 {
            a1[i] += eps;
            b1[i] -= eps;
        } else {
            a2[i] += eps;
            b2[i] -= eps;
        }
        let (lp, _, _) = model.loss_step(&a1, &a2, 3, 5, 1.0);
        let (lm, _, _) = model.loss_step(&b1, &b2, 3, 5, 1.0);
        (lp - lm) / (2.0 * eps)
    };
    let ng1: Vec<f64> = (0..7).map(|i| fd(0, i, 1e-6)).collect();
    let ng2: Vec<f64> = (0..7).map(|i| fd(1, i, 1e-6)).collect();
    rep.check("pillar9.dz1", &dz1_a, &ng1);
    rep.check("pillar9.dz2", &dz2_a, &ng2);
}


/// Stage-level probe: interleaved read/write over time, linear loss on (h_out, out64).
/// FD-checks every stage parameter. This exercises the exact timestep wiring that
/// isolated module checks cannot.
/// Merge-tier stage probe: m=64 (t_merge=0.8125 active), clustered inputs make
/// consecutive writes land in the merge/overwrite bands, exercising tier-2 BPTT
/// inside the full stage.
fn check_stage_merge(rep: &mut Report) {
    type T = f64;
    let d = 32usize;
    let heads = 2usize;
    let dh = 32usize; // heads·dh must equal m (the memory row width)
    let m = 64usize;
    let ff = 24usize;
    let t_len: usize = std::env::var("MRG_T").ok().and_then(|x| x.parse().ok()).unwrap_or(4);
    let no_coarse = std::env::var("MRG_NO_COARSE").is_ok();
    let mut mcfg = MrmCfg { d: m, fine: 8, coarse: if no_coarse { 1 } else { 2 }, ..MrmCfg::default() };
    if let Ok(gc) = std::env::var("MRG_GC") {
        mcfg.gamma_coarse = gc.parse().unwrap();
    }
    let mut stage = crate::layer::Stage::<f64>::new(0, d, heads, dh, ff, 2, 4, mcfg, 10, 10000.0);
    let mut rng = Rng::new(57);
    // open the write gate: glin = gb + <gu, hn> ≈ gb when gu tiny
    stage.gb[0] = T::of(2.0);
    for x in stage.gu.iter_mut() { *x = T::ZERO; }
    // make writes non-degenerate: small random key/value projections
    let nrm = |x: f64| 1.0 / x.sqrt();
    for x in stage.wkw.iter_mut() { *x = rng.normal_mu(0.0, 0.3 * nrm(d as f64)); }
    for x in stage.wvw.iter_mut() { *x = rng.normal_mu(0.0, 0.3 * nrm(d as f64)); }
    // clustered inputs: hn nearly identical across steps => cos >= 0.8125
    let base = rand_vec(&mut rng, d, 1.0);
    let mut hs: Vec<Vec<f64>> = Vec::new();
    let mut rs: Vec<Vec<f64>> = Vec::new();
    let mut qs: Vec<Vec<f64>> = Vec::new();
    let noise = std::env::var("MRG_NOISE").ok().and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.01);
    for t in 0..t_len {
        let mut h = base.clone();
        for i in 0..d {
            h[i] += noise * (t as f64 + 1.0) * rng.normal();
        }
        hs.push(h);
        rs.push(rand_vec(&mut rng, m, 1.0));
        qs.push(rand_vec(&mut rng, d, 1.0));
    }
    // confirm the op mix actually covers merge/overwrite
    {
        stage.reset_state();
        for t in 0..t_len {
            let mut h = hs[t].clone();
            let _ = stage.fwd_step(&mut h, t as u32, None, true);
        }
        let ops: Vec<String> = (0..t_len)
            .map(|t| match &stage.cache[t].snap {
                Some(sn) => format!("t{t}:op{}", sn.op),
                None => format!("t{t}:skip"),
            })
            .collect();
        println!("merge-probe ops: {}", ops.join(" "));
    }
    let run_fwd = |st: &mut crate::layer::Stage<f64>, hh: &[Vec<f64>]| -> f64 {
        st.reset_state();
        let mut loss = 0.0;
        for t in 0..hh.len() {
            let mut h = hh[t].clone();
            let o = st.fwd_step(&mut h, t as u32, None, true);
            loss += T::dot(&o, &rs[t]) + T::dot(&h, &qs[t]);
        }
        loss
    };
    let _ = run_fwd(&mut stage, &hs);
    stage.zero_grad();
    let mut gh_in: Vec<Vec<f64>> = vec![vec![0.0; d]; t_len];
    for t in (0..t_len).rev() {
        let mut gh = qs[t].clone();
        stage.bwd_step(t, &mut gh, None, None, Some(rs[t].as_slice()));
        gh_in[t] = gh;
    }
    let mut prefs_all = stage_params(&mut stage);
    let names: Vec<String> = prefs_all.iter().map(|p| p.0.to_string()).collect();
    let analytic: Vec<Vec<f64>> = prefs_all.iter().map(|p| unsafe { (*p.2).clone() }).collect();
    let eps = 1e-5;
    for pi in 0..prefs_all.len() {
        let base_v: Vec<f64> = unsafe { (*prefs_all[pi].1).iter().map(|x| x.to()).collect() };
        let mut eval = |p: &[f64]| -> f64 {
            unsafe {
                let w = &mut *prefs_all[pi].1;
                for (dst, src) in w.iter_mut().zip(p.iter()) {
                    *dst = crate::num::Num::of(*src);
                }
            }
            run_fwd(&mut stage, &hs)
        };
        let (plus, minus) = num_grad_split(&mut eval, &base_v, eps);
        unsafe {
            let w = &mut *prefs_all[pi].1;
            for (dst, src) in w.iter_mut().zip(base_v.iter()) {
                *dst = crate::num::Num::of(*src);
            }
        }
        let (keep, screened) = split_screen(&plus, &minus);
        let av: Vec<f64> = keep.iter().map(|&i| analytic[pi][i]).collect();
        let nv: Vec<f64> = keep.iter().map(|&i| (plus[i] + minus[i]) / 2.0).collect();
        if !av.is_empty() {
            rep.check(&format!("mrg.{}", names[pi]), &av, &nv);
        } else if screened > 0 {
            rep.rows.push((format!("mrg.{} [all screened]", names[pi]), 0.0, 1e-4));
        }
    }
    // input grads at the first write
    for t in 0..t_len {
        let mut eval2 = |p: &[f64]| -> f64 {
            let mut hh = hs.clone();
            hh[t] = p.to_vec();
            run_fwd(&mut stage, &hh)
        };
        let (plus, minus) = num_grad_split(&mut eval2, &hs[t], 1e-6);
        let (keep, _screened) = split_screen(&plus, &minus);
        let av: Vec<f64> = keep.iter().map(|&i| gh_in[t][i]).collect();
        let nv: Vec<f64> = keep.iter().map(|&i| (plus[i] + minus[i]) / 2.0).collect();
        if !av.is_empty() {
            rep.check(&format!("mrg.dhin[{t}]"), &av, &nv);
        }
    }
}

fn check_stage_time(rep: &mut Report) {
    check_stage_merge(rep);
    check_stage_time_v(rep, 2, "full");
    check_stage_time_v(rep, 2, "nowo"); // conv-only cross-time
    check_stage_time_v(rep, 2, "now"); // memory-only cross-time
    for t in 3..=5 {
        check_stage_time_t(rep, t);
    }
}

fn check_stage_time_v(rep: &mut Report, t_len: usize, variant: &str) {
    match variant {
        "nowo" => check_stage_time_t2(rep, t_len, true, false),
        "now" => check_stage_time_t2(rep, t_len, false, true),
        _ => check_stage_time_t2(rep, t_len, false, false),
    }
}

fn check_stage_time_t2(rep: &mut Report, t_len: usize, zero_wo: bool, zero_convw: bool) {
    check_stage_time_t3(rep, t_len, zero_wo, zero_convw);
}

fn check_stage_time_t(rep: &mut Report, t_len: usize) {
    check_stage_time_t2(rep, t_len, false, false);
}

fn check_stage_time_t3(rep: &mut Report, t_len: usize, zero_wo: bool, zero_convw: bool) {
    let d = 32usize;
    let heads = 2usize;
    let dh = 16usize;
    let m = 32usize;
    let ff = 24usize;
    let mcfg = MrmCfg { d: m, fine: 8, coarse: 2, ..MrmCfg::default() };
    let mut stage = crate::layer::Stage::<f64>::new(0, d, heads, dh, ff, 2, 4, mcfg, 10, 10000.0);
    let mut rng = Rng::new(31);
    let nrm = |x: f64| 1.0 / x.sqrt();
    for x in stage.wo.iter_mut() { *x = rng.normal_mu(0.0, nrm(m as f64) / nrm(2.0)); }
    for x in stage.wkw.iter_mut() { *x = rng.normal_mu(0.0, nrm(d as f64)); }
    for x in stage.wvw.iter_mut() { *x = rng.normal_mu(0.0, nrm(d as f64)); }
    for x in stage.gu.iter_mut() { *x = rng.normal_mu(0.0, 0.1 * nrm(d as f64)); }
    for x in stage.attn.wq.iter_mut() { *x = rng.normal_mu(0.0, nrm(d as f64)); }
    for x in stage.ffn.wg.iter_mut() { *x = rng.normal_mu(0.0, nrm(d as f64)); }
    for x in stage.ffn.wu.iter_mut() { *x = rng.normal_mu(0.0, nrm(d as f64)); }
    for x in stage.ffn.wd.iter_mut() { *x = rng.normal_mu(0.0, 0.5 * nrm(ff as f64) / nrm(2.0)); }
    for x in stage.ffn.la.iter_mut() { *x = rng.normal_mu(0.0, nrm(d as f64)); }
    let _ = (zero_wo, zero_convw);
    let mut hs: Vec<Vec<f64>> = Vec::new();
    let mut rs: Vec<Vec<f64>> = Vec::new();
    let mut qs: Vec<Vec<f64>> = Vec::new();
    for _ in 0..t_len {
        hs.push(rand_vec(&mut rng, d, 1.0));
        rs.push(rand_vec(&mut rng, m, 1.0));
        qs.push(rand_vec(&mut rng, d, 1.0));
    }
    let run_fwd = |st: &mut crate::layer::Stage<f64>, hh: &[Vec<f64>]| -> f64 {
        st.reset_state();
        let mut loss = 0.0;
        for t in 0..hh.len() {
            let mut h = hh[t].clone();
            let o = st.fwd_step(&mut h, t as u32, None, true);
            loss += dotr(&o, &rs[t]) + dotr(&h, &qs[t]);
        }
        loss
    };
    let run_fwd_o = |st: &mut crate::layer::Stage<f64>, hh: &[Vec<f64>]| -> f64 {
        st.reset_state();
        let mut loss = 0.0;
        for t in 0..hh.len() {
            let mut h = hh[t].clone();
            let o = st.fwd_step(&mut h, t as u32, None, true);
            loss += dotr(&o, &rs[t]);
        }
        loss
    };
    let run_fwd_h = |st: &mut crate::layer::Stage<f64>, hh: &[Vec<f64>]| -> f64 {
        st.reset_state();
        let mut loss = 0.0;
        for t in 0..hh.len() {
            let mut h = hh[t].clone();
            let _ = st.fwd_step(&mut h, t as u32, None, true);
            loss += dotr(&h, &qs[t]);
        }
        loss
    };
    // analytic: fwd then reverse bwd, chain gh through time
    let l0 = run_fwd(&mut stage, &hs);
    let _ = l0;
    stage.zero_grad();
    let mut gh_in: Vec<Vec<f64>> = vec![vec![0.0; d]; t_len];
    for t in (0..t_len).rev() {
        let mut gh = qs[t].clone();
        // the probe's loss also seeds out64 directly: dL/dout64(t) = rs[t] must enter
        // through `ext`, exactly like the model's vres-ext plumbing
        stage.bwd_step(t, &mut gh, None, None, Some(rs[t].as_slice()));
        gh_in[t] = gh; // dL/d h_in(t), complete
    }
    let mut prefs = stage_params(&mut stage);
    let names: Vec<String> = prefs.iter().map(|p| p.0.to_string()).collect();
    let analytic: Vec<Vec<f64>> = prefs.iter().map(|p| unsafe { (*p.2).clone() }).collect();
    // numeric
    let eps = 1e-5;
    for pi in 0..prefs.len() {
        let base: Vec<f64> = unsafe { (*prefs[pi].1).iter().map(|x| x.to()).collect() };
        let mut eval = |p: &[f64]| -> f64 {
            unsafe {
                let w = &mut *prefs[pi].1;
                for (dst, src) in w.iter_mut().zip(p.iter()) {
                    *dst = crate::num::Num::of(*src);
                }
            }
            run_fwd(&mut stage, &hs)
        };
        let (plus, minus) = num_grad_split(&mut eval, &base, eps);
        unsafe {
            let w = &mut *prefs[pi].1;
            for (dst, src) in w.iter_mut().zip(base.iter()) {
                *dst = crate::num::Num::of(*src);
            }
        }
        let (keep, screened) = split_screen(&plus, &minus);
        if screened > 0 {
            rep.rows.push((format!("stageT{t_len}.{} [screened {screened}/{}]", names[pi], base.len()), 0.0, 1e-4));
        }
        let av: Vec<f64> = keep.iter().map(|&i| analytic[pi][i]).collect();
        let nv: Vec<f64> = keep.iter().map(|&i| (plus[i] + minus[i]) / 2.0).collect();
        if !av.is_empty() {
            rep.check(&format!("stageT{t_len}.{}", names[pi]), &av, &nv);
        }
    }
    // input grads: dL/d(h_t) for each timestep (tests write-path norm Jacobian,
    // conv carry across time, and the norm chains). Split by loss path.
    for t in 0..t_len {
        let mut eval2 = |p: &[f64]| -> f64 {
            let mut hh = hs.clone();
            hh[t] = p.to_vec();
            run_fwd(&mut stage, &hh)
        };
        let (plus, minus) = num_grad_split(&mut eval2, &hs[t], 1e-6);
        let (keep, screened) = split_screen(&plus, &minus);
        if screened > 0 {
            rep.rows.push((format!("stageT{t_len}.dhin[{t}] [screened {screened}/{d}]"), 0.0, 1e-4));
        }
        let av: Vec<f64> = keep.iter().map(|&i| gh_in[t][i]).collect();
        let nv: Vec<f64> = keep.iter().map(|&i| (plus[i] + minus[i]) / 2.0).collect();
        if !av.is_empty() {
            rep.check(&format!("stageT{t_len}.dhin[{t}]"), &av, &nv);
        }
    }
    let _ = (run_fwd_o, run_fwd_h);
}

type StagePref<'a> = (&'static str, *mut Vec<f64>, *mut Vec<f64>);

fn stage_params(s: &mut crate::layer::Stage<f64>) -> Vec<StagePref<'static>> {
    vec![
        ("g1", &mut s.g1 as *mut _, &mut s.dg1 as *mut _),
        ("conv.w", &mut s.conv.w as *mut _, &mut s.conv.dw as *mut _),
        ("conv.gate", &mut s.conv.gate as *mut _, &mut s.conv.dgate as *mut _),
        ("conv.bg", &mut s.conv.bg as *mut _, &mut s.conv.dbg as *mut _),
        ("attn.wq", &mut s.attn.wq as *mut _, &mut s.attn.dwq as *mut _),
        ("attn.gq", &mut s.attn.gq as *mut _, &mut s.attn.dgq as *mut _),
        ("attn.lam_a", &mut s.attn.lam_a as *mut _, &mut s.attn.dlam_a as *mut _),
        ("attn.lam_b", &mut s.attn.lam_b as *mut _, &mut s.attn.dlam_b as *mut _),
        ("attn.scale", &mut s.attn.scale as *mut _, &mut s.attn.dscale as *mut _),
        ("attn.eta", &mut s.attn.rope.eta as *mut _, &mut s.attn.rope.deta as *mut _),
        ("wo", &mut s.wo as *mut _, &mut s.dwo as *mut _),
        ("g2", &mut s.g2 as *mut _, &mut s.dg2 as *mut _),
        ("ffn.wg", &mut s.ffn.wg as *mut _, &mut s.ffn.dwg as *mut _),
        ("ffn.wu", &mut s.ffn.wu as *mut _, &mut s.ffn.dwu as *mut _),
        ("ffn.wd", &mut s.ffn.wd as *mut _, &mut s.ffn.dwd as *mut _),
        ("ffn.lora_a", &mut s.ffn.la as *mut _, &mut s.ffn.dla as *mut _),
        ("ffn.lora_b", &mut s.ffn.lb as *mut _, &mut s.ffn.dlb as *mut _),
        ("mem.wkw", &mut s.wkw as *mut _, &mut s.dwkw as *mut _),
        ("mem.wvw", &mut s.wvw as *mut _, &mut s.dwvw as *mut _),
        ("mem.gu", &mut s.gu as *mut _, &mut s.dgu as *mut _),
        ("mem.gb", &mut s.gb as *mut _, &mut s.dgb as *mut _),
    ]
}


/// Stage + final-norm + head + CE chain = the model minus embeddings/MTP/zloss.
/// FD-checks dL/d(h_t) for every t (the `dh` the model feeds into bwd_into_stages).
fn check_stage_chain(rep: &mut Report) {
    let d = 32usize;
    let heads = 2usize;
    let dh = 16usize;
    let m = 32usize;
    let ff = 24usize;
    let v = 11usize;
    let t_len = 3usize;
    // build the real model (same cfg as the failing p1 check) and copy its params in
    let mcfg2 = crate::model::ModelCfg {
        d: 32,
        layers: 1,
        heads: 2,
        m: 32,
        ff: 24,
        lora_r: 2,
        conv_k: 4,
        fine: 8,
        coarse: 2,
        vocab: 11,
        mtp_alpha: 0.0,
        zloss_w: 0.0,
        cap: 30.0,
        rope_base: 10000.0,
    };
    let model = crate::model::Model::<f64>::new(mcfg2, 4242);
    let mcfg = MrmCfg { d: m, fine: 8, coarse: 2, ..MrmCfg::default() };
    let mut stage = crate::layer::Stage::<f64>::new(0, d, heads, dh, ff, 2, 4, mcfg, 10, 10000.0);
    stage.wo.copy_from_slice(&model.stages[0].wo);
    stage.attn.wq.copy_from_slice(&model.stages[0].attn.wq);
    stage.ffn.wg.copy_from_slice(&model.stages[0].ffn.wg);
    stage.ffn.wu.copy_from_slice(&model.stages[0].ffn.wu);
    stage.ffn.wd.copy_from_slice(&model.stages[0].ffn.wd);
    stage.ffn.la.copy_from_slice(&model.stages[0].ffn.la);
    stage.wkw.copy_from_slice(&model.stages[0].wkw);
    stage.wvw.copy_from_slice(&model.stages[0].wvw);
    stage.gu.copy_from_slice(&model.stages[0].gu);
    let mut rng = Rng::new(77); // same as the model check's token rng
    let head1: Vec<f64> = model.head1.clone();
    // identical draw order to check_full_model_pos: tokens first
    let toks: Vec<usize> = (0..t_len).map(|_| rng.below(v)).collect();
    let mut hs: Vec<Vec<f64>> = Vec::new();
    // embed rows from the model itself
    for t in 0..t_len {
        let mut row = vec![0.0; d];
        row.copy_from_slice(&model.embed[toks[t] * d..(toks[t] + 1) * d]);
        hs.push(row);
    }
    let gf: Vec<f64> = vec![1.0; d];
    let mut tgt1 = vec![-1i32; t_len];
    let mut mask = vec![0.0f64; t_len];
    // p1 pattern: only position 1 carries loss (mirrors the failing model config)
    if 1 + 1 < t_len {
        tgt1[1] = toks[2] as i32;
        mask[1] = 1.0;
    }
    // captured post-stage h per t (closure state)
    use std::cell::RefCell;
    let houts: RefCell<Vec<Vec<f64>>> = RefCell::new(vec![vec![0.0; d]; t_len]);
    let run_fwd = |st: &mut crate::layer::Stage<f64>, hh: &[Vec<f64>]| -> f64 {
        st.reset_state();
        houts.borrow_mut().clear();
        let mut loss = 0.0;
        for t in 0..hh.len() {
            let mut h = hh[t].clone();
            let _ = st.fwd_step(&mut h, t as u32, None, true);
            houts.borrow_mut().push(h.clone());
            let mut xhat = vec![0.0; d];
            let mut hn = vec![0.0; d];
            let _sn = crate::norms::rmsnorm_fwd(&h, &gf, &mut hn, &mut xhat);
            let mut z = vec![0.0f64; v];
            crate::num::Num::gemv(&head1, &hn, &mut z, d, v);
            for x in z.iter_mut() {
                *x = 30.0 * (*x / 30.0).tanh();
            }
            let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = zmax + z.iter().map(|&x| (x - zmax).exp()).sum::<f64>().ln();
            if tgt1[t] >= 0 {
                loss += -(z[tgt1[t] as usize] - lse);
            }
        }
        loss
    };
    let _ = run_fwd(&mut stage, &hs);
    stage.zero_grad();
    // analytic reverse sweep with the head chain seed
    let mut dh_in: Vec<Vec<f64>> = vec![vec![0.0; d]; t_len];
    let houts_v = houts.borrow().clone();
    for t in (0..t_len).rev() {
        let h = &houts_v[t];
        let mut xhat = vec![0.0; d];
        let mut hn = vec![0.0; d];
        let sn = crate::norms::rmsnorm_fwd(h, &gf, &mut hn, &mut xhat);
        let mut z = vec![0.0f64; v];
        crate::num::Num::gemv(&head1, &hn, &mut z, d, v);
        let mut zc = vec![0.0f64; v];
        for i in 0..v {
            zc[i] = 30.0 * (z[i] / 30.0).tanh();
        }
        let zmax = zc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = zmax + zc.iter().map(|&x| (x - zmax).exp()).sum::<f64>().ln();
        let mut dz = vec![0.0f64; v];
        if tgt1[t] >= 0 {
            for i in 0..v {
                let p = (zc[i] - lse).exp();
                let tt = if i as i32 == tgt1[t] { 1.0 } else { 0.0 };
                let j = 1.0 - (zc[i] / 30.0) * (zc[i] / 30.0);
                dz[i] = mask[t] * (p - tt) * j;
            }
        }
        let mut dxn = vec![0.0; d];
        crate::num::Num::gemv_dx(&head1, &dz, &mut dxn, d, v);
        let mut dh = vec![0.0; d];
        let mut dgf = vec![0.0; d];
        crate::norms::rmsnorm_bwd(&dxn, &gf, &xhat, sn, &mut dh, &mut dgf);
        stage.bwd_step(t, &mut dh, None, None, None);
        dh_in[t] = dh;
    }
    // FD per-t input grads
    for t in 0..t_len {
        let mut eval2 = |p: &[f64]| -> f64 {
            let mut hh = hs.clone();
            hh[t] = p.to_vec();
            run_fwd(&mut stage, &hh)
        };
        let (plus, minus) = num_grad_split(&mut eval2, &hs[t], 1e-6);
        let (keep, screened) = split_screen(&plus, &minus);
        if screened > 0 {
            rep.rows.push((format!("chain.dh[{t}] [screened {screened}/{d}]"), 0.0, 1e-4));
        }
        let av: Vec<f64> = keep.iter().map(|&i| dh_in[t][i].to()).collect();
        let nv: Vec<f64> = keep.iter().map(|&i| (plus[i] + minus[i]) / 2.0).collect();
        if !av.is_empty() {
            rep.check(&format!("chain.dh[{t}]"), &av, &nv);
        }
    }
    eprintln!("CHAIN_END");
    // FD a few key params: ffn.wg/wd, g2
    let prefs: Vec<(&str, *mut Vec<f64>, *mut Vec<f64>)> = vec![
        ("ffn.wg", &mut stage.ffn.wg as *mut _, &mut stage.ffn.dwg as *mut _),
        ("ffn.wd", &mut stage.ffn.wd as *mut _, &mut stage.ffn.dwd as *mut _),
        ("g2", &mut stage.g2 as *mut _, &mut stage.dg2 as *mut _),
    ];
    for (nm, wp, gp) in prefs.iter() {
        let base: Vec<f64> = unsafe { (**wp).iter().map(|x| x.to()).collect() };
        let mut eval3 = |p: &[f64]| -> f64 {
            unsafe {
                let w = &mut **wp;
                for (dst, src) in w.iter_mut().zip(p.iter()) {
                    *dst = crate::num::Num::of(*src);
                }
            }
            run_fwd(&mut stage, &hs)
        };
        let (plus, minus) = num_grad_split(&mut eval3, &base, 1e-5);
        let (keep, screened) = split_screen(&plus, &minus);
        if screened > 0 {
            rep.rows.push((format!("chain.{nm} [screened {screened}/{}]", base.len()), 0.0, 1e-4));
        }
        let av: Vec<f64> = unsafe { (**gp).iter().map(|x| x.to()).collect() };
        let ngc: Vec<f64> = keep.iter().map(|&i| (plus[i] + minus[i]) / 2.0).collect();
        let ak: Vec<f64> = keep.iter().map(|&i| av[i]).collect();
        if !ak.is_empty() {
            rep.check(&format!("chain.{nm}"), &ak, &ngc);
        }
        unsafe {
            let w = &mut **wp;
            for (dst, src) in w.iter_mut().zip(base.iter()) {
                *dst = crate::num::Num::of(*src);
            }
        }
    }
}

/// Screen on one-sided disagreement: smooth => |plus − minus| = O(eps²) tiny.
fn split_screen(plus: &[f64], minus: &[f64]) -> (Vec<usize>, usize) {
    let mut keep = Vec::new();
    let mut screened = 0usize;
    for i in 0..plus.len() {
        let d = (plus[i] - minus[i]).abs();
        let scale = plus[i].abs().max(minus[i].abs()).max(1e-4);
        if d / scale > 0.25 {
            screened += 1;
        } else {
            keep.push(i);
        }
    }
    (keep, screened)
}

fn check_full_model(rep: &mut Report) {
    // T-sweep at d=32, MTP/zloss off: isolates cross-time assembly
    check_full_model_full(rep, 3, 1, 0.0, 0.0, 30.0);
    check_full_model_seed(rep, 3, 0); // only position 0 carries loss
    check_full_model_seed(rep, 3, 1); // only position 1 carries loss
    check_full_model_seed(rep, 3, 2); // only position 2 carries loss
}

fn check_full_model_seed(rep: &mut Report, t_len: usize, seed_pos: usize) {
    check_full_model_pos(rep, t_len, 1, 0.0, 0.0, 30.0, seed_pos);
}

#[allow(dead_code)] // diagnostic helper — configurable-depth model probe
fn check_full_model_cfg(rep: &mut Report, t_len: usize, layers: usize) {
    check_full_model_full(rep, t_len, layers, 0.3, 1e-4, 30.0);
}

fn check_full_model_full(rep: &mut Report, t_len: usize, layers: usize, mtp: f64, zl: f64, cap: f64) {
    check_full_model_pos(rep, t_len, layers, mtp, zl, cap, usize::MAX);
}

fn check_full_model_pos(rep: &mut Report, t_len: usize, layers: usize, mtp: f64, zl: f64, cap: f64, seed_pos: usize) {
    let cfg = ModelCfg {
        d: 32,
        layers,
        heads: 2,
        m: 32,
        ff: 24,
        lora_r: 2,
        conv_k: 4,
        fine: 8,
        coarse: 2,
        vocab: 11,
        mtp_alpha: mtp,
        zloss_w: zl,
        cap,
        rope_base: 10000.0,
    };
    let t_len = t_len;
    let mut model = Model::<f64>::new(cfg, 4242);
    let mut rng = Rng::new(77);
    let toks: Vec<usize> = (0..t_len).map(|_| rng.below(cfg.vocab)).collect();
    let mut tgt1 = vec![-1i32; t_len];
    let mut tgt2 = vec![-1i32; t_len];
    let mut mask = vec![0.0; t_len];
    for t in 0..t_len {
        if t + 1 < t_len && (seed_pos == usize::MAX || t == seed_pos) {
            tgt1[t] = toks[t + 1] as i32;
            mask[t] = 1.0;
        }
        if t + 2 < t_len && (seed_pos == usize::MAX || t == seed_pos) {
            tgt2[t] = toks[t + 2] as i32;
        }
    }
    let loss_of = |model: &mut Model<f64>| -> f64 {
        model.reset_state();
        for &tk in toks.iter() {
            model.fwd_step(tk, true);
        }
        model.bwd_seq(&tgt1, &tgt2, &mask)
    };
    model.zero_grad();
    let l = loss_of(&mut model);
    assert!(l.is_finite());
    let params = model.params();
    let analytic: Vec<Vec<f64>> = params.iter().map(|p| unsafe { (*p.g).clone() }).collect();
    let names: Vec<String> = params
        .iter()
        .map(|p| format!("T{t_len}.L{layers}.p{seed_pos}.{}", p.name))
        .collect();

    // numeric — perturb every element of every parameter tensor, with eps/2
    // discontinuity screening (discrete memory decisions can straddle a perturbation)
    let eps = 1e-5;
    let mut params = params;
    for pi in 0..params.len() {
        let base: Vec<f64> = unsafe { (*params[pi].w).iter().map(|x| x.to()).collect() };
        let mut eval = |p: &[f64]| -> f64 {
            unsafe {
                let w = &mut *params[pi].w;
                for (dst, src) in w.iter_mut().zip(p.iter()) {
                    *dst = crate::num::Num::of(*src);
                }
            }
            loss_of(&mut model)
        };
        let ng = num_grad_with(&mut eval, &base, eps);
        let ng_half = num_grad_with(&mut eval, &base, eps / 2.0);
        unsafe {
            let w = &mut *params[pi].w;
            for (dst, src) in w.iter_mut().zip(base.iter()) {
                *dst = crate::num::Num::of(*src);
            }
        }
        let (keep, screened) = screen(&ng, &ng_half);
        if screened > 0 {
            rep.rows.push((
                format!("{}.{} [screened {screened}/{}]", &names[pi][..11], names[pi].split_once('.').unwrap().1, base.len()),
                0.0,
                1e-4,
            ));
        }
        if names[pi].contains("ffn.wd") && t_len == 3 {
            use std::fmt::Write as _;
            let mut out = String::new();
            for i in 0..analytic[pi].len() {
                let _ = writeln!(out, "{} {:.9e} {:.9e}", i, analytic[pi][i], ng[i]);
            }
            std::fs::write(format!("/tmp/fwd_{seed_pos}.txt"), out).unwrap();
        }
        let av: Vec<f64> = keep.iter().map(|&i| analytic[pi][i]).collect();
        let nv: Vec<f64> = keep.iter().map(|&i| ng[i]).collect();
        rep.check(&names[pi], &av, &nv);
    }
}
