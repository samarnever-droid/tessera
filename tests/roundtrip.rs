//! Checkpoint save/load must be bit-exact: a poisoned checkpoint silently
//! destroys resumed training.
#[test]
fn checkpoint_roundtrip_is_exact() {
    let mut m = tessera::model::Model::<f32>::new(tessera::model::ModelCfg::default(), 9);
    // distinctive values
    for p in m.params() {
        unsafe {
            for (i, x) in (*p.w).iter_mut().enumerate() {
                *x = tessera::num::Num::of((((i % 251) as f32 - 125.0) * 0.37 + 0.001 * (i % 7) as f32) as f64);
            }
        }
    }
    m.save("/tmp/rt_test.bin").expect("save");
    let mut m2 = tessera::model::Model::<f32>::new(tessera::model::ModelCfg::default(), 0);
    m2.load("/tmp/rt_test.bin").expect("load");
    let mut bad = 0usize;
    let mut mx = 0.0f64;
    for (pa, pb) in m.params().iter().zip(m2.params().iter()) {
        unsafe {
            for (a, b) in (*pa.w).iter().zip((*pb.w).iter()) {
                let d = (*a as f64 - *b as f64).abs();
                mx = mx.max(d);
                if d != 0.0 {
                    bad += 1;
                }
            }
        }
    }
    assert_eq!(bad, 0, "roundtrip mismatch: {bad} elems differ, max |Δ| = {mx:.3e}");
}
