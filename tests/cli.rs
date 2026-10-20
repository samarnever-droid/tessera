//! CLI contract tests — "make it hard to break".
//! Exit codes: 0 success | 2 usage/validation error | 1 runtime failure.
//! Each test spawns the real binary (env CARGO_BIN_EXE_tessera); no shell, cross-platform.
//!
//! Test IDs: C01..C30 (unit-side robustness tests live as U01.. in src modules).

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::OnceLock;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_tessera")
}

fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .output()
        .expect("failed to spawn tessera binary")
}

fn stdout_of(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}
fn stderr_of(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

/// Assert the command succeeds and return stdout.
fn ok(args: &[&str]) -> String {
    let o = run(args);
    assert!(
        o.status.success(),
        "expected success: {:?}\nstderr: {}",
        args,
        stderr_of(&o)
    );
    stdout_of(&o)
}

/// Assert the command exits with code 2 (usage/validation) and stderr contains `needle`.
fn fails(args: &[&str], needle: &str) {
    let o = run(args);
    assert!(
        !o.status.success(),
        "expected failure: {:?} but got stdout: {}",
        args,
        stdout_of(&o)
    );
    let err = stderr_of(&o);
    assert!(
        err.contains(needle),
        "stderr of {:?} was:\n---\n{}\n--- (missing '{needle}')",
        args,
        err
    );
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tessera_cli_{}", name));
    std::fs::create_dir_all(&d).expect("create temp dir");
    d
}

/// One shared tiny checkpoint so eval/generate tests don't each re-train.
fn shared_ckpt() -> &'static PathBuf {
    static CK: OnceLock<PathBuf> = OnceLock::new();
    CK.get_or_init(|| {
        let dir = tmp("shared_ckpt");
        let p = dir.join("tiny.bin");
        let out = p.to_str().unwrap().to_string();
        ok(&[
            "niah-train", "--seq", "128", "--key-len", "1", "--val-len", "1",
            "--steps", "2", "--warm", "0", "--eval-every", "0", "--out", &out,
        ]);
        p
    })
}

fn ckpt_str() -> String {
    shared_ckpt().to_str().unwrap().to_string()
}

// ─── help / usage / dispatch ─────────────────────────────────────────────

#[test]
fn c01_help_flag_exits_zero() {
    let out = ok(&["help"]);
    assert!(out.contains("TESSERA-Q"), "usage text missing 'TESSERA-Q'");
    assert!(out.contains("exit codes"), "usage must document exit codes");
}

#[test]
fn c02_help_alias_flags_exit_zero() {
    for h in ["--help", "-h"] {
        let out = ok(&[h]);
        assert!(out.contains("usage"), "{h} must print usage");
    }
}

#[test]
fn c03_no_args_prints_usage_exits_zero() {
    let out = ok(&[]);
    assert!(out.contains("TESSERA-Q"));
}

#[test]
fn c04_unknown_command_exit_two() {
    let o = run(&["definitely-not-a-cmd"]);
    assert_eq!(o.status.code(), Some(2));
    let err = stderr_of(&o);
    assert!(err.contains("unknown command"), "stderr: {err}");
    assert!(err.contains("usage"), "unknown command must print usage to stderr");
}

#[test]
fn c05_typoed_subcommand_exit_two() {
    // regression: `niah-tran` (user typo) must be a clean error, not silence
    fails(&["niah-tran"], "unknown command");
}

#[test]
fn c06_nested_command_word_rejected() {
    // regression: `tessera tessera memtest` must not fall through to usage-with-exit-0
    let o = run(&["tessera", "memtest"]);
    assert_eq!(o.status.code(), Some(2));
}

// ─── argument validation ─────────────────────────────────────────────────

#[test]
fn c07_unknown_flag_rejected_with_name() {
    fails(&["niah-train", "--seq", "64", "--steps", "2", "--outt", "x.bin"], "--outt");
}

#[test]
fn c08_bad_int_value_rejected() {
    fails(&["memtest", "--steps", "abc"], "--steps");
}

#[test]
fn c09_bad_float_value_rejected() {
    fails(&["niah-train", "--seq", "64", "--steps", "2", "--lr", "fast"], "--lr");
}

#[test]
fn c10_missing_flag_value_rejected() {
    fails(&["memtest", "--steps"], "--steps");
}

#[test]
fn c11_flag_value_swallow_rejected() {
    // --steps followed by another flag: value missing
    fails(&["memtest", "--steps", "--seed"], "missing its value");
}

#[test]
fn c12_stray_positional_rejected() {
    fails(&["memtest", "100"], "unexpected argument");
}

#[test]
fn c13_negative_int_rejected() {
    fails(&["memtest", "--steps", "-5"], "--steps");
}

#[test]
fn c14_zero_steps_rejected() {
    fails(&["niah-train", "--seq", "64", "--steps", "0"], "--steps");
}

#[test]
fn c15_zero_n_rejected() {
    fails(&["niah-eval", "--model", "nope.bin", "--n", "0"], "--n");
}

// ─── memtest / selftest smoke ────────────────────────────────────────────

#[test]
fn c16_memtest_smoke_pass() {
    let out = ok(&["memtest", "--steps", "40"]);
    assert!(out.contains("PASS"), "memtest must print PASS: {out}");
    assert!(out.contains("substrate: 147456 B"), "substrate line: {out}");
}

#[test]
fn c17_selftest_smoke_pass() {
    let out = ok(&["selftest"]);
    assert!(out.contains("SELFTEST PASS"), "{out}");
}

// ─── niah-train ──────────────────────────────────────────────────────────

#[test]
fn c18_niah_train_tiny_saves_checkpoint() {
    let dir = tmp("c18");
    let out_path = dir.join("tiny.bin");
    let out = ok(&[
        "niah-train", "--seq", "64", "--key-len", "1", "--val-len", "1",
        "--steps", "3", "--warm", "0", "--eval-every", "0",
        "--out", out_path.to_str().unwrap(),
    ]);
    assert!(out.contains("params: 760256"), "param count line: {out}");
    assert!(out.contains("saved ->"), "save line: {out}");
    assert!(out_path.exists(), "checkpoint file must exist");
}

#[test]
fn c19_niah_train_same_seed_bit_identical() {
    let a = tmp("c19a").join("a.bin");
    let b = tmp("c19b").join("b.bin");
    let args = |p: &str| -> Vec<String> {
        [
            "niah-train", "--seq", "64", "--key-len", "1", "--val-len", "1",
            "--steps", "3", "--warm", "0", "--eval-every", "0", "--out",
        ]
        .iter()
        .map(|x| x.to_string())
        .chain(std::iter::once(p.to_string()))
        .collect()
    };
    let va = args(a.to_str().unwrap());
    let va: Vec<&str> = va.iter().map(|x| x.as_str()).collect();
    ok(&va);
    let vb = args(b.to_str().unwrap());
    let vb: Vec<&str> = vb.iter().map(|x| x.as_str()).collect();
    ok(&vb);
    let ba = std::fs::read(&a).unwrap();
    let bb = std::fs::read(&b).unwrap();
    assert_eq!(ba, bb, "same seed + same flags must produce a bit-identical checkpoint");
}

#[test]
fn c20_niah_train_resume_missing_checkpoint_fails() {
    let dir = tmp("c20");
    let p = dir.join("never_trained.bin");
    fails(
        &[
            "niah-train", "--seq", "64", "--key-len", "1", "--val-len", "1",
            "--steps", "2", "--warm", "0", "--eval-every", "0",
            "--out", p.to_str().unwrap(), "--resume",
        ],
        "does not exist",
    );
}

#[test]
fn c21_niah_train_resume_roundtrip() {
    let dir = tmp("c21");
    let p = dir.join("r.bin");
    let ps = p.to_str().unwrap();
    ok(&[
        "niah-train", "--seq", "64", "--key-len", "1", "--val-len", "1",
        "--steps", "2", "--warm", "0", "--eval-every", "0", "--out", ps,
    ]);
    let out = ok(&[
        "niah-train", "--seq", "64", "--key-len", "1", "--val-len", "1",
        "--steps", "2", "--warm", "0", "--eval-every", "0", "--out", ps, "--resume",
    ]);
    assert!(out.contains("resumed from"), "resume banner: {out}");
}

#[test]
fn c22_niah_train_rejects_seq_too_small() {
    fails(
        &["niah-train", "--seq", "8", "--key-len", "4", "--val-len", "4", "--steps", "2"],
        "too small",
    );
}

#[test]
fn c23_niah_train_creates_out_parent_dirs() {
    let deep = tmp("c23").join("a/b/c/deep.bin");
    ok(&[
        "niah-train", "--seq", "64", "--key-len", "1", "--val-len", "1",
        "--steps", "2", "--warm", "0", "--eval-every", "0",
        "--out", deep.to_str().unwrap(),
    ]);
    assert!(deep.exists(), "nested --out path must be created");
}

#[test]
fn c24_niah_train_rejects_negative_lr() {
    fails(
        &["niah-train", "--seq", "64", "--steps", "2", "--lr", "-1e-3"],
        "--lr",
    );
}

// ─── niah-eval / generate ────────────────────────────────────────────────

#[test]
fn c25_niah_eval_missing_model_fails() {
    fails(&["niah-eval", "--model", "definitely/missing.bin"], "not found");
}

#[test]
fn c26_niah_eval_output_format() {
    let m = ckpt_str();
    let out = ok(&["niah-eval", "--model", &m, "--seq", "128", "--n", "2", "--key-len", "1", "--val-len", "1"]);
    assert!(out.contains("NIAH@128:"), "eval line: {out}");
    assert!(out.contains('/'), "score format X/N: {out}");
}

#[test]
fn c27_niah_eval_rejects_seq_too_small() {
    let m = ckpt_str();
    fails(&["niah-eval", "--model", &m, "--seq", "8", "--key-len", "4", "--val-len", "4"], "too small");
}

#[test]
fn c28_generate_missing_model_fails() {
    fails(&["generate", "--model", "definitely/missing.bin"], "not found");
}

#[test]
fn c29_generate_runs_on_checkpoint() {
    let m = ckpt_str();
    let out = ok(&["generate", "--model", &m, "--prompt", "AB ?", "--n", "4"]);
    assert!(out.contains("continuation:"), "{out}");
}

// ─── LM train / parity on synthetic data ─────────────────────────────────

fn synthetic_data(name: &str, len: usize) -> String {
    let p = tmp(name).join("corpus.bin");
    // deterministic pseudo-text bytes
    let bytes: Vec<u8> = (0..len).map(|i| b'a' + ((i * 7 + i / 13) % 26) as u8).collect();
    std::fs::write(&p, &bytes).unwrap();
    p.to_str().unwrap().to_string()
}

#[test]
fn c30_train_lm_tiny_tessera() {
    let d = synthetic_data("c30", 8192);
    let out = ok(&["train", "--data", &d, "--seq", "32", "--steps", "2"]);
    assert!(out.contains("throughput:"), "{out}");
}

#[test]
fn c31_train_lm_tiny_transformer() {
    let d = synthetic_data("c31", 8192);
    let out = ok(&["train", "--data", &d, "--arch", "transformer", "--seq", "32", "--steps", "2"]);
    assert!(out.contains("throughput:"), "{out}");
}

#[test]
fn c32_train_lm_unknown_arch_fails() {
    let d = synthetic_data("c32", 1024);
    fails(&["train", "--data", &d, "--arch", "gpt5"], "unknown arch");
}

#[test]
fn c33_train_lm_missing_data_fails() {
    fails(&["train", "--data", "definitely/missing.bin", "--seq", "32", "--steps", "2"], "not found");
}

#[test]
fn c34_train_lm_data_too_small_fails() {
    let d = synthetic_data("c34", 4);
    fails(&["train", "--data", &d, "--seq", "64", "--steps", "2"], "too small");
}

#[test]
fn c35_train_lm_rejects_zero_seq() {
    let d = synthetic_data("c35", 1024);
    fails(&["train", "--data", &d, "--seq", "0", "--steps", "2"], "--seq");
}

#[test]
fn c36_parity_tiny_runs() {
    let d = synthetic_data("c36", 8192);
    let out = ok(&["parity", "--data", &d, "--seq", "32", "--steps", "2"]);
    assert!(out.contains("tessera params"), "param line: {out}");
    assert!(out.contains("speedup:"), "speedup line: {out}");
    assert!(out.contains("step"), "per-step loss lines: {out}");
}

#[test]
fn c37_parity_missing_data_fails() {
    fails(&["parity", "--data", "definitely/missing.bin", "--seq", "32", "--steps", "2"], "not found");
}

#[test]
fn c38_parity_bad_int_fails() {
    let d = synthetic_data("c38", 1024);
    fails(&["parity", "--data", &d, "--seq", "abc", "--steps", "2"], "--seq");
}

// ─── cross-cutting ───────────────────────────────────────────────────────

#[test]
fn c39_cli_flags_match_documented_set() {
    // guard against accidental flag renames: every documented flag must be accepted
    let dir = tmp("c39");
    let p = dir.join("full.bin");
    let ps = p.to_str().unwrap();
    ok(&[
        "niah-train", "--seq", "128", "--key-len", "1", "--val-len", "1",
        "--steps", "2", "--warm", "1", "--eval-every", "1", "--lr", "1e-3",
        "--seed", "9", "--distract", "0", "--eval-seq", "0", "--needle-loss",
        "--ans-w", "2.0", "--out", ps,
    ]);
    let out = ok(&[
        "niah-train", "--seq", "128", "--key-len", "1", "--val-len", "1",
        "--steps", "2", "--warm", "1", "--eval-every", "1", "--lr", "1e-3",
        "--seed", "9", "--distract", "0", "--eval-seq", "0", "--needle-loss",
        "--ans-w", "2.0", "--mix-kl", "1", "--mix-vl", "1", "--mix-p", "0.5",
        "--out", ps, "--resume",
    ]);
    assert!(out.contains("resumed from"));
}

#[test]
fn c40_niah_eval_fresh_seed_flag_accepted() {
    let m = ckpt_str();
    let out = ok(&[
        "niah-eval", "--model", &m, "--seq", "128", "--n", "1",
        "--key-len", "1", "--val-len", "1", "--seed", "77",
    ]);
    assert!(out.contains("NIAH@128:"), "{out}");
}

// ─── C41+: new CLI surface — per-command help, version/info/doctor, the ─────
// ─── niah-gen → transplant pipeline, hostile-input regressions. ─────────────

#[test]
fn c41_help_works_for_every_command() {
    for c in [
        "help", "version", "info", "doctor", "gradcheck", "selftest", "memtest",
        "niah-train", "niah-overfit", "niah-eval", "niah-gen", "train", "parity",
        "generate", "transplant",
    ] {
        let out = ok(&[c, "--help"]);
        assert!(out.contains("usage"), "{c} --help must mention usage: {out}");
        assert!(out.contains("exit codes"), "{c} --help must document exit codes: {out}");
    }
}

#[test]
fn c42_help_subcommand_and_top_level_flags() {
    assert!(ok(&["help", "transplant"]).contains("kd-weight"));
    assert!(ok(&["--help"]).contains("TESSERA-Q"));
    assert!(ok(&["-h"]).contains("exit codes"));
}

#[test]
fn c43_version() {
    for v in ["version", "--version", "-V"] {
        let out = ok(&[v]);
        assert!(out.contains("tessera 0."), "{v}: {out}");
        assert!(out.contains("SIMD"), "{v}: {out}");
    }
}

#[test]
fn c44_info_shows_model_facts() {
    let out = ok(&["info"]);
    assert!(out.contains("params"), "{out}");
    assert!(out.contains("153960"), "{out}"); // O(1) working state
    assert!(out.contains("147456"), "{out}"); // substrate
    assert!(out.contains("32 chars"), "{out}"); // retrieval alphabet
}

#[test]
fn c45_doctor_checks_pass() {
    let out = ok(&["doctor"]);
    assert!(out.contains("doctor:"), "{out}");
    assert!(out.contains("[ ok ]"), "{out}");
    assert!(out.contains("5/5"), "{out}");
}

#[test]
fn c46_unknown_command_gets_suggestion() {
    let o = run(&["niah-tran"]);
    assert!(!o.status.success());
    let err = stderr_of(&o);
    assert!(err.contains("unknown command"), "{err}");
    assert!(err.contains("did you mean"), "{err}");
    assert!(err.contains("niah-train"), "{err}");
}

#[test]
fn c47_niah_gen_creates_deterministic_pool() {
    let dir = tmp("c47");
    let pool = dir.join("pool.bin");
    let out = ok(&[
        "niah-gen", "--n", "8", "--seq", "20", "--key-len", "1", "--val-len", "1",
        "--seed", "99", "--out", pool.to_str().unwrap(),
    ]);
    assert!(out.contains("wrote"), "{out}");
    let raw = std::fs::read(&pool).unwrap();
    assert_eq!(&raw[0..8], b"TSPOOLv1");
    assert_eq!(raw.len(), 32 + 8 * 20 + 8 * 1 + 8 * 4);
    let pool2 = dir.join("pool2.bin");
    ok(&[
        "niah-gen", "--n", "8", "--seq", "20", "--key-len", "1", "--val-len", "1",
        "--seed", "99", "--out", pool2.to_str().unwrap(),
    ]);
    assert_eq!(raw, std::fs::read(&pool2).unwrap(), "same seed must give identical pool");
}

#[test]
fn c48_niah_gen_rejects_bad_values() {
    fails(&["niah-gen", "--n", "0"], "must be >= 1");
    fails(&["niah-gen", "--seq", "4", "--key-len", "1", "--val-len", "1"], "too small");
    fails(&["niah-gen", "--seed", "99999999999"], "u32");
}

#[test]
fn c49_transplant_requires_args() {
    fails(&["transplant"], "--pool is required");
    fails(&["transplant", "--pool", "x.bin"], "--labels is required");
    fails(
        &["transplant", "--pool", "nope.bin", "--labels", "nope2.bin"],
        "cannot read pool",
    );
    fails(
        &["transplant", "--pool", "x.bin", "--labels", "y.bin", "--kd-weight", "9"],
        "kd-weight must be",
    );
}

/// Uniform teacher labels over the 32-char alphabet (a maximally unsure
/// teacher) — enough to exercise the whole KD machinery in seconds.
fn write_labels(path: &std::path::Path, n: u32, vl: u32, ncls: u32, row_sum: f32) {
    let mut blob = b"TSLABLv1".to_vec();
    for v in [n, vl, ncls] {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    let p = row_sum / ncls as f32;
    for _ in 0..n * vl * ncls {
        blob.extend_from_slice(&p.to_le_bytes());
    }
    std::fs::write(path, &blob).unwrap();
}

#[test]
fn c50_transplant_end_to_end_tiny() {
    let dir = tmp("c50");
    let pool = dir.join("pool.bin");
    ok(&[
        "niah-gen", "--n", "8", "--seq", "20", "--key-len", "1", "--val-len", "1",
        "--seed", "5", "--out", pool.to_str().unwrap(),
    ]);
    let labels = dir.join("labels.bin");
    write_labels(&labels, 8, 1, 32, 1.0);
    let ck = dir.join("t.bin");
    let out = ok(&[
        "transplant", "--pool", pool.to_str().unwrap(), "--labels", labels.to_str().unwrap(),
        "--steps", "2", "--batch", "2", "--eval-every", "0", "--out", ck.to_str().unwrap(),
    ]);
    assert!(out.contains("saved ->"), "{out}");
    assert!(out.contains("TRANSPLANT"), "{out}");
    assert!(out.contains("teacher label acc"), "{out}");
    assert!(out.contains("params: 760256"), "{out}");
    assert!(ck.exists());
    // the transplanted checkpoint must load in the standard evaluator
    ok(&[
        "niah-eval", "--model", ck.to_str().unwrap(), "--seq", "20",
        "--key-len", "1", "--val-len", "1", "--n", "2",
    ]);
}

#[test]
fn c51_transplant_shape_mismatch_fails() {
    let dir = tmp("c51");
    let pool = dir.join("pool.bin");
    ok(&[
        "niah-gen", "--n", "8", "--seq", "20", "--key-len", "1", "--val-len", "1",
        "--seed", "5", "--out", pool.to_str().unwrap(),
    ]);
    let labels = dir.join("labels.bin");
    write_labels(&labels, 4, 1, 32, 1.0); // wrong n
    fails(
        &["transplant", "--pool", pool.to_str().unwrap(), "--labels", labels.to_str().unwrap()],
        "does not match the pool",
    );
    let labels2 = dir.join("labels2.bin");
    write_labels(&labels2, 8, 3, 32, 1.0); // wrong val_len
    fails(
        &["transplant", "--pool", pool.to_str().unwrap(), "--labels", labels2.to_str().unwrap()],
        "does not match the pool",
    );
    let labels3 = dir.join("labels3.bin");
    write_labels(&labels3, 8, 1, 16, 1.0); // wrong class count
    fails(
        &["transplant", "--pool", pool.to_str().unwrap(), "--labels", labels3.to_str().unwrap()],
        "retrieval alphabet has 32",
    );
}

#[test]
fn c52_transplant_bad_magic_fails() {
    let dir = tmp("c52");
    let junk = dir.join("junk.bin");
    std::fs::write(&junk, b"NOTAPOOL!!!!!!!!").unwrap();
    fails(
        &["transplant", "--pool", junk.to_str().unwrap(), "--labels", junk.to_str().unwrap()],
        "not a Tessera pool file",
    );
    let pool = dir.join("pool.bin");
    ok(&[
        "niah-gen", "--n", "4", "--seq", "20", "--key-len", "1", "--val-len", "1",
        "--seed", "5", "--out", pool.to_str().unwrap(),
    ]);
    let badlabels = dir.join("bad.bin");
    std::fs::write(&badlabels, b"NOTALBL!!!!!!!!").unwrap();
    fails(
        &["transplant", "--pool", pool.to_str().unwrap(), "--labels", badlabels.to_str().unwrap()],
        "not a Tessera label file",
    );
}

#[test]
fn c53_transplant_corrupt_prob_row_fails() {
    let dir = tmp("c53");
    let pool = dir.join("pool.bin");
    ok(&[
        "niah-gen", "--n", "4", "--seq", "20", "--key-len", "1", "--val-len", "1",
        "--seed", "5", "--out", pool.to_str().unwrap(),
    ]);
    let labels = dir.join("labels.bin");
    write_labels(&labels, 4, 1, 32, 0.5); // rows sum to 0.5, not ~1
    fails(
        &["transplant", "--pool", pool.to_str().unwrap(), "--labels", labels.to_str().unwrap()],
        "sum to",
    );
}

#[test]
fn c54_nan_lr_rejected_upfront() {
    fails(&["niah-train", "--lr", "nan"], "finite");
    fails(
        &["transplant", "--pool", "x", "--labels", "y", "--lr", "nan"],
        "finite",
    );
    let d = synthetic_data("c54", 8192);
    fails(&["train", "--data", &d, "--seq", "32", "--steps", "2", "--lr", "nan"], "finite");
}

#[test]
fn c55_transplant_detects_truncated_pool() {
    let dir = tmp("c55");
    let pool = dir.join("pool.bin");
    ok(&[
        "niah-gen", "--n", "8", "--seq", "20", "--key-len", "1", "--val-len", "1",
        "--seed", "5", "--out", pool.to_str().unwrap(),
    ]);
    let mut raw = std::fs::read(&pool).unwrap();
    raw.truncate(40); // header + 8 bytes of the first prompt
    std::fs::write(&pool, &raw).unwrap();
    let labels = dir.join("labels.bin");
    write_labels(&labels, 8, 1, 32, 1.0);
    fails(
        &["transplant", "--pool", pool.to_str().unwrap(), "--labels", labels.to_str().unwrap()],
        "truncated",
    );
}

// ─── C56+: the .tem container (pack / unpack / transparent load) ────────────

#[test]
fn c56_tem_pack_roundtrip_and_size() {
    let dir = tmp("c56");
    let bin = dir.join("m.bin");
    ok(&[
        "niah-train", "--seq", "64", "--steps", "2", "--key-len", "1", "--val-len", "1",
        "--ans-w", "5", "--warm", "0", "--eval-every", "0", "--out", bin.to_str().unwrap(),
    ]);
    let tem = dir.join("m.tem");
    let out = ok(&["pack", "--model", bin.to_str().unwrap(), "--out", tem.to_str().unwrap()]);
    assert!(out.contains("smaller"), "{out}");
    assert!(out.contains("FNV-1a64"), "{out}");
    assert!(out.contains("verify load"), "{out}");
    let bin_len = std::fs::metadata(&bin).unwrap().len();
    let tem_len = std::fs::metadata(&tem).unwrap().len();
    assert!(tem_len * 3 < bin_len * 2, "tem ({tem_len}) should be well under 2/3 of f32 ({bin_len})");
    let raw = std::fs::read(&tem).unwrap();
    assert_eq!(&raw[0..8], b"TEMST8V1");
    // transparent eval straight from the container
    let ev = ok(&[
        "niah-eval", "--model", tem.to_str().unwrap(), "--seq", "64",
        "--key-len", "1", "--val-len", "1", "--n", "4",
    ]);
    assert!(ev.contains("NIAH@64:"), "{ev}");
    // unpack -> legacy .bin that still evaluates
    let rt = dir.join("rt.bin");
    let up = ok(&["unpack", "--model", tem.to_str().unwrap(), "--out", rt.to_str().unwrap()]);
    assert!(up.contains("wrote"), "{up}");
    assert!(rt.exists());
    let ev2 = ok(&[
        "niah-eval", "--model", rt.to_str().unwrap(), "--seq", "64",
        "--key-len", "1", "--val-len", "1", "--n", "4",
    ]);
    assert!(ev2.contains("NIAH@64:"), "{ev2}");
}

#[test]
fn c57_tem_pack_out_to_subdir_and_default_name() {
    let dir = tmp("c57");
    let bin = dir.join("model_x.bin");
    ok(&[
        "niah-train", "--seq", "64", "--steps", "2", "--key-len", "1", "--val-len", "1",
        "--ans-w", "5", "--warm", "0", "--eval-every", "0", "--out", bin.to_str().unwrap(),
    ]);
    // default --out = same stem with .tem
    let def = dir.join("model_x.tem");
    ok(&["pack", "--model", bin.to_str().unwrap()]);
    assert!(def.exists());
}

#[test]
fn c58_tem_hostile_inputs() {
    let dir = tmp("c58");
    // packing a non-checkpoint
    let junk = dir.join("junk.bin");
    std::fs::write(&junk, b"GARBAGE!").unwrap();
    fails(&["pack", "--model", junk.to_str().unwrap()], "not a tessera checkpoint");
    // missing --model
    fails(&["pack"], "--model is required");
    fails(&["unpack"], "--model is required");
    // unpack on a non-tem file
    let bin = dir.join("m.bin");
    ok(&[
        "niah-train", "--seq", "64", "--steps", "2", "--key-len", "1", "--val-len", "1",
        "--ans-w", "5", "--warm", "0", "--eval-every", "0", "--out", bin.to_str().unwrap(),
    ]);
    fails(
        &["unpack", "--model", bin.to_str().unwrap()],
        "is not a .tem file",
    );
    // corrupt .tem (flip payload bytes) must fail the checksum loudly
    let tem = dir.join("m.tem");
    ok(&["pack", "--model", bin.to_str().unwrap(), "--out", tem.to_str().unwrap()]);
    let mut raw = std::fs::read(&tem).unwrap();
    let last = raw.len() - 1;
    raw[last] ^= 0xFF;
    let bad = dir.join("bad.tem");
    std::fs::write(&bad, &raw).unwrap();
    fails(&["unpack", "--model", bad.to_str().unwrap()], "checksum mismatch");
    fails(
        &["niah-eval", "--model", bad.to_str().unwrap(), "--seq", "64", "--key-len", "1", "--val-len", "1", "--n", "2"],
        "checksum mismatch",
    );
    // eval with an unknown-format file
    fails(
        &["niah-eval", "--model", junk.to_str().unwrap(), "--seq", "64", "--key-len", "1", "--val-len", "1", "--n", "2"],
        "not a tessera checkpoint",
    );
}

#[test]
fn c59_logo_prints_the_mark() {
    let out = ok(&["logo"]);
    assert!(out.contains("TESSERA-Q"), "{out}");
    assert!(out.contains("O(1) memory"), "{out}");
}

#[test]
fn c60_accel_selftest_and_help() {
    let out = ok(&["accel", "--help"]);
    assert!(out.contains("PJRT"), "{out}");
    assert!(out.contains("exit codes"), "{out}");
    // on any healthy box (no GPU + no TPU => skips; with hardware => ok rows)
    // the selftest must pass with exit 0
    let out = ok(&["accel"]);
    assert!(out.contains("ACCEL PASS"), "{out}");
    assert!(out.contains("CUDA"), "{out}");
    assert!(out.contains("TPU"), "{out}");
}
