//! TESSERA CLI — verification suites, training, NIAH, parity, generation.

use tessera::data::{lm_targets, Corpus, Niah, NiahCfg};
use tessera::gradcheck;
use tessera::model::{Model, ModelCfg};
use tessera::num::{simd_level_name, Num};
use tessera::optim::{wsd_lr, AdamW};

/// Bare-metal ANSI styling — zero crates. Color auto-off when stdout is not a
/// terminal (pipes, tests, notebooks); NO_COLOR always wins, CLICOLOR_FORCE=1
/// forces on. Piped/test output stays byte-identical to the historical format.
mod ui {
    use std::io::IsTerminal;
    use std::sync::atomic::{AtomicU8, Ordering};
    static MODE: AtomicU8 = AtomicU8::new(0); // 0 undecided, 1 off, 2 on

    fn on() -> bool {
        match MODE.load(Ordering::Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let v = if std::env::var_os("NO_COLOR").map(|v| !v.is_empty()).unwrap_or(false) {
                    false
                } else if std::env::var("CLICOLOR_FORCE").map(|v| v != "0").unwrap_or(false) {
                    true
                } else {
                    std::io::stdout().is_terminal()
                };
                MODE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
                v
            }
        }
    }

    pub fn paint(code: &str, s: &str) -> String {
        if on() {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn bold(s: &str) -> String {
        paint("1", s)
    }
    pub fn dim(s: &str) -> String {
        paint("2", s)
    }
    pub fn red(s: &str) -> String {
        paint("1;31", s)
    }
    pub fn green(s: &str) -> String {
        paint("1;32", s)
    }
    pub fn yellow(s: &str) -> String {
        paint("1;33", s)
    }
    pub fn cyan(s: &str) -> String {
        paint("1;36", s)
    }

    /// Right-aligned table row:  │ a │ b │
    pub fn row(cells: &[String], widths: &[usize]) -> String {
        let mut line = String::from("  │");
        for (c, w) in cells.iter().zip(widths) {
            line.push_str(&format!(" {c:>w$} │"));
        }
        line
    }
    pub fn rule_top(widths: &[usize]) -> String {
        let mut line = String::from("  ┌");
        for (k, w) in widths.iter().enumerate() {
            line.push_str(&"─".repeat(w + 2));
            line.push(if k + 1 == widths.len() { '┐' } else { '┬' });
        }
        line
    }
    pub fn rule_mid(widths: &[usize]) -> String {
        let mut line = String::from("  ├");
        for (k, w) in widths.iter().enumerate() {
            line.push_str(&"─".repeat(w + 2));
            line.push(if k + 1 == widths.len() { '┤' } else { '┼' });
        }
        line
    }
    pub fn rule_bot(widths: &[usize]) -> String {
        let mut line = String::from("  └");
        for (k, w) in widths.iter().enumerate() {
            line.push_str(&"─".repeat(w + 2));
            line.push(if k + 1 == widths.len() { '┘' } else { '┴' });
        }
        line
    }
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

/// CLI contract: exit 0 = success, 2 = usage/validation error, 1 = runtime failure.
fn fail(msg: String) -> ! {
    eprintln!("{} {msg}", ui::red("error:"));
    std::process::exit(2);
}

/// Options that take no value, per subcommand.
fn valueless_flags(cmd: &str) -> &'static [&'static str] {
    match cmd {
        "niah-train" => &["--resume", "--needle-loss"],
        _ => &[],
    }
}

/// All options accepted by each subcommand (unknown options are rejected).
fn known_flags(cmd: &str) -> &'static [&'static str] {
    match cmd {
        "niah-train" => &[
            "--seq", "--steps", "--lr", "--seed", "--warm", "--eval-every", "--out",
            "--distract", "--eval-seq", "--key-len", "--val-len", "--resume",
            "--needle-loss", "--mix-kl", "--mix-vl", "--mix-p", "--ans-w",
        ],
        "niah-eval" => &["--model", "--seq", "--n", "--distract", "--seed", "--key-len", "--val-len"],
        "memtest" => &["--steps"],
        "train" => &[
            "--data", "--max-bytes", "--seq", "--steps", "--lr", "--seed", "--arch",
            "--mtp-alpha", "--zloss-w", "--cap", "--wd", "--batch", "--holdout-bytes", "--bpc-every",
        ],
        "parity" => &["--data", "--max-bytes", "--seq", "--steps", "--lr", "--seed"],
        "generate" => &["--model", "--prompt", "--n"],
        "niah-gen" => &["--n", "--seq", "--key-len", "--val-len", "--distract", "--seed", "--out"],
        "transplant" => &["--pool", "--labels", "--out", "--steps", "--lr", "--seed", "--kd-weight", "--ans-w", "--eval-every", "--batch"],
        "pack" | "unpack" => &["--model", "--out"],
        "accel" => &[],
        "gradcheck" | "selftest" | "niah-overfit" | "info" | "doctor" | "version" | "logo" => &[],
        _ => &[],
    }
}

/// Reject unknown flags, missing values, stray positionals — before any work starts.
fn checked_validate(args: &[String], cmd: &str) -> Result<(), String> {
    let known = known_flags(cmd);
    let valueless = valueless_flags(cmd);
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if !a.starts_with("--") {
            return Err(format!("unexpected argument '{a}' for '{cmd}'"));
        }
        if !known.contains(&a.as_str()) {
            return Err(format!(
                "unknown option '{a}' for '{cmd}' (did you typo a flag? accepted: {})",
                known.join(" ")
            ));
        }
        if valueless.contains(&a.as_str()) {
            i += 1;
            continue;
        }
        match args.get(i + 1) {
            None => return Err(format!("option '{a}' is missing its value")),
            Some(v) if v.starts_with("--") => {
                return Err(format!("option '{a}' is missing its value (next token looks like another flag: '{v}')"))
            }
            Some(_) => {}
        }
        i += 2;
    }
    Ok(())
}

fn validate_args(args: &[String], cmd: &str) {
    if let Err(e) = checked_validate(args, cmd) {
        fail(e);
    }
}

fn parse_int(args: &[String], key: &str, default: u64) -> Result<u64, String> {
    match args.iter().position(|a| a == key) {
        None => Ok(default),
        Some(i) => args
            .get(i + 1)
            .ok_or_else(|| format!("{key}: missing value"))
            .and_then(|v| v.parse::<u64>().map_err(|_| format!("{key}: expected a non-negative integer, got '{v}'"))),
    }
}
fn parse_float(args: &[String], key: &str, default: f64) -> Result<f64, String> {
    match args.iter().position(|a| a == key) {
        None => Ok(default),
        Some(i) => args
            .get(i + 1)
            .ok_or_else(|| format!("{key}: missing value"))
            .and_then(|v| v.parse::<f64>().map_err(|_| format!("{key}: expected a number, got '{v}'"))),
    }
}
fn arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    match parse_int(args, key, default) {
        Ok(v) => v,
        Err(e) => fail(e),
    }
}
fn arg_f64(args: &[String], key: &str, default: f64) -> f64 {
    match parse_float(args, key, default) {
        Ok(v) => v,
        Err(e) => fail(e),
    }
}
fn arg_str<'a>(args: &'a [String], key: &str, default: &'a str) -> &'a str {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(|v| v.as_str())
        .unwrap_or(default)
}

fn ensure_model(path: &str) {
    if !std::path::Path::new(path).exists() {
        fail(format!(
            "model file not found: {path} — train it first, e.g. `tessera niah-train --out {path}`"
        ));
    }
}

fn ensure_data(path: &str, seq: usize) {
    let p = std::path::Path::new(path);
    if !p.exists() {
        fail(format!("data file not found: {path}"));
    }
    let len = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) as usize;
    if len < seq + 1 {
        fail(format!(
            "data file too small: {path} has {len} bytes, need at least seq+1 = {} bytes for --seq {seq}",
            seq + 1
        ));
    }
}

fn ensure_pos(name: &str, v: u64, min: u64) {
    if v < min {
        fail(format!("{name} must be >= {min}, got {v}"));
    }
}

/// NIAH sequences must fit the needle + query tail; reject too-small --seq up front.
fn ensure_niah_seq(cmd: &str, seq: usize, key_len: usize, val_len: usize) {
    let min = NiahCfg { key_len, val_len, ..NiahCfg::default() }.min_seq();
    if seq < min {
        fail(format!(
            "{cmd}: --seq {seq} is too small for --key-len {key_len} --val-len {val_len} (need >= {min})"
        ));
    }
}

fn create_parent(path: &str) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        if !dir.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                fail(format!("cannot create directory for '{path}': {e}"));
            }
        }
    }
}

/// Feasibility probe: overfit ONE fixed 512-token needle sample and eval it.
fn niah_overfit() {
    let mut model = Model::<f32>::new(ModelCfg::default(), 7);
    let mut gen = Niah::new(NiahCfg { seed: 42, n_distract: 0, ..NiahCfg::default() });
    let s = gen.sample(512);
    let toks: Vec<usize> = s.tokens.iter().map(|&b| b as usize).collect();
    let params = model.params();
    let mut opt = AdamW::new(&params, 0.01);
    for it in 0..400 {
        let lr = wsd_lr(it, 400, 3e-3);
        model.zero_grad();
        model.fwd_seq(&toks, true);
        let loss = model.bwd_seq(&s.tgt1, &s.tgt2, &s.mask);
        opt.clip(&params, 1.0);
        opt.step(&params, lr);
        if it % 50 == 0 || it == 399 {
            println!("overfit step {it:3}  loss {loss:.4}");
        }
    }
    // greedy recall on the same sample
    let mut prompt = s.tokens.clone();
    prompt.truncate(s.answer_start);
    let pt: Vec<usize> = prompt.iter().map(|&b| b as usize).collect();
    model.reset_state();
    let mut z = Vec::new();
    for &t in &pt {
        let (zz, _) = model.fwd_step(t, false);
        z = zz;
    }
    let mut pred = Vec::new();
    for _ in 0..s.answer.len() {
        let arg = z.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
        pred.push(arg as u8);
        let (zz, _) = model.fwd_step(arg, false);
        z = zz;
    }
    println!(
        "answer expected {:?} predicted {:?} -> {}",
        String::from_utf8_lossy(&s.answer),
        String::from_utf8_lossy(&pred),
        if pred == s.answer { "RECALL OK" } else { "RECALL FAILED" }
    );
}

fn train_niah(args: &[String]) {
    let seq = arg_u64(args, "--seq", 8192) as usize;
    let steps = arg_u64(args, "--steps", 300);
    let lr_peak = arg_f64(args, "--lr", 2e-3);
    let seed = arg_u64(args, "--seed", 7);
    let warm = arg_u64(args, "--warm", 30) as usize; // curriculum steps at short length
    let eval_every = arg_u64(args, "--eval-every", 25);
    let out = arg_str(args, "--out", "checkpoints/niah.bin").to_string();
    let distract = arg_u64(args, "--distract", 0) as usize;
    let resume = args.contains(&"--resume".to_string());
    let eval_seq = arg_u64(args, "--eval-seq", 0) as usize;
    let key_len = arg_u64(args, "--key-len", 4) as usize;
    let val_len = arg_u64(args, "--val-len", 4) as usize;
    let needle_loss = args.contains(&"--needle-loss".to_string());
    // annealed mixed-batch rehearsal: with decaying probability, draw the sample
    // from the PREVIOUS task config (--mix-kl/--mix-vl) so the old circuit stays
    // alive while the new one forms (curriculum interference fix)
    let mix_kl = arg_u64(args, "--mix-kl", 0) as usize;
    let mix_vl = arg_u64(args, "--mix-vl", 0) as usize;
    let mix_p0 = arg_f64(args, "--mix-p", 0.4);
    let ans_w = arg_f64(args, "--ans-w", 1.0);
    ensure_pos("--steps", steps, 1);
    ensure_pos("--seq", seq as u64, 1);
    if !lr_peak.is_finite() || lr_peak <= 0.0 {
        fail("--lr must be a finite number > 0".to_string());
    }
    if ans_w <= 0.0 {
        fail("--ans-w must be > 0".to_string());
    }
    if !(0.0..=1.0).contains(&mix_p0) {
        fail("--mix-p must be in [0, 1]".to_string());
    }
    ensure_niah_seq("niah-train", seq, key_len, val_len);
    if mix_kl > 0 {
        ensure_niah_seq("niah-train (mix)", seq, mix_kl, mix_vl);
    }
    if resume && !std::path::Path::new(&out).exists() {
        fail(format!(
            "cannot --resume: checkpoint '{out}' does not exist (omit --resume to start fresh)"
        ));
    }
    println!("== TESSERA NIAH trainer: seq={seq} steps={steps} lr={lr_peak} seed={seed} ({} SIMD) ==", simd_level_name());
    let cfg = ModelCfg::default();
    let mut model = Model::<f32>::new(cfg, seed);
    let nparams = model.count_params();
    println!("params: {nparams}  inference working state: {} B (substrate {} B)", model.state_bytes(), model.substrate_bytes());
    let params = model.params();
    let mut opt = AdamW::new(&params, 0.01);
    let mut gen = Niah::new(NiahCfg { n_distract: distract, key_len, val_len, ans_w, ..NiahCfg::default() });
    let mut gen_old = if mix_kl > 0 {
        Some(Niah::new(NiahCfg { n_distract: distract, key_len: mix_kl, val_len: mix_vl, ans_w, ..NiahCfg::default() }))
    } else {
        None
    };
    if resume {
        if let Err(e) = model.load(&out) {
            fail(format!("resume: cannot load checkpoint '{out}': {e}"));
        }
        println!("resumed from {out}");
        for p in model.params() {
            let bad = unsafe { (*p.w).iter().filter(|x| !x.to().is_finite()).count() };
            assert_eq!(bad, 0, "checkpoint {} contains non-finite weights in {}", out, p.name);
        }
    }

    let resolve_seq = |it: u64| -> usize {
        if warm > 0 && it < warm as u64 {
            // curriculum: 512 -> 2048 -> seq
            let frac = it as f64 / warm as f64;
            if frac < 0.4 { 512 } else if frac < 0.75 { 2048 } else { seq }
        } else {
            seq
        }
    };
    std::fs::create_dir_all("checkpoints").ok();
    let t0 = now_ms();
    for it in 0..steps {
        let sl = resolve_seq(it);
        let p_old = mix_p0 * (1.0 - it as f64 / steps as f64);
        let use_old = match gen_old.as_mut() {
            Some(g) => g.rng.f64() < p_old,
            None => false,
        };
        let mut s = if use_old {
            gen_old.as_mut().unwrap().sample(sl)
        } else {
            gen.sample(sl)
        };
        if needle_loss {
            // auxiliary supervision: predict the needle's own characters (the exact
            // inserted spans only — filler text stays unmasked)
            for &(a, b) in &s.needle_spans {
                for t in a..b.min(s.tokens.len() - 1) {
                    s.mask[t] = 1.0;
                }
            }
        }
        let toks: Vec<usize> = s.tokens.iter().map(|&b| b as usize).collect();
        let lr = wsd_lr(it, steps, lr_peak);
        model.zero_grad();
        model.fwd_seq(&toks, true);
        let loss = model.bwd_seq(&s.tgt1, &s.tgt2, &s.mask);
        {
            let mut first_bad: Option<(&str, usize)> = None;
            for p in params.iter() {
                let bad: usize = unsafe { (*p.g).iter().filter(|x| !x.to().is_finite()).count() };
                if bad > 0 && first_bad.is_none() {
                    first_bad = Some((p.name, bad));
                }
            }
            if let Some((nm, bad)) = first_bad {
                eprintln!("NON-FINITE GRADS after bwd at step {it}: first {} ({bad} elems), loss={loss}", nm);
                std::process::exit(2);
            }
        }
        if !loss.is_finite() {
            eprintln!("NON-FINITE LOSS at step {it}");
            for p in params.iter() {
                let bad: usize = unsafe { (*p.g).iter().filter(|x| !x.to().is_finite()).count() };
                if bad > 0 {
                    let mx = unsafe { (*p.g).iter().map(|x| x.to().abs()).fold(0.0f64, f64::max) };
                    eprintln!("  grad {}: {bad} non-finite (max abs {mx:.3e})", p.name);
                }
            }
            std::process::exit(2);
        }
        opt.clip(&params, 1.0);
        opt.step(&params, lr);
        if it % 5 == 0 || it == steps - 1 {
            let dt = (now_ms() - t0) as f64 / 1000.0;
            println!("step {it:5}  seq {sl:5}  loss {loss:.4}  lr {lr:.2e}  tok/s {:.0}", (it + 1) as f64 * sl as f64 / dt.max(1e-9));
        }
        if eval_every > 0 && (it + 1) % eval_every == 0 { // u64 ok
            let acc = tessera::data::niah_eval_model(&mut model, &mut gen, if eval_seq > 0 { eval_seq } else { seq }, 16);
            println!("  eval@{}: {}/16 correct", if eval_seq > 0 { eval_seq } else { seq }, acc);
        }
    }
    create_parent(&out);
    save_model(&mut model, &out);
    println!("saved -> {out}");
}

/// Load any supported checkpoint format: `.tem` (TEMSQ8 container) or the
/// legacy f32 `TSS1` dump — detected by magic, not extension.
fn open_model(path: &str) -> Model<f32> {
    let mut mg = [0u8; 8];
    {
        use std::io::Read;
        let mut f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => fail(format!("cannot open model '{path}': {e}")),
        };
        if let Err(e) = f.read_exact(&mut mg) {
            fail(format!("model '{path}' is too small to hold a header: {e}"));
        }
    }
    if mg == *tessera::tem::MAGIC {
        return tessera::tem::load(path).unwrap_or_else(|e| fail(format!("cannot load model '{path}': {e}")));
    }
    if &mg[0..4] != *b"TSS1" {
        fail(format!(
            "'{path}' is not a tessera checkpoint (unknown header {:02x?}; expected .tem or TSS1)",
            mg
        ));
    }
    let mut model = Model::<f32>::new(ModelCfg::default(), 0);
    if let Err(e) = model.load(path) {
        fail(format!("cannot load model '{path}': {e}"));
    }
    model
}

/// Save honoring the extension: `*.tem` -> quantized container, else legacy f32.
fn save_model(model: &mut Model<f32>, out: &str) {
    if out.ends_with(".tem") {
        tessera::tem::save(model, out).unwrap_or_else(|e| fail(format!("cannot save checkpoint '{out}': {e}")));
    } else {
        model.save(out).unwrap_or_else(|e| fail(format!("cannot save checkpoint '{out}': {e}")));
    }
}

fn fmt_b(b: u64) -> String {
    if b >= 1 << 20 {
        format!("{:.2} MB", b as f64 / (1 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KB", b as f64 / (1 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

fn cmd_niah_eval(args: &[String]) {
    let seq = arg_u64(args, "--seq", 8192) as usize;
    let n = arg_u64(args, "--n", 64) as usize;
    let path = arg_str(args, "--model", "checkpoints/niah.bin");
    let distract = arg_u64(args, "--distract", 0) as usize;
    let gseed = arg_u64(args, "--seed", 999_123);
    let key_len = arg_u64(args, "--key-len", 4) as usize;
    let val_len = arg_u64(args, "--val-len", 4) as usize;
    ensure_pos("--n", n as u64, 1);
    ensure_pos("--seq", seq as u64, 1);
    ensure_niah_seq("niah-eval", seq, key_len, val_len);
    ensure_model(path);
    let mut model = open_model(path);
    let mut gen = Niah::new(NiahCfg { seed: gseed, n_distract: distract, key_len, val_len, ..NiahCfg::default() });
    let t0 = now_ms();
    let correct = tessera::data::niah_eval_model(&mut model, &mut gen, seq, n);
    let dt = (now_ms() - t0) as f64 / 1000.0;
    println!("NIAH@{seq}: {correct}/{n} = {:.1}%  ({:.1} samples/s)", 100.0 * correct as f64 / n as f64, n as f64 / dt);
}

/// Held-out quality: streaming (O(1)) next-byte CE on `bytes`, in bits/char.
/// Chunks of `seq+1` bytes, state reset per chunk (mirrors training contexts).
fn eval_bpc(model: &mut Model<f32>, bytes: &[u8], seq: usize) -> f64 {
    let ln2 = std::f64::consts::LN_2;
    let mut nats = 0.0f64;
    let mut count = 0usize;
    let mut i = 0;
    while i + 1 < bytes.len() {
        let end = (i + seq + 1).min(bytes.len());
        model.reset_state();
        let mut z: Vec<f32> = Vec::new();
        for &b in &bytes[i..end] {
            let (zz, _) = model.fwd_step(b as usize, false);
            if !z.is_empty() {
                // CE of `b` given previous logits z
                let mut m = f64::NEG_INFINITY;
                for &x in z.iter() {
                    m = m.max(x.to() as f64);
                }
                let mut lse = 0.0f64;
                for &x in z.iter() {
                    lse += (x.to() as f64 - m).exp();
                }
                lse = m + lse.ln();
                nats += lse - z[b as usize].to() as f64;
                count += 1;
            }
            z = zz;
        }
        i = end;
    }
    if count == 0 {
        return f64::NAN;
    }
    nats / ln2 / count as f64
}

fn train_lm(args: &[String], arch: &str) {
    let data = arg_str(args, "--data", "data/enwik8");
    let maxb = arg_u64(args, "--max-bytes", 20_000_000) as usize;
    let seq = arg_u64(args, "--seq", 1024) as usize;
    let steps = arg_u64(args, "--steps", 200);
    let seed = arg_u64(args, "--seed", 11);
    let mut lr_peak = arg_f64(args, "--lr", -1.0);
    let mtp_alpha = arg_f64(args, "--mtp-alpha", -1.0);
    let zloss_w = arg_f64(args, "--zloss-w", -1.0);
    let cap = arg_f64(args, "--cap", -1.0);
    let wd_raw = arg_f64(args, "--wd", -1.0);
    // tuned per-arch defaults (enwik8 quality sweep, see README):
    // tessera tolerates high lr thanks to QK-norm; the plain transformer needs 5e-4.
    if lr_peak < 0.0 {
        lr_peak = if arch == "tessera" { 3e-3 } else { 5e-4 };
    }
    let wd = if wd_raw < 0.0 { 0.1 } else { wd_raw };
    let batch = arg_u64(args, "--batch", 1) as usize;
    let holdout = arg_u64(args, "--holdout-bytes", 32_768) as usize;
    let bpc_every = arg_u64(args, "--bpc-every", 150) as u64;
    ensure_pos("--steps", steps, 1);
    ensure_pos("--seq", seq as u64, 1);
    ensure_pos("--batch", batch as u64, 1);
    if arch != "tessera" && arch != "transformer" {
        fail(format!("unknown arch '{arch}' (expected 'tessera' or 'transformer')"));
    }
    ensure_data(data, seq);
    if !lr_peak.is_finite() || lr_peak <= 0.0 {
        fail("--lr must be a finite number > 0".to_string());
    }
    if wd < 0.0 {
        fail("--wd must be >= 0".to_string());
    }
    let mut quality_cfg = ModelCfg::default();
    let mtp_alpha_eff = if mtp_alpha >= 0.0 { mtp_alpha } else if arch == "tessera" { 0.1 } else { quality_cfg.mtp_alpha };
    quality_cfg.mtp_alpha = mtp_alpha_eff;
    if mtp_alpha >= 0.0 {
        quality_cfg.mtp_alpha = mtp_alpha;
    }
    if zloss_w >= 0.0 {
        quality_cfg.zloss_w = zloss_w;
    }
    if cap > 0.0 {
        quality_cfg.cap = cap;
    }
    let _ = &mut quality_cfg;
    println!(
        "== LM trainer ({arch}) data={data} seq={seq} steps={steps} batch={batch} wd={wd}          alpha={} zloss={} ==\n",
        quality_cfg.mtp_alpha, quality_cfg.zloss_w
    );
    let mut corpus = match Corpus::from_file(data, maxb) {
        Ok(c) => c,
        Err(e) => fail(format!("cannot read corpus '{data}': {e}")),
    };
    // held-out bytes: everything past the training window (disjoint by construction)
    let holdout_bytes: Vec<u8> = {
        let all = std::fs::read(data).unwrap_or_default();
        if all.len() > maxb {
            all[maxb..(maxb + holdout).min(all.len())].to_vec()
        } else {
            Vec::new()
        }
    };
    let eval_holdout = |m: &mut Model<f32>| -> f64 {
        if holdout_bytes.is_empty() {
            f64::NAN
        } else {
            eval_bpc(m, &holdout_bytes, seq)
        }
    };
    let t0 = now_ms();
    let mut toks_done = 0u64;
    match arch {
        "tessera" => {
            let mut model = Model::<f32>::new(quality_cfg, seed);
            println!("params: {}", model.count_params());
            let params = model.params();
            let mut opt = AdamW::new(&params, wd);
            for it in 0..steps {
                let lr = wsd_lr(it, steps, lr_peak);
                model.zero_grad();
                let mut loss = 0.0f64;
                let mut mask_sum = 0.0f64;
                for _ in 0..batch {
                    let chunk = corpus.next_chunk(seq + 1);
                    let (tgt1, tgt2, mask) = lm_targets(&chunk);
                    mask_sum += mask[..seq].iter().sum::<f64>();
                    let toks = chunk[..seq].to_vec();
                    model.fwd_seq(&toks, true);
                    loss += model.bwd_seq(&tgt1[..seq], &tgt2[..seq], &mask[..seq]);
                }
                opt.clip(&params, 1.0);
                opt.step(&params, lr);
                toks_done += (seq * batch) as u64;
                if it % 10 == 0 || it == steps - 1 {
                    let bpc = loss / std::f64::consts::LN_2 / mask_sum.max(1.0);
                    println!("step {it:5}  loss/tok {loss:.1}  bpc~{bpc:.3}  lr {lr:.2e}");
                }
                if bpc_every > 0 && (it as u64 + 1) % bpc_every == 0 {
                    let h = eval_holdout(&mut model);
                    println!("  holdout BPC @ step {it}: {h:.4}");
                }
            }
            let h = eval_holdout(&mut model);
            println!("FINAL holdout BPC: {h:.4}");
        }
        "transformer" => {
            let mut model = tessera::baseline::Baseline::<f32>::new(tessera::baseline::BaselineCfg::default(), seed);
            println!("params: {}", model.count_params());
            let params = model.params();
            let mut opt = AdamW::new(&params, wd);
            // Baseline has no streaming state to reuse for held-out eval; wrap the
            // bytes through its forward by chunk (same protocol as tessera's eval).
            let eval_holdout_t = |m: &mut tessera::baseline::Baseline<f32>, bytes: &[u8]| -> f64 {
                if bytes.is_empty() {
                    return f64::NAN;
                }
                let ln2 = std::f64::consts::LN_2;
                let mut nats = 0.0f64;
                let mut count = 0usize;
                let mut i = 0;
                while i + 1 < bytes.len() {
                    let end = (i + seq + 1).min(bytes.len());
                    let chunk: Vec<usize> = bytes[i..end].iter().map(|&b| b as usize).collect();
                    let (tgt1, _t2, _m) = lm_targets(&chunk);
                    let ce = m.eval_ce(&chunk[..], &tgt1[..]);
                    nats += ce;
                    count += chunk.len().saturating_sub(1);
                    i = end;
                }
                if count == 0 { f64::NAN } else { nats / ln2 / count as f64 }
            };
            for it in 0..steps {
                let chunk = corpus.next_chunk(seq + 1);
                let (tgt1, _tgt2, mask) = lm_targets(&chunk);
                let lr = wsd_lr(it, steps, lr_peak);
                model.zero_grad();
                let loss = model.fwd_bwd(&chunk[..seq], &tgt1[..seq], &mask[..seq]);
                opt.clip(&params, 1.0);
                opt.step(&params, lr);
                toks_done += seq as u64;
                if it % 10 == 0 || it == steps - 1 {
                    let bpc = loss / std::f64::consts::LN_2 / mask[..seq].iter().sum::<f64>().max(1.0);
                    println!("step {it:5}  loss/tok {loss:.1}  bpc~{bpc:.3}  lr {lr:.2e}");
                }
                if bpc_every > 0 && (it as u64 + 1) % bpc_every == 0 {
                    let h = eval_holdout_t(&mut model, &holdout_bytes);
                    println!("  holdout BPC @ step {it}: {h:.4}");
                }
            }
            let h = eval_holdout_t(&mut model, &holdout_bytes);
            println!("FINAL holdout BPC: {h:.4}");
        }
        _ => fail(format!("unknown arch '{arch}' (expected 'tessera' or 'transformer')")),
    }
    let dt = (now_ms() - t0) as f64 / 1000.0;
    println!("throughput: {:.0} tok/s  ({dt:.1}s total)", toks_done as f64 / dt);
}

fn parity(args: &[String]) {
    // Parameter-matched parity: trains TESSERA and a standard causal Transformer with
    // FFN width adjusted so total parameter counts match within ~1%.
    let data = arg_str(args, "--data", "data/enwik8");
    let maxb = arg_u64(args, "--max-bytes", 20_000_000) as usize;
    let seq = arg_u64(args, "--seq", 1024) as usize;
    let steps = arg_u64(args, "--steps", 200);
    let lr_peak = arg_f64(args, "--lr", 1e-3);
    let seed = arg_u64(args, "--seed", 11);
    ensure_pos("--steps", steps, 1);
    ensure_pos("--seq", seq as u64, 1);
    ensure_data(data, seq);
    println!("== PARITY: TESSERA vs parameter-matched Transformer ({seq}-token chunks, {steps} steps) ==");
    let tcfg = ModelCfg::default();
    let tessera_params = {
        let mut m = Model::<f32>::new(tcfg, seed);
        m.count_params()
    };
    let mut bcfg = tessera::baseline::BaselineCfg::default();
    // adjust baseline FF so param counts match (each ff unit ~ 3d params per layer)
    let mut b = tessera::baseline::Baseline::<f32>::new(bcfg, seed);
    let base_params = b.count_params();
    let delta = tessera_params as i64 - base_params as i64;
    let dff_adj = (delta as f64 / (3.0 * tcfg.d as f64 * tcfg.layers as f64)).round() as i64;
    bcfg.ff = (bcfg.ff as i64 + dff_adj).max(8) as usize;
    b = tessera::baseline::Baseline::<f32>::new(bcfg, seed);
    println!(
        "tessera params {} | transformer params {} (ff={})",
        tessera_params,
        b.count_params(),
        bcfg.ff
    );
    let mut corpus_a = match Corpus::from_file(data, maxb) {
        Ok(c) => c,
        Err(e) => fail(format!("cannot read corpus '{data}': {e}")),
    };
    let mut corpus_b = match Corpus::from_file(data, maxb) {
        Ok(c) => c,
        Err(e) => fail(format!("cannot read corpus '{data}': {e}")),
    };
    let mut model_t = Model::<f32>::new(tcfg, seed);
    let p_t = model_t.params();
    let mut opt_t = AdamW::new(&p_t, 0.01);
    let params_b = b.params();
    let mut opt_b = AdamW::new(&params_b, 0.01);
    let mut lt;
    let mut lb;
    let t0 = now_ms();
    let (mut tok_t, mut tok_b) = (0u64, 0u64);
    for it in 0..steps {
        let lr = wsd_lr(it, steps, lr_peak);
        // tessera
        {
            let chunk = corpus_a.next_chunk(seq + 1);
            let (tgt1, tgt2, mask) = lm_targets(&chunk);
            model_t.zero_grad();
            model_t.fwd_seq(&chunk[..seq], true);
            lt = model_t.bwd_seq(&tgt1[..seq], &tgt2[..seq], &mask[..seq]);
            opt_t.clip(&p_t, 1.0);
            opt_t.step(&p_t, lr);
            tok_t += seq as u64;
        }
        // transformer (same data stream position)
        {
            let chunk = corpus_b.next_chunk(seq + 1);
            let (tgt1, _t2, mask) = lm_targets(&chunk);
            b.zero_grad();
            lb = b.fwd_bwd(&chunk[..seq], &tgt1[..seq], &mask[..seq]);
            opt_b.clip(&params_b, 1.0);
            opt_b.step(&params_b, lr);
            tok_b += seq as u64;
        }
        if it % 20 == 0 || it == steps - 1 {
            println!(
                "step {it:5}  tessera loss {lt:9.3} | transformer loss {lb:9.3}   (bpc: {:.3} vs {:.3})",
                lt / std::f64::consts::LN_2 / seq as f64,
                lb / std::f64::consts::LN_2 / seq as f64
            );
        }
    }
    let dt = (now_ms() - t0) as f64 / 1000.0;
    println!(
        "speedup: tessera {:.0} tok/s vs transformer {:.0} tok/s  ->  {:.2}x",
        tok_t as f64 / dt,
        tok_b as f64 / dt,
        tok_t as f64 / tok_b as f64
    );
}

fn memtest(args: &[String]) {
    let steps = arg_u64(args, "--steps", 100_000);
    ensure_pos("--steps", steps, 1);
    let seq_short = 64usize;
    let mut model = Model::<f32>::new(ModelCfg::default(), 3);
    // run 64 tokens, record footprint
    let mut rng = tessera::rng::Rng::new(5);
    let run = |model: &mut Model<f32>, n: usize, rng: &mut tessera::rng::Rng| {
        model.reset_state();
        for _ in 0..n {
            let tk = rng.below(model.cfg.vocab);
            model.fwd_step(tk, false);
        }
        model.state_bytes()
    };
    let b_short = run(&mut model, seq_short, &mut rng);
    println!("inference working state @ T={seq_short:>6}: {b_short} B (substrate {} B)", model.substrate_bytes());
    let b_long = run(&mut model, steps as usize, &mut rng);
    println!("inference working state @ T={steps:>6}: {b_long} B (substrate {} B)", model.substrate_bytes());
    assert_eq!(b_short, b_long, "O(1) memory invariance violated");
    println!("PASS: O(1) memory invariance — footprint identical at T=64 and T={steps}");
    println!("substrate: {} B ~= {:.1} KiB (spec target ~145 KB)", model.substrate_bytes(), model.substrate_bytes() as f64 / 1024.0);
}

fn generate(args: &[String]) {
    let path = arg_str(args, "--model", "checkpoints/niah.bin");
    let prompt = arg_str(args, "--prompt", "The key is K7Q2 ? ");
    let n = arg_u64(args, "--n", 40) as usize;
    ensure_pos("--n", n as u64, 1);
    ensure_model(path);
    let mut model = open_model(path);
    model.reset_state();
    let bytes = prompt.as_bytes();
    let mut last = 0usize;
    for &b in bytes {
        let (z, _) = model.fwd_step(b as usize, false);
        last = z.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    }
    let mut out = String::new();
    for _ in 0..n {
        out.push(last as u8 as char);
        let (z, _) = model.fwd_step(last, false);
        last = z.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    }
    println!("prompt: {prompt}\ncontinuation: {out}");
}

fn selftest() {
    println!("SIMD level: {}", simd_level_name());
    // tiny overfit: same needle every time, T=96 — must reach near-zero loss quickly
    let cfg = ModelCfg { d: 32, m: 16, ff: 64, fine: 24, coarse: 4, ..ModelCfg::default() };
    let mut model = Model::<f32>::new(cfg, 1);
    let mut gen = Niah::new(NiahCfg { seed: 42, ..NiahCfg::default() });
    let params = model.params();
    let mut opt = AdamW::new(&params, 0.0);
    let seq = 96;
    let s = gen.sample(seq);
    let toks: Vec<usize> = s.tokens.iter().map(|&b| b as usize).collect();
    let mut first = 0.0;
    let mut last = 0.0;
    for it in 0..400 {
        model.zero_grad();
        model.fwd_seq(&toks, true);
        let loss = model.bwd_seq(&s.tgt1, &s.tgt2, &s.mask);
        opt.clip(&params, 1.0);
        opt.step(&params, 3e-3);
        if it == 0 {
            first = loss;
        }
        last = loss;
    }
    println!("overfit loss: {first:.3} -> {last:.3} (target < 0.4)");
    assert!(last < 0.4, "selftest overfit failed: {last}");
    println!("SELFTEST PASS");
}

// ────────────────────────────────────────────────────────────────────────────
//  New CLI surface: version / info / doctor / niah-gen / transplant, per-cmd
//  help, typo suggestions. ("transplant" = teacher soft-labels → Tessera,
//  the LoLCATs-style distill-into-new-architecture pipeline, Section X of
//  tessera_vs_gemma.py is the Gemma-side labeler.)
// ────────────────────────────────────────────────────────────────────────────

const COMMANDS: &[&str] = &[
    "help", "version", "info", "doctor", "gradcheck", "selftest", "memtest",
    "niah-train", "niah-overfit", "niah-eval", "niah-gen", "train", "parity",
    "generate", "transplant", "pack", "unpack", "logo", "accel",
];

fn version_line() -> String {
    format!(
        "tessera {} (SIMD: {}, {}-{}, profile: {})",
        env!("CARGO_PKG_VERSION"),
        simd_level_name(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        if cfg!(debug_assertions) { "debug" } else { "release" }
    )
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn suggest(cmd: &str) -> Option<&'static str> {
    let mut best: Option<(usize, &'static str)> = None;
    for c in COMMANDS {
        if *c == cmd {
            continue;
        }
        let d = levenshtein(cmd, c);
        if best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, c));
        }
    }
    let (d, c) = best?;
    let prefix: String = cmd.chars().take(4).collect();
    if d <= 2 || c.chars().take(4).eq(prefix.chars()) {
        Some(c)
    } else {
        None
    }
}

fn cmd_help_text(cmd: &str) -> Option<&'static str> {
    Some(match cmd {
        "help" => "tessera help [command]\n\nusage:\n  tessera help [command]\n\n  show the overview, or detailed help for one command.\n  try: tessera help transplant\n\nexit codes: 0 success | 2 usage error (unknown command)",
        "version" => "tessera version\n\nusage:\n  tessera version\n\n  print the version line: crate version, active SIMD path, platform, build profile.\n  also accepts: tessera --version | -V\n\nexit codes: 0 success",
        "info" => "tessera info\n\nusage:\n  tessera info\n\n  build + model summary table: version, SIMD path, parameter count,\n  O(1) working-state bytes, substrate bytes, architecture constants.\n\n  exit codes: 0 success | 2 usage error | 1 runtime failure",
        "doctor" => "tessera doctor\n\nusage:\n  tessera doctor\n\n  environment checks (each printed as [ ok ] / [FAIL] / [warn]):\n    - one real forward pass through the whole model (finite logits)\n    - RNG stream determinism\n    - checkpoints/ directory writable\n    - enwik8 corpus present? (only needed by train/parity)\n    - flag validator rejects junk\n\n  exit 1 if any hard check fails. exit codes: 0 ok | 1 runtime failure",
        "gradcheck" => "tessera gradcheck\n\nusage:\n  tessera gradcheck\n\n  finite-difference verification of all 9 pillars (f64 reference path).\n  prints one line per check, then 'GRADCHECK PASS' on success.\n\n  exit codes: 0 pass | 1 any check above threshold",
        "selftest" => "tessera selftest\n\nusage:\n  tessera selftest\n\n  quick end-to-end learning sanity check. prints 'SELFTEST PASS'.\n\n  exit codes: 0 pass | 1 failure",
        "memtest" => "tessera memtest [--steps N]\n\nusage:\n  tessera memtest [--steps N]\n\n  O(1) memory-invariance proof: runs N tokens (default 100000) and asserts\n  the inference working state is byte-identical at every length.\n  prints the state/substrate lines and 'PASS'.\n\n  exit codes: 0 pass | 1 failure",
        "niah-train" => "tessera niah-train — train the NIAH store+recall circuit\n\nusage:\n  tessera niah-train [--seq T] [--steps N] [--lr F] [--seed S] [--out PATH]\n                     [--key-len K] [--val-len K] [--distract K] [--ans-w F]\n                     [--warm N] [--eval-every N] [--eval-seq T]\n                     [--needle-loss] [--resume]\n                     [--mix-kl K --mix-vl K --mix-p F]\n\nflags (default):\n  --seq        sequence length (8192)\n  --steps      optimizer steps (300)\n  --lr         peak WSD learning rate (2e-3)\n  --seed       RNG seed (7)\n  --out        checkpoint path (checkpoints/niah.bin)\n  --key-len    needle key bytes (4)\n  --val-len    needle value bytes (4)\n  --distract   extra decoy needles (0)\n  --ans-w      loss weight on answer positions (1.0); recipes often want >= 5\n  --warm       curriculum steps 512->2048->seq (30)\n  --eval-every evaluate every N steps on 16 fresh prompts (25; 0 disables)\n  --eval-seq   eval at a different length than --seq (0 = same)\n  --needle-loss  also supervise the needle-span characters\n  --resume     continue from the --out checkpoint\n  --mix-kl/--mix-vl/--mix-p  annealed rehearsal of a previous task shape\n\nexit codes: 0 success | 2 usage/validation | 1 runtime",
        "niah-overfit" => "tessera niah-overfit\n\nusage:\n  tessera niah-overfit\n\n  single-sample store+recall probe (RECALL OK / RECALL FAILED).\n\n  exit codes: 0 pass | 1 failure",
        "niah-eval" => "tessera niah-eval — fresh-prompt recall of a trained checkpoint\n\nusage:\n  tessera niah-eval [--model PATH] [--seq T] [--n N] [--key-len K]\n                    [--val-len K] [--distract K] [--seed S]\n\nflags (default):\n  --model    checkpoint (checkpoints/niah.bin)\n  --seq      prompt length (8192)\n  --n        number of fresh prompts (64)\n  --key-len  needle key bytes (4)\n  --val-len  needle value bytes (4)\n  --distract decoy needles (0)\n  --seed     eval seed (999123)\n\nprints 'NIAH@<seq>: <correct>/<n> = <pct>%  (<rate> samples/s)'.\nexit codes: 0 success | 2 usage/validation | 1 runtime",
        "niah-gen" => "tessera niah-gen — export a NIAH prompt pool (transplant step 1)\n\nusage:\n  tessera niah-gen [--n N] [--seq T] [--key-len K] [--val-len K]\n                   [--distract K] [--seed S] [--out PATH]\n\nflags (default):\n  --n        number of prompts (32, max 1000000)\n  --seq      sequence length (128)\n  --key-len  needle key bytes (1)\n  --val-len  needle value bytes (1)\n  --distract decoy needles (0)\n  --seed     pool RNG seed (4321, stored as u32)\n  --out      output path (niah_pool.bin)\n\nWrites a 'TSPOOLv1' binary: header (n, seq, kl, vl, distract, seed),\nthe prompts, the hidden answers, and the answer offsets. Give the pool to\nthe teacher (Gemma, Section X of tessera_vs_gemma.py) to produce labels,\nthen run `tessera transplant`.\n\nexit codes: 0 success | 2 usage/validation | 1 runtime",
        "transplant" => "tessera transplant — distill teacher soft-labels into Tessera (step 3)\n\nusage:\n  tessera transplant --pool PATH --labels PATH [--out PATH]\n                     [--steps N] [--lr F] [--seed S]\n                     [--kd-weight F] [--ans-w F] [--eval-every N] [--batch N]\n\nflags (default):\n  --pool       prompt pool from `tessera niah-gen` (required)\n  --labels     teacher soft-labels 'TSLABLv1' from the Gemma side (required):\n               header (n, val_len, ncls=32), then n x val_len x 32 f32 probs\n  --out        checkpoint path (checkpoints/transplanted.bin)\n  --steps      optimizer steps (1500)\n  --batch      grad-accumulation: pool samples per step (16; the validated recipe)\n  --lr         peak WSD learning rate (5e-4)\n  --seed       RNG seed (7)\n  --kd-weight  teacher loss weight on answer positions (0.7, range [0, 4])\n  --ans-w      hard-CE answer weight (5.0 — the validated recipe)\n  --eval-every fresh-prompt eval cadence (250; 0 disables)\n\nloss = ans-w-weighted hard CE + kd-weight x KL(teacher || alphabet softmax)\nseeded through the same analytical BPTT (bwd_seq_kd).\nValidates pool/label shapes and probability rows with friendly errors.\n\nexit codes: 0 success | 2 usage/validation | 1 runtime",
        "train" => "tessera train — byte-level LM training on a corpus\n\nusage:\n  tessera train --data PATH [--arch tessera|transformer] [--steps N]\n                [--seq T] [--max-bytes N] [--batch N] [--holdout-bytes N]\n                [--bpc-every N] [--lr F] [--seed S] [--mtp-alpha F]\n                [--zloss-w F] [--cap F] [--wd F]\n\nflags (default):\n  --data           corpus path (data/enwik8)\n  --arch           tessera | transformer (tessera)\n  --steps          optimizer steps (200)\n  --seq            sequence length (1024)\n  --max-bytes      corpus bytes to use (20000000)\n  --batch          grad-accumulation batches (1)\n  --holdout-bytes  held-out tail for BPC eval (32768)\n  --bpc-every      holdout eval cadence (150; 0 = only at the end)\n  --lr             peak WSD lr (-1 = arch default: tessera 3e-3 / transformer 5e-4)\n  --seed           RNG seed (11)\n  --mtp-alpha      dual-head MTP weight (-1 = spec default)\n  --zloss-w        z-loss weight (-1 = spec default 1e-4)\n  --cap            soft-cap constant (-1 = spec default 30)\n  --wd             AdamW weight decay (-1 = spec default)\n\nprints 'FINAL holdout BPC: x.xxxx' and 'throughput: ... tok/s'.\nexit codes: 0 success | 2 usage/validation | 1 runtime",
        "pack" => "tessera pack — convert a checkpoint into the .tem container\n\nusage:\n  tessera pack --model PATH [--out PATH]\n\nflags (default):\n  --model   checkpoint to convert (required; legacy .bin / TSS1)\n  --out     output path (same stem with the .tem extension)\n\n.tem = TESSERA Embedded Model: 8-bit grouped quantization (~4x smaller\nthan the f32 dump), FNV-1a64 integrity over the payload, and a\nzero-decoder load that streams at memory bandwidth. Every --model flag\n(niah-eval, generate, unpack) accepts .tem files directly.\n\nexit codes: 0 success | 2 usage/validation | 1 runtime",
        "unpack" => "tessera unpack — convert a .tem container back to a legacy checkpoint\n\nusage:\n  tessera unpack --model PATH.tem [--out PATH]\n\nflags (default):\n  --model   .tem file to convert (required)\n  --out     output path (same stem with the .bin extension)\n\nVerifies the FNV-1a64 checksum, dequantizes, and writes the f32 TSS1\nformat. (Round-trips are not bit-exact: quantization is lossy by design.)\n\nexit codes: 0 success | 2 usage/validation | 1 runtime",
        "accel" => "tessera accel — accelerator selftest (T4 CUDA + TPU bridge)\n\nusage:\n  tessera accel\n\nCUDA (T4): probes the GPU through the CUDA Driver API loaded at runtime\n(dlopen, zero deps), does a bit-exact 1 MiB H2D/D2H roundtrip, then JITs\ntwo embedded PTX kernels (vadd + 64x64x64 matmul) and verifies them\nagainst Rust-computed references.\n\nTPU: native PJRT handshake — Rust dlopens the TPU runtime plugin\n(libtpu.so / pjrt_c_api_tpu_plugin.so, TESSERA_TPU_LIB overrides the\nsearch), initializes the plugin, creates a PJRT_Client (the same ABI\nJAX/TF/PyTorch-XLA use) and enumerates the attached TPU chips. Zero\nPython, zero PyTorch. HLO buffer/execute is the phase-2 milestone.\n\nAbsent hardware is [skip], never a failure. Any present-but-broken\nhardware is [FAIL] and the command exits 1.\n\nKaggle notes: T4 needs nothing extra; a TPU VM needs rustup (one curl\nline) and ships libtpu already.\n\nexit codes: 0 pass (skips allowed) | 1 any present device failed",
        "logo" => "tessera logo\n\nusage:\n  tessera logo\n\n  print the tessera mosaic mark (truecolor on a terminal, plain when piped).\n\nexit codes: 0 success",
        "parity" => "tessera parity — parameter-matched tessera-vs-transformer duel\n\nusage:\n  tessera parity --data PATH [--max-bytes N] [--seq T] [--steps N]\n                 [--lr F] [--seed S]\n\nprints 'tessera params ... | transformer params ...', per-step losses for\nboth, and the 'speedup:' line.\n\nexit codes: 0 success | 2 usage/validation | 1 runtime",
        "generate" => "tessera generate — greedy continuation from a checkpoint\n\nusage:\n  tessera generate [--model PATH] [--prompt TEXT] [--n N]\n\nflags (default):\n  --model   checkpoint (checkpoints/niah.bin)\n  --prompt  seed text ('The key is K7Q2 ? ')\n  --n       tokens to generate (40)\n\nprints 'continuation: ...'.\nexit codes: 0 success | 2 usage/validation | 1 runtime",
        _ => return None,
    })
}

// ─── pool / label file formats (the transplant pipeline) ────────────────────
// pool:   "TSPOOLv1" | u32 n | u32 seq | u32 kl | u32 vl | u32 distract | u32 seed
//         | n x seq prompt bytes | n x vl answer bytes | n x u32 answer_start
// labels: "TSLABLv1" | u32 n | u32 vl | u32 ncls | n x vl x ncls f32 probs
const POOL_MAGIC: &[u8; 8] = b"TSPOOLv1";
const LABEL_MAGIC: &[u8; 8] = b"TSLABLv1";

struct Pool {
    #[allow(dead_code)] // part of the on-disk format; read during validation
    n: usize,
    seq: usize,
    key_len: usize,
    val_len: usize,
    n_distract: usize,
    seed: u64,
    prompts: Vec<Vec<u8>>,
    answers: Vec<Vec<u8>>,
    #[allow(dead_code)] // part of the on-disk format; validated on load
    starts: Vec<usize>,
}

fn rd_u32(raw: &[u8], o: usize, what: &str) -> usize {
    if o + 4 > raw.len() {
        fail(format!("file truncated: missing {what} (offset {o})"));
    }
    u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize
}

fn load_pool(path: &str) -> Pool {
    let raw = std::fs::read(path)
        .unwrap_or_else(|e| fail(format!("cannot read pool '{path}': {e}")));
    if raw.len() < 8 || &raw[0..8] != POOL_MAGIC {
        fail(format!(
            "'{path}' is not a Tessera pool file (magic mismatch; expected 'TSPOOLv1'). Generate one: tessera niah-gen --out {path}"
        ));
    }
    let n = rd_u32(&raw, 8, "n");
    let seq = rd_u32(&raw, 12, "seq");
    let key_len = rd_u32(&raw, 16, "key_len");
    let val_len = rd_u32(&raw, 20, "val_len");
    let n_distract = rd_u32(&raw, 24, "n_distract");
    let seed = rd_u32(&raw, 28, "seed") as u64;
    if n == 0 {
        fail(format!("pool '{path}' contains 0 prompts"));
    }
    if n > 1_000_000 {
        fail(format!("pool '{path}' claims n={n} — corrupt header"));
    }
    if seq == 0 || val_len == 0 {
        fail(format!("pool '{path}': seq and val_len must be >= 1"));
    }
    let need = 32usize + n * seq + n * val_len + n * 4;
    if raw.len() < need {
        fail(format!(
            "pool '{path}' is truncated: have {} B, need {need} B (n={n}, seq={seq}, val_len={val_len})",
            raw.len()
        ));
    }
    let mut o = 32usize;
    let mut prompts = Vec::with_capacity(n);
    for _ in 0..n {
        prompts.push(raw[o..o + seq].to_vec());
        o += seq;
    }
    let mut answers = Vec::with_capacity(n);
    for _ in 0..n {
        answers.push(raw[o..o + val_len].to_vec());
        o += val_len;
    }
    let mut starts = Vec::with_capacity(n);
    for _ in 0..n {
        starts.push(rd_u32(&raw, o, "answer_start"));
        o += 4;
    }
    for (si, st) in starts.iter().enumerate() {
        if *st < val_len || *st + val_len > seq {
            fail(format!(
                "pool '{path}': sample {si} answer_start {st} is out of range (seq={seq}, val_len={val_len}) — corrupt pool"
            ));
        }
    }
    Pool { n, seq, key_len, val_len, n_distract, seed, prompts, answers, starts }
}

struct Labels {
    n: usize,
    val_len: usize,
    ncls: usize,
    probs: Vec<Vec<Vec<f32>>>, // [sample][answer byte][class]
}

fn load_labels(path: &str, want_n: usize, want_vl: usize) -> Labels {
    let ncls_expected = tessera::data::key_alphabet().len();
    let raw = std::fs::read(path)
        .unwrap_or_else(|e| fail(format!("cannot read labels '{path}': {e}")));
    if raw.len() < 8 || &raw[0..8] != LABEL_MAGIC {
        fail(format!(
            "'{path}' is not a Tessera label file (magic mismatch; expected 'TSLABLv1'). The teacher labels a pool via Section X of tessera_vs_gemma.py."
        ));
    }
    let n = rd_u32(&raw, 8, "n");
    let vl = rd_u32(&raw, 12, "val_len");
    let ncls = rd_u32(&raw, 16, "ncls");
    if ncls != ncls_expected {
        fail(format!(
            "labels '{path}': ncls={ncls} but the retrieval alphabet has {ncls_expected} classes"
        ));
    }
    if n != want_n || vl != want_vl {
        fail(format!(
            "labels '{path}' shape ({n} samples x val_len {vl}) does not match the pool ({want_n} samples x val_len {want_vl})"
        ));
    }
    let need = 20usize + n * vl * ncls * 4;
    if raw.len() < need {
        fail(format!(
            "labels '{path}' is truncated: have {} B, need {need} B",
            raw.len()
        ));
    }
    let mut probs = Vec::with_capacity(n);
    let mut o = 20usize;
    for si in 0..n {
        let mut per_sample = Vec::with_capacity(vl);
        for j in 0..vl {
            let mut row = Vec::with_capacity(ncls);
            let mut sum = 0.0f32;
            for _ in 0..ncls {
                let p = f32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
                if !p.is_finite() || p < 0.0 {
                    fail(format!(
                        "labels '{path}': sample {si} pos {j}: probability {p} is negative or non-finite"
                    ));
                }
                row.push(p);
                sum += p;
                o += 4;
            }
            if !(0.85..=1.15).contains(&sum) {
                fail(format!(
                    "labels '{path}': sample {si} pos {j}: probabilities sum to {sum:.4} (expected ~1.0) — teacher labels look corrupt"
                ));
            }
            per_sample.push(row);
        }
        probs.push(per_sample);
    }
    Labels { n, val_len: vl, ncls, probs }
}

fn cmd_niah_gen(args: &[String]) {
    let n = arg_u64(args, "--n", 32);
    let seq = arg_u64(args, "--seq", 128) as usize;
    let key_len = arg_u64(args, "--key-len", 1) as usize;
    let val_len = arg_u64(args, "--val-len", 1) as usize;
    let distract = arg_u64(args, "--distract", 0) as usize;
    let seed = arg_u64(args, "--seed", 4321);
    let out = arg_str(args, "--out", "niah_pool.bin").to_string();
    ensure_pos("--n", n, 1);
    ensure_pos("--seq", seq as u64, 1);
    if n > 1_000_000 {
        fail("--n must be <= 1000000".to_string());
    }
    if seed > u32::MAX as u64 {
        fail("--seed must fit in u32 for pool files (<= 4294967295)".to_string());
    }
    ensure_niah_seq("niah-gen", seq, key_len, val_len);
    let mut gen = Niah::new(NiahCfg {
        n_distract: distract,
        key_len,
        val_len,
        seed,
        ..NiahCfg::default()
    });
    let mut prompts = Vec::with_capacity(n as usize);
    let mut answers = Vec::with_capacity(n as usize);
    let mut starts = Vec::with_capacity(n as usize);
    let mut gen = gen; // quiet
    for i in 0..n {
        let s = gen.sample(seq);
        if s.answer.len() != val_len || s.answer_start + val_len > seq {
            fail(format!(
                "niah-gen: at --seq {seq} the answer does not fit the sequence (this task needs a longer --seq; try {})", 
                NiahCfg { key_len, val_len, ..NiahCfg::default() }.min_seq()
            ));
        }
        let _ = i;
        prompts.push(s.tokens);
        answers.push(s.answer);
        starts.push(s.answer_start as u32);
    }
    let mut blob = Vec::with_capacity(32 + prompts.len() * (seq + val_len + 4));
    blob.extend_from_slice(POOL_MAGIC);
    for v in [n as u32, seq as u32, key_len as u32, val_len as u32, distract as u32, seed as u32] {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    for p in &prompts {
        blob.extend_from_slice(p);
    }
    for a in &answers {
        blob.extend_from_slice(a);
    }
    for s in &starts {
        blob.extend_from_slice(&s.to_le_bytes());
    }
    create_parent(&out);
    std::fs::write(&out, &blob)
        .unwrap_or_else(|e| fail(format!("cannot write pool '{out}': {e}")));
    let w = [10usize, 7, 8, 8, 10, 12];
    println!("{}", ui::cyan("── NIAH prompt pool ─────────────────────────────────"));
    println!("{}", ui::row(&["n".into(), "seq".into(), "key".into(), "val".into(), "distract".into(), "seed".into()], &w));
    println!("{}", ui::rule_mid(&w));
    println!("{}", ui::row(&[n.to_string(), seq.to_string(), key_len.to_string(), val_len.to_string(), distract.to_string(), seed.to_string()], &w));
    println!("{}", ui::rule_bot(&w));
    let end = prompts[0].iter().rposition(|&b| b != b' ').map(|p| p + 1).unwrap_or(0).min(70);
    println!("  {} {}", ui::dim("sample[0]:"), String::from_utf8_lossy(&prompts[0][..end]));
    println!("  {} {}", ui::dim("answer[0]:"), String::from_utf8_lossy(&answers[0]));
    println!("  {} {} ({} B)", ui::dim("wrote"), ui::bold(&out), blob.len());
    println!("  {}", ui::dim("next: have the teacher label this pool (Section X of tessera_vs_gemma.py), then `tessera transplant`"));
}

fn cmd_transplant(args: &[String]) {
    let pool_p = arg_str(args, "--pool", "");
    let labels_p = arg_str(args, "--labels", "");
    if pool_p.is_empty() {
        fail("--pool is required (create one: tessera niah-gen --out niah_pool.bin)".to_string());
    }
    if labels_p.is_empty() {
        fail("--labels is required (teacher soft-labels over the retrieval alphabet; Section X of tessera_vs_gemma.py produces them)".to_string());
    }
    let steps = arg_u64(args, "--steps", 1500);
    let lr_peak = arg_f64(args, "--lr", 5e-4);
    let seed = arg_u64(args, "--seed", 7);
    let kd_w = arg_f64(args, "--kd-weight", 0.7);
    let ans_w = arg_f64(args, "--ans-w", 5.0);
    let eval_every = arg_u64(args, "--eval-every", 250);
    let batch = arg_u64(args, "--batch", 16) as usize;
    let out = arg_str(args, "--out", "checkpoints/transplanted.bin").to_string();
    ensure_pos("--steps", steps, 1);
    ensure_pos("--batch", batch as u64, 1);
    if !lr_peak.is_finite() || lr_peak <= 0.0 {
        fail("--lr must be a finite number > 0".to_string());
    }
    if !kd_w.is_finite() || !(0.0..=4.0).contains(&kd_w) {
        fail("--kd-weight must be a finite number in [0, 4]".to_string());
    }
    if ans_w <= 0.0 {
        fail("--ans-w must be > 0".to_string());
    }
    let pool = load_pool(pool_p);
    let labels = load_labels(labels_p, pool.n, pool.val_len);
    ensure_niah_seq("transplant", pool.seq, pool.key_len, pool.val_len);
    // teacher-vs-truth agreement on the first answer byte (uses the pool's
    // hidden answers; reported alongside the verdict)
    let mut teach_ok = 0usize;
    for si in 0..pool.n {
        let r = &labels.probs[si][0];
        let best = r
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let truth = tessera::data::key_alphabet()
            .iter()
            .position(|&b| b == pool.answers[si][0])
            .unwrap_or(usize::MAX);
        teach_ok += (best == truth) as usize;
    }
    let alpha: Vec<usize> = tessera::data::key_alphabet().iter().map(|&b| b as usize).collect();
    println!("{}", ui::cyan("── TESSERA transplant: teacher soft-labels → Rust Tessera ──"));
    println!(
        "  {} {} (n={}, seq={}, k{}v{} distract={})   {} {} (kd_weight={kd_w})",
        ui::dim("pool:"),
        pool_p,
        pool.n,
        pool.seq,
        pool.key_len,
        pool.val_len,
        pool.n_distract,
        ui::dim("labels:"),
        labels_p,
    );
    println!(
        "== TESSERA transplant trainer: seq={} steps={} batch={batch} lr={lr_peak} kd={kd_w} ans-w={ans_w} seed={seed} ({} SIMD) ==",
        pool.seq,
        steps,
        simd_level_name()
    );
    let cfg = ModelCfg::default();
    let mut model = Model::<f32>::new(cfg, seed);
    let nparams = model.count_params();
    println!("params: {nparams}  inference working state: {} B (substrate {} B)", model.state_bytes(), model.substrate_bytes());
    let params = model.params();
    let mut opt = AdamW::new(&params, 0.01);
    let mut pick = tessera::rng::Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15 ^ (pool.seed << 1));
    let mut gen_eval = Niah::new(NiahCfg {
        key_len: pool.key_len,
        val_len: pool.val_len,
        ans_w,
        seed: 999_123,
        ..NiahCfg::default()
    });
    let t0 = now_ms();
    let mut last_loss = 0.0f64;
    for it in 0..steps {
        let lr = wsd_lr(it, steps, lr_peak);
        model.zero_grad();
        // gradient accumulation over `batch` pool samples (the validated
        // recipe trains with an effective batch of 16; single-sample steps
        // are too noisy at short sequence lengths)
        let mut loss_acc = 0.0f64;
        for _ in 0..batch {
            let si = pick.below(pool.n);
            let (t1, t2, mask, st) =
                tessera::data::targets_from_tokens(&pool.prompts[si], pool.val_len, ans_w);
            let mut kd: Vec<Option<Vec<f64>>> = vec![None; pool.seq];
            for (j, row) in labels.probs[si].iter().enumerate() {
                let pos = st as isize + j as isize - 1;
                if pos >= 0 && (pos as usize) < pool.seq {
                    kd[pos as usize] = Some(row.iter().map(|&x| x as f64).collect());
                }
            }
            let toks: Vec<usize> = pool.prompts[si].iter().map(|&b| b as usize).collect();
            model.fwd_seq(&toks, true);
            let ls = model.bwd_seq_kd(&t1, &t2, &mask, &kd, kd_w, &alpha);
            if !ls.is_finite() {
                eprintln!("NON-FINITE LOSS at step {it} (sample {si})");
                std::process::exit(2);
            }
            loss_acc += ls;
        }
        let loss = loss_acc / batch as f64;
        last_loss = loss;
        {
            let mut first_bad: Option<(&str, usize)> = None;
            for p in params.iter() {
                let bad: usize = unsafe { (*p.g).iter().filter(|x| !x.to().is_finite()).count() };
                if bad > 0 && first_bad.is_none() {
                    first_bad = Some((p.name, bad));
                }
            }
            if let Some((nm, bad)) = first_bad {
                eprintln!("NON-FINITE GRADS after bwd at step {it}: first {nm} ({bad} elems), loss={loss}");
                std::process::exit(2);
            }
        }
        opt.clip(&params, 1.0);
        opt.step(&params, lr);
        if it % 5 == 0 || it == steps - 1 {
            let dt = (now_ms() - t0) as f64 / 1000.0;
            println!(
                "step {it:5}  seq {:5}  loss {loss:.4}  lr {lr:.2e}  tok/s {:.0}",
                pool.seq,
                (it + 1) as f64 * (batch * pool.seq) as f64 / dt.max(1e-9)
            );
        }
        if eval_every > 0 && (it + 1) % eval_every == 0 {
            let acc = tessera::data::niah_eval_model(&mut model, &mut gen_eval, pool.seq, 16);
            println!("  eval@{}: {}/16 correct", pool.seq, acc);
        }
    }
    create_parent(&out);
    save_model(&mut model, &out);
    println!("saved -> {out}");
    let acc = tessera::data::niah_eval_model(&mut model, &mut gen_eval, pool.seq, 32);
    println!("{}", ui::cyan("── transplant verdict ──────────────────────────────"));
    println!("  {} {kd_w}", ui::dim("teacher weight (kd):"));
    println!("  {} {teach_ok}/{}", ui::dim("teacher label acc  :"), pool.n);
    println!("  {} {last_loss:.4}", ui::dim("final pooled loss  :"));
    println!("  {} NIAH@{}: {acc}/32 correct", ui::dim("fresh-prompt recall:"), pool.seq);
    if acc == 32 {
        println!("  {}", ui::green("TRANSPLANT PASS"));
    } else {
        println!("  {}", ui::yellow("TRANSPLANT DONE (recall < 32/32 — more steps / better teacher labels may help)"));
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  ACCEL — native accelerator selftest: CUDA driver API via dlopen (zero
//  deps, runtime-JIT'd PTX kernels) + a TPU bridge (Rust <-> torch_xla
//  parity; the only honest way to reach a TPU without XLA).
//  Run: tessera accel     (T4: real CUDA compute; TPU VM: real XLA compute)
// ────────────────────────────────────────────────────────────────────────────

const CUDA_VADD_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64
.visible .entry vadd(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u32 pn) {
    .reg .pred %p<2>;
    .reg .f32  %f<5>;
    .reg .b32  %r<7>;
    .reg .b64  %rd<12>;
    ld.param.u64 %rd1, [pa];
    ld.param.u64 %rd2, [pb];
    ld.param.u64 %rd3, [pc];
    ld.param.u32 %r1, [pn];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd4, %rd1;
    mul.wide.u32 %rd5, %r5, 4;
    add.u64 %rd6, %rd4, %rd5;
    ld.global.f32 %f1, [%rd6];
    cvta.to.global.u64 %rd7, %rd2;
    add.u64 %rd8, %rd7, %rd5;
    ld.global.f32 %f2, [%rd8];
    add.f32 %f3, %f1, %f2;
    cvta.to.global.u64 %rd9, %rd3;
    add.u64 %rd10, %rd9, %rd5;
    st.global.f32 [%rd10], %f3;
DONE:
    ret;
}
"#;

const CUDA_MM_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64
.visible .entry mm(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u32 pm, .param .u32 pk, .param .u32 pn) {
    .reg .pred %p<3>;
    .reg .f32  %f<5>;
    .reg .b32  %r<18>;
    .reg .b64  %rd<22>;
    ld.param.u64 %rd1, [pa];
    ld.param.u64 %rd2, [pb];
    ld.param.u64 %rd3, [pc];
    ld.param.u32 %r1, [pm];
    ld.param.u32 %r2, [pk];
    ld.param.u32 %r3, [pn];
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mov.u32 %r7, %ctaid.y;
    mov.u32 %r8, %ntid.y;
    mov.u32 %r9, %tid.y;
    mad.lo.u32 %r10, %r4, %r5, %r6;
    mad.lo.u32 %r11, %r7, %r8, %r9;
    setp.ge.u32 %p1, %r10, %r3;
    @%p1 bra DONE;
    setp.ge.u32 %p1, %r11, %r1;
    @%p1 bra DONE;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r12, 0;
KLOOP:
    setp.ge.u32 %p1, %r12, %r2;
    @%p1 bra STORE;
    cvt.u64.u32 %rd4, %r11;
    cvt.u64.u32 %rd5, %r12;
    cvt.u64.u32 %rd6, %r10;
    mul.wide.u32 %rd8, %r11, %r2;
    add.u64 %rd8, %rd8, %rd5;
    mul.wide.u32 %rd9, %r12, %r3;
    add.u64 %rd9, %rd9, %rd6;
    cvta.to.global.u64 %rd10, %rd1;
    mad.lo.u64 %rd11, %rd8, 4, %rd10;
    ld.global.f32 %f2, [%rd11];
    cvta.to.global.u64 %rd12, %rd2;
    mad.lo.u64 %rd13, %rd9, 4, %rd12;
    ld.global.f32 %f3, [%rd13];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r12, %r12, 1;
    bra KLOOP;
STORE:
    cvta.to.global.u64 %rd14, %rd3;
    mul.wide.u32 %rd15, %r11, %r3;
    add.u64 %rd15, %rd15, %rd6;
    mad.lo.u64 %rd16, %rd15, 4, %rd14;
    st.global.f32 [%rd16], %f1;
DONE:
    ret;
}
"#;


mod accel_ffi {
    use std::ffi::{c_char, c_int, c_void};
    #[link(name = "dl")]
    extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
}

type CuFn = *mut std::ffi::c_void;

unsafe fn dlsym_or_zero(h: *mut std::ffi::c_void, name: &str) -> CuFn {
    let c = std::ffi::CString::new(name).unwrap();
    accel_ffi::dlsym(h, c.as_ptr())
}

struct CudaApi {
    cuInit: unsafe extern "C" fn(c_uint_t) -> c_int_t,
    cuDeviceGetCount: unsafe extern "C" fn(*mut c_int_t) -> c_int_t,
    cuDeviceGet: unsafe extern "C" fn(*mut c_int_t, c_int_t) -> c_int_t,
    cuDeviceGetName: unsafe extern "C" fn(*mut c_char_t, c_int_t, c_int_t) -> c_int_t,
    cuDeviceTotalMem: unsafe extern "C" fn(*mut usize, c_int_t) -> c_int_t,
    cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int_t, c_int_t, c_int_t) -> c_int_t,
    cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CuFn, c_int_t) -> c_int_t,
    cuCtxSetCurrent: unsafe extern "C" fn(CuFn) -> c_int_t,
    cuMemAlloc: unsafe extern "C" fn(*mut CuFn, usize) -> c_int_t,
    cuMemFree: unsafe extern "C" fn(CuFn) -> c_int_t,
    cuMemcpyHtoD: unsafe extern "C" fn(CuFn, *const c_void_t, usize) -> c_int_t,
    cuMemcpyDtoH: unsafe extern "C" fn(*mut c_void_t, CuFn, usize) -> c_int_t,
    cuModuleLoadDataEx: unsafe extern "C" fn(*mut CuFn, *const c_void_t, c_uint_t, *mut c_uint_t, *mut *mut c_void_t) -> c_int_t,
    cuModuleGetFunction: unsafe extern "C" fn(*mut CuFn, CuFn, *const c_char_t) -> c_int_t,
    cuLaunchKernel: unsafe extern "C" fn(CuFn, c_uint_t, c_uint_t, c_uint_t, c_uint_t, c_uint_t, c_uint_t, c_uint_t, CuFn, *mut *mut c_void_t, *mut *mut c_void_t) -> c_int_t,
    cuCtxSynchronize: unsafe extern "C" fn() -> c_int_t,
    cuGetErrorString: unsafe extern "C" fn(c_int_t, *mut *const c_char_t) -> c_int_t,
}

#[allow(non_camel_case_types)]
type c_int_t = std::ffi::c_int;
#[allow(non_camel_case_types)]
type c_uint_t = std::ffi::c_uint;
#[allow(non_camel_case_types)]
type c_char_t = std::ffi::c_char;
#[allow(non_camel_case_types)]
type c_void_t = std::ffi::c_void;

fn load_cuda() -> Result<CudaApi, String> {
    use std::ffi::CString;
    let mut h = std::ptr::null_mut();
    for lib in ["libcuda.so.2", "libcuda.so.1", "libcuda.so"] {
        let c = CString::new(lib).unwrap();
        h = unsafe { accel_ffi::dlopen(c.as_ptr(), 2) }; // RTLD_NOW
        if !h.is_null() {
            break;
        }
    }
    if h.is_null() {
        return Err("libcuda not present — no NVIDIA driver on this box".to_string());
    }
    unsafe {
        macro_rules! need {
            ($name:literal) => {
                match dlsym_or_zero(h, $name) {
                    p if !p.is_null() => std::mem::transmute::<CuFn, _>(p),
                    _ => return Err(format!("driver symbol {} missing", $name)),
                }
            };
        }
        Ok(CudaApi {
            cuInit: need!("cuInit"),
            cuDeviceGetCount: need!("cuDeviceGetCount"),
            cuDeviceGet: need!("cuDeviceGet"),
            cuDeviceGetName: need!("cuDeviceGetName"),
            cuDeviceTotalMem: need!("cuDeviceTotalMem_v2"),
            cuDeviceGetAttribute: need!("cuDeviceGetAttribute"),
            cuDevicePrimaryCtxRetain: need!("cuDevicePrimaryCtxRetain"),
            cuCtxSetCurrent: need!("cuCtxSetCurrent"),
            cuMemAlloc: need!("cuMemAlloc_v2"),
            cuMemFree: need!("cuMemFree_v2"),
            cuMemcpyHtoD: need!("cuMemcpyHtoD_v2"),
            cuMemcpyDtoH: need!("cuMemcpyDtoH_v2"),
            cuModuleLoadDataEx: need!("cuModuleLoadDataEx"),
            cuModuleGetFunction: need!("cuModuleGetFunction"),
            cuLaunchKernel: need!("cuLaunchKernel"),
            cuCtxSynchronize: need!("cuCtxSynchronize"),
            cuGetErrorString: need!("cuGetErrorString"),
        })
    }
}

fn cu_err(d: &CudaApi, rc: c_int_t, what: &str) -> String {
    unsafe {
        let mut p: *const c_char_t = std::ptr::null();
        let msg = if (d.cuGetErrorString)(rc, &mut p) == 0 && !p.is_null() {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        } else {
            format!("rc={rc}")
        };
        format!("CUDA {what}: {msg}")
    }
}

/// JIT a PTX string, capturing the driver's error log on failure.
unsafe fn jit_module(d: &CudaApi, ptx: &str) -> Result<CuFn, String> {
    let mut logbuf = [0u8; 4096];
    let mut loglen = 4096usize;
    let mut opts = [6u32, 7u32]; // CU_JIT_ERROR_LOG_BUFFER, _SIZE_BYTES
    let mut vals = [
        logbuf.as_mut_ptr() as *mut c_void_t,
        &mut loglen as *mut usize as *mut c_void_t,
    ];
    let mut module: CuFn = std::ptr::null_mut();
    let rc = (d.cuModuleLoadDataEx)(
        &mut module,
        ptx.as_ptr() as *const c_void_t,
        2,
        opts.as_mut_ptr(),
        vals.as_mut_ptr(),
    );
    if rc != 0 {
        let log = String::from_utf8_lossy(&logbuf[..loglen.min(4096)]).into_owned();
        return Err(format!("{} (JIT log: {})", cu_err(d, rc, "cuModuleLoadDataEx"), log.trim()));
    }
    Ok(module)
}

/// Deterministic accel-test inputs from tessera's own RNG (LCG).
fn accel_inputs(n_vadd: usize, m: usize, k: usize, n_mm: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f64>, Vec<f64>) {
    let mut rng = tessera::rng::Rng::new(0xACCE1);
    let (mut a, mut b) = (Vec::with_capacity(n_vadd), Vec::with_capacity(n_vadd));
    for _ in 0..n_vadd {
        a.push((rng.f64() * 2.0 - 1.0) as f32);
        b.push((rng.f64() * 2.0 - 1.0) as f32);
    }
    let (mut am, mut bm) = (Vec::with_capacity(m * k), Vec::with_capacity(k * n_mm));
    for _ in 0..m * k {
        am.push((rng.f64() * 2.0 - 1.0) as f32);
    }
    for _ in 0..k * n_mm {
        bm.push((rng.f64() * 2.0 - 1.0) as f32);
    }
    let exp_v = a.iter().zip(&b).map(|(x, y)| (*x + *y) as f64).collect();
    let mut exp_m = vec![0f64; m * n_mm];
    for i in 0..m {
        for j in 0..n_mm {
            let mut s = 0f64;
            for t in 0..k {
                s += am[i * k + t] as f64 * bm[t * n_mm + j] as f64;
            }
            exp_m[i * n_mm + j] = s;
        }
    }
    (a, b, am, bm, exp_v, exp_m)
}

fn max_abs_diff(xs: &[f32], want: &[f64]) -> f64 {
    xs.iter().zip(want).map(|(x, w)| (*x as f64 - w).abs()).fold(0.0f64, f64::max)
}

fn cuda_selftest(rows: &mut Vec<(String, String, bool)>, fails: &mut usize) {
    let d = match load_cuda() {
        Ok(d) => d,
        Err(e) => {
            rows.push(("CUDA".into(), format!("[skip] {e}"), true));
            return;
        }
    };
    unsafe {
        let rc = (d.cuInit)(0);
        if rc != 0 {
            rows.push(("CUDA".into(), format!("[FAIL] {}", cu_err(&d, rc, "cuInit")), false));
            *fails += 1;
            return;
        }
        let mut ndev: c_int_t = 0;
        if (d.cuDeviceGetCount)(&mut ndev) != 0 || ndev < 1 {
            rows.push(("CUDA".into(), "[skip] no CUDA device visible".into(), true));
            return;
        }
        let mut dev: c_int_t = 0;
        (d.cuDeviceGet)(&mut dev, 0);
        let mut namebuf = [0 as c_char_t; 256];
        (d.cuDeviceGetName)(namebuf.as_mut_ptr(), 256, dev);
        let cname = std::ffi::CStr::from_ptr(namebuf.as_ptr()).to_string_lossy().into_owned();
        let mut bytes = 0usize;
        (d.cuDeviceTotalMem)(&mut bytes, dev);
        let (mut maj, mut min, mut sms) = (0 as c_int_t, 0 as c_int_t, 0 as c_int_t);
        (d.cuDeviceGetAttribute)(&mut maj, 75, dev); // COMPUTE_CAPABILITY_MAJOR
        (d.cuDeviceGetAttribute)(&mut min, 76, dev);
        (d.cuDeviceGetAttribute)(&mut sms, 16, dev); // MULTIPROCESSOR
        let mut ctx: CuFn = std::ptr::null_mut();
        if (d.cuDevicePrimaryCtxRetain)(&mut ctx, dev) != 0 || (d.cuCtxSetCurrent)(ctx) != 0 {
            rows.push(("CUDA".into(), "[FAIL] cannot retain primary context".into(), false));
            *fails += 1;
            return;
        }
        rows.push((
            "CUDA".into(),
            format!("[ok  ] {cname}, sm_{maj}{min}, {sms} SMs, {:.1} GB", bytes as f64 / (1 << 30) as f64),
            true,
        ));

        // memory roundtrip: 1 MiB pattern
        let n_pat = 1usize << 18;
        let mut pat: Vec<f32> = (0..n_pat).map(|i| (i % 251) as f32 * 0.5).collect();
        let mut dptr: CuFn = std::ptr::null_mut();
        let mut ok = (d.cuMemAlloc)(&mut dptr, n_pat * 4) == 0
            && (d.cuMemcpyHtoD)(dptr, pat.as_ptr() as *const c_void_t, n_pat * 4) == 0;
        let mut back = vec![0f32; n_pat];
        ok = ok && (d.cuMemcpyDtoH)(back.as_mut_ptr() as *mut c_void_t, dptr, n_pat * 4) == 0;
        if ok {
            let identical = back.iter().zip(&pat).all(|(x, y)| *x == *y);
            rows.push(("CUDA".into(), format!("[{} ] H2D/D2H roundtrip 1 MiB, bit-exact", if identical { "ok " } else { "FAIL" }), identical));
            if !identical {
                *fails += 1;
            }
        } else {
            rows.push(("CUDA".into(), "[FAIL] H2D/D2H memcpy".into(), false));
            *fails += 1;
        }
        if !dptr.is_null() {
            (d.cuMemFree)(dptr);
        }

        // real compute: vadd + matmul via runtime-JIT'd PTX
        let nv = 8192usize;
        let (m, k, nmm) = (64usize, 64usize, 64usize);
        let (a, b, am, bm, exp_v, exp_m) = accel_inputs(nv, m, k, nmm);
        let mut da: CuFn = std::ptr::null_mut();
        let mut db: CuFn = std::ptr::null_mut();
        let mut dc: CuFn = std::ptr::null_mut();
        let mem_ok = (d.cuMemAlloc)(&mut da, nv * 4) == 0
            && (d.cuMemAlloc)(&mut db, nv * 4) == 0
            && (d.cuMemAlloc)(&mut dc, nv * 4) == 0
            && (d.cuMemcpyHtoD)(da, a.as_ptr() as *const c_void_t, nv * 4) == 0
            && (d.cuMemcpyHtoD)(db, b.as_ptr() as *const c_void_t, nv * 4) == 0;
        if !mem_ok {
            rows.push(("CUDA".into(), "[FAIL] device alloc/upload for kernels".into(), false));
            *fails += 1;
            return;
        }
        match jit_module(&d, CUDA_VADD_PTX) {
            Ok(modu) => {
                let mut fnv: CuFn = std::ptr::null_mut();
                let cname = std::ffi::CString::new("vadd").unwrap();
                let mut n_arg: c_uint_t = nv as c_uint_t;
                let mut pargs: [*mut c_void_t; 4] = [
                    &mut da as *mut _ as *mut c_void_t,
                    &mut db as *mut _ as *mut c_void_t,
                    &mut dc as *mut _ as *mut c_void_t,
                    &mut n_arg as *mut _ as *mut c_void_t,
                ];
                let get_ok = (d.cuModuleGetFunction)(&mut fnv, modu, cname.as_ptr()) == 0;
                let launch_ok = get_ok
                    && (d.cuLaunchKernel)(fnv, (nv / 256) as c_uint_t, 1, 1, 256, 1, 1, 0, std::ptr::null_mut(), pargs.as_mut_ptr(), std::ptr::null_mut()) == 0
                    && (d.cuCtxSynchronize)() == 0;
                let mut out = vec![0f32; nv];
                let copy_ok = launch_ok && (d.cuMemcpyDtoH)(out.as_mut_ptr() as *mut c_void_t, dc, nv * 4) == 0;
                if copy_ok {
                    let err = max_abs_diff(&out, &exp_v);
                    let good = err == 0.0;
                    rows.push(("CUDA".into(), format!("[{} ] PTX JIT vadd 8192 elems, max err {err}", if good { "ok " } else { "FAIL" }), good));
                    if !good {
                        *fails += 1;
                    }
                } else {
                    rows.push(("CUDA".into(), "[FAIL] vadd launch/copy".into(), false));
                    *fails += 1;
                }
            }
            Err(e) => {
                rows.push(("CUDA".into(), format!("[FAIL] {e}"), false));
                *fails += 1;
            }
        }
        let mut dma: CuFn = std::ptr::null_mut();
        let mut dmb: CuFn = std::ptr::null_mut();
        let mut dmc: CuFn = std::ptr::null_mut();
        let mm_ok = (d.cuMemAlloc)(&mut dma, m * k * 4) == 0
            && (d.cuMemAlloc)(&mut dmb, k * nmm * 4) == 0
            && (d.cuMemAlloc)(&mut dmc, m * nmm * 4) == 0
            && (d.cuMemcpyHtoD)(dma, am.as_ptr() as *const c_void_t, m * k * 4) == 0
            && (d.cuMemcpyHtoD)(dmb, bm.as_ptr() as *const c_void_t, k * nmm * 4) == 0;
        if !mm_ok {
            rows.push(("CUDA".into(), "[FAIL] matmul alloc/upload".into(), false));
            *fails += 1;
            return;
        }
        match jit_module(&d, CUDA_MM_PTX) {
            Ok(modu) => {
                let mut fnm: CuFn = std::ptr::null_mut();
                let cname = std::ffi::CString::new("mm").unwrap();
                let (mut um, mut uk, mut un) = (m as c_uint_t, k as c_uint_t, nmm as c_uint_t);
                let mut pargs: [*mut c_void_t; 6] = [
                    &mut dma as *mut _ as *mut c_void_t,
                    &mut dmb as *mut _ as *mut c_void_t,
                    &mut dmc as *mut _ as *mut c_void_t,
                    &mut um as *mut _ as *mut c_void_t,
                    &mut uk as *mut _ as *mut c_void_t,
                    &mut un as *mut _ as *mut c_void_t,
                ];
                let get_ok = (d.cuModuleGetFunction)(&mut fnm, modu, cname.as_ptr()) == 0;
                let launch_ok = get_ok
                    && (d.cuLaunchKernel)(fnm, (nmm / 16) as c_uint_t, (m / 16) as c_uint_t, 1, 16, 16, 1, 0, std::ptr::null_mut(), pargs.as_mut_ptr(), std::ptr::null_mut()) == 0
                    && (d.cuCtxSynchronize)() == 0;
                let mut out = vec![0f32; m * nmm];
                let copy_ok = launch_ok && (d.cuMemcpyDtoH)(out.as_mut_ptr() as *mut c_void_t, dmc, m * nmm * 4) == 0;
                if copy_ok {
                    let err = max_abs_diff(&out, &exp_m);
                    let good = err < 5e-4;
                    rows.push(("CUDA".into(), format!("[{} ] PTX JIT matmul 64x64x64 vs Rust f64, max err {err:.2e}", if good { "ok " } else { "FAIL" }), good));
                    if !good {
                        *fails += 1;
                    }
                } else {
                    rows.push(("CUDA".into(), "[FAIL] matmul launch/copy".into(), false));
                    *fails += 1;
                }
            }
            Err(e) => {
                rows.push(("CUDA".into(), format!("[FAIL] {e}"), false));
                *fails += 1;
            }
        }
    }
}

fn tpu_selftest(rows: &mut Vec<(String, String, bool)>, fails: &mut usize) {
    // Native PJRT handshake — the ABI JAX/TF/PyTorch-XLA use, driven straight
    // from Rust via dlopen. Zero Python, zero PyTorch, zero crates.
    match tessera::pjrt::probe() {
        Ok(info) => {
            rows.push((
                "TPU ".into(),
                format!(
                    "[ok  ] {} via {} (PJRT API v{}.{})",
                    info.platform, info.lib, info.api_version.0, info.api_version.1
                ),
                true,
            ));
            if !info.platform_version.is_empty() {
                rows.push((
                    "TPU ".into(),
                    format!("[ok  ] runtime: {}", info.platform_version),
                    true,
                ));
            }
            rows.push((
                "TPU ".into(),
                format!(
                    "[ok  ] {} addressable device(s), process index {}",
                    info.devices, info.process_index
                ),
                true,
            ));
            rows.push((
                "TPU ".into(),
                "[skip] HLO buffer/execute: phase-2 milestone (handshake verified natively)".to_string(),
                true,
            ));
        }
        Err(tessera::pjrt::Probe::Skip(r)) => {
            rows.push(("TPU ".into(), format!("[skip] {r}"), true));
        }
        Err(tessera::pjrt::Probe::Fail(r)) => {
            rows.push(("TPU ".into(), format!("[FAIL] {r}"), false));
            *fails += 1;
        }
    }
}

fn cmd_accel() {
    let w = [8usize, 82];
    println!("{}", ui::cyan("── TESSERA accelerator selftest (native CUDA · native PJRT TPU) ──"));
    let mut rows: Vec<(String, String, bool)> = Vec::new();
    let mut fails = 0usize;
    cuda_selftest(&mut rows, &mut fails);
    tpu_selftest(&mut rows, &mut fails);
    println!("{}", ui::rule_top(&w));
    for (dev, line, _hard) in &rows {
        let painted = if line.starts_with("[ok") {
            line.replacen("[ok", &ui::green("[ok"), 1)
        } else if line.starts_with("[FAIL") {
            line.replacen("[FAIL", &ui::red("[FAIL"), 1)
        } else if line.starts_with("[skip") {
            line.replacen("[skip", &ui::yellow("[skip"), 1)
        } else {
            line.clone()
        };
        println!("{}", ui::row(&[dev.clone(), painted], &w));
    }
    println!("{}", ui::rule_bot(&w));
    if fails == 0 {
        println!("{}", ui::green("ACCEL PASS"));
    } else {
        println!("{} ({fails} hardware check(s) failed)", ui::red("ACCEL FAIL"));
        std::process::exit(1);
    }
}

fn cmd_logo() {
    // truecolor 3x3 mosaic — nine pillars around the amber memory
    let tiles = [
        ["34;211;238", "56;189;248", "34;211;238"],
        ["129;140;248", "245;158;11", "129;140;248"],
        ["34;211;238", "56;189;248", "34;211;238"],
    ];
    let on = std::env::var("CLICOLOR_FORCE").map(|v| v != "0").unwrap_or(false);
    for r in 0..3 {
        let mut line = String::from("    ");
        for (c, t) in tiles[r].iter().enumerate() {
            if c > 0 {
                line.push_str(&ui::dim(" │ "));
            }
            let block = "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}";
            if on {
                line.push_str(&format!("\x1b[38;2;{t}m{block}\x1b[0m"));
            } else {
                line.push_str(block);
            }
        }
        println!("{line}");
    }
    println!();
    println!("    {}", ui::bold("TESSERA-Q"));
    println!("    {}", ui::dim("O(1) memory · differential attention · nine pillars"));
}

fn cmd_pack(args: &[String]) {
    let path = arg_str(args, "--model", "");
    if path.is_empty() {
        fail("--model is required (the .bin checkpoint to convert)".to_string());
    }
    ensure_model(path);
    let default_out = format!(
        "{}.tem",
        std::path::Path::new(path).with_extension("").to_string_lossy()
    );
    let out = arg_str(args, "--out", &default_out).to_string();
    // reject .tem inputs early with a friendly message
    let mut mg = [0u8; 8];
    {
        use std::io::Read;
        let mut f = std::fs::File::open(path).unwrap_or_else(|e| fail(format!("cannot open '{path}': {e}")));
        f.read_exact(&mut mg).unwrap_or_else(|e| fail(format!("'{path}' too small for a header: {e}")));
    }
    if mg == *tessera::tem::MAGIC {
        fail(format!("'{path}' is already a .tem file — nothing to pack"));
    }
    if &mg[0..4] != *b"TSS1" {
        fail(format!("'{path}' is not a tessera checkpoint (unknown header)"));
    }
    let mut model = open_model(path);
    let n_tensors = model.params().len();
    let n_params = model.count_params();
    let raw = tessera::tem::raw_f32_bytes(&mut model) as u64;
    let t0 = std::time::Instant::now();
    let blob = tessera::tem::encode(&mut model);
    let pack_ms = t0.elapsed().as_millis();
    create_parent(&out);
    std::fs::write(&out, &blob).unwrap_or_else(|e| fail(format!("cannot write '{out}': {e}")));
    let t1 = std::time::Instant::now();
    let reloaded = tessera::tem::load(&out).unwrap_or_else(|e| fail(format!("verification load failed: {e}")));
    let load_us = t1.elapsed().as_micros();
    drop(reloaded);
    let file = blob.len() as u64;
    let ratio = raw as f64 / file as f64;
    let w = [26usize, 24];
    println!("{}", ui::cyan("── TESSERA .tem pack ──────────────────────────────"));
    println!("{}", ui::row(&["params".into(), format!("{n_params}")], &w));
    println!("{}", ui::row(&["tensors".into(), format!("{n_tensors}")], &w));
    println!("{}", ui::row(&["legacy f32".into(), fmt_b(raw)], &w));
    println!("{}", ui::row(&[".tem file".into(), format!("{} ({ratio:.2}x smaller)", fmt_b(file))], &w));
    println!("{}", ui::rule_mid(&w));
    println!("{}", ui::row(&["codec".into(), "i8 groups + FNV-1a64".into()], &w));
    println!("{}", ui::row(&["pack time".into(), format!("{pack_ms} ms")], &w));
    println!("{}", ui::row(&["verify load".into(), format!("{load_us} us")], &w));
    println!("{}", ui::rule_bot(&w));
    println!("  {} {}", ui::dim("wrote"), ui::bold(&out));
    println!("  {}", ui::dim("tip: tessera niah-eval --model this.tem just works"));
}

fn cmd_unpack(args: &[String]) {
    let path = arg_str(args, "--model", "");
    if path.is_empty() {
        fail("--model is required (the .tem file to convert)".to_string());
    }
    ensure_model(path);
    let default_out = format!(
        "{}.bin",
        std::path::Path::new(path).with_extension("").to_string_lossy()
    );
    let out = arg_str(args, "--out", &default_out).to_string();
    let mut mg = [0u8; 8];
    {
        use std::io::Read;
        let mut f = std::fs::File::open(path).unwrap_or_else(|e| fail(format!("cannot open '{path}': {e}")));
        f.read_exact(&mut mg).unwrap_or_else(|e| fail(format!("'{path}' too small for a header: {e}")));
    }
    if mg != *tessera::tem::MAGIC {
        fail(format!("'{path}' is not a .tem file (run: tessera pack --model {path})"));
    }
    let mut model = tessera::tem::load(path).unwrap_or_else(|e| fail(format!("cannot load '{path}': {e}")));
    create_parent(&out);
    model.save(&out).unwrap_or_else(|e| fail(format!("cannot write '{out}': {e}")));
    let n = model.count_params();
    println!("  {} {} ({} params, f32 TSS1) — note: dequantization is lossy by design", ui::dim("wrote"), ui::bold(&out), n);
}

fn cmd_info() {
    let cfg = ModelCfg::default();
    let w = [22usize, 34];
    println!("{}", ui::cyan("── TESSERA build & model ───────────────────────────"));
    println!("{}", ui::row(&["version".into(), version_line()], &w));
    let mut model = Model::<f32>::new(cfg, 1);
    let params = model.count_params();
    println!("{}", ui::row(&["params".into(), format!("{params}")], &w));
    println!("{}", ui::row(&["working state".into(), format!("{} B (O(1) at any context)", model.state_bytes())], &w));
    println!("{}", ui::row(&["substrate".into(), format!("{} B", model.substrate_bytes())], &w));
    println!("{}", ui::rule_mid(&w));
    println!("{}", ui::row(&["d / layers / heads".into(), format!("{} / {} / {}", 128, 2, 2)], &w));
    println!("{}", ui::row(&["hopfield memory m".into(), format!("{} (fine {} + coarse {})", 64, 128, 16)], &w));
    println!("{}", ui::row(&["ff / lora_r / conv_k".into(), format!("{} / {} / {}", 768, 8, 4)], &w));
    println!("{}", ui::row(&["vocab".into(), "256 (byte-level)".into()], &w));
    println!("{}", ui::row(&["retrieval alphabet".into(), format!("{} chars", tessera::data::key_alphabet().len())], &w));
    println!("{}", ui::row(&["softcap / rope_base".into(), "30 / 10000".into()], &w));
    println!("{}", ui::rule_bot(&w));
    println!("  {}", ui::dim("try: tessera doctor  |  tessera help <command>"));
}

fn cmd_doctor() {
    let mut results: Vec<(String, bool, bool)> = Vec::new(); // name, ok, hard
    {
        let mut m = Model::<f32>::new(ModelCfg::default(), 3);
        let (z, _) = m.fwd_step(65, false);
        let finite = z.iter().all(|x| x.to().is_finite());
        results.push(("model forward: 256 finite logits after 1 token".to_string(), finite, true));
    }
    {
        let mut a = tessera::rng::Rng::new(7);
        let mut b = tessera::rng::Rng::new(7);
        results.push(("rng stream deterministic".to_string(), a.u64() == b.u64(), true));
    }
    {
        let ok = std::fs::create_dir_all("checkpoints").is_ok()
            && std::fs::write("checkpoints/.doctor_probe", b"ok").is_ok()
            && std::fs::remove_file("checkpoints/.doctor_probe").is_ok();
        results.push(("checkpoints/ directory writable".to_string(), ok, true));
    }
    {
        let cands = [
            std::env::var("TESSERA_DATA").unwrap_or_default(),
            "data/enwik8".to_string(),
            "../data/enwik8".to_string(),
            "/home/user/data/enwik8".to_string(),
        ];
        let found = cands
            .iter()
            .find(|p| !p.is_empty() && std::path::Path::new(p).is_file())
            .cloned();
        match found {
            Some(p) => results.push((format!("corpus found: {p}"), true, false)),
            None => results.push((
                "corpus enwik8 not found (only needed by train/parity)".to_string(),
                false,
                false,
            )),
        }
    }
    {
        let rejects = checked_validate(&["--oops".to_string()], "memtest").is_err();
        results.push(("flag validator rejects unknown options".to_string(), rejects, true));
    }
    for (name, ok, hard) in &results {
        let mark = if *ok {
            ui::green("[ ok ]")
        } else if *hard {
            ui::red("[FAIL]")
        } else {
            ui::yellow("[warn]")
        };
        println!("  {mark} {name}");
    }
    let total = results.len();
    let passed = results.iter().filter(|(_, ok, _)| *ok).count();
    let hard_fails = results.iter().filter(|(_, ok, hard)| !ok && *hard).count();
    println!("doctor: {passed}/{total} checks passed");
    if hard_fails > 0 {
        std::process::exit(1);
    }
}


fn usage_text() -> &'static str {
    "TESSERA-Q — O(1)-memory sequence model\n\
         usage:\n  tessera help [command]           overview, or detailed help (try: help transplant)\n  \
         tessera version                   version, SIMD path, profile\n  \
         tessera info                      build + model summary table\n  \
         tessera doctor                    environment checks\n  \
         tessera gradcheck                 finite-difference verification (all 9 pillars)\n  \
         tessera memtest [--steps N]       O(1) memory invariance test\n  \
         tessera selftest                  quick end-to-end learning sanity check\n  \
         tessera niah-gen [--n N] [--seq T] [--key-len K] [--val-len K] [--out PATH]\n                                     export a NIAH prompt pool (transplant step 1)\n  \
         tessera niah-train [...]          train the store+recall circuit (tessera help niah-train)\n  \
         tessera niah-eval [--model PATH] [--seq T] [--n N]   fresh-prompt recall\n  \
         tessera pack --model CKPT.bin [--out M.tem]     ~4x smaller 8-bit container\\n  \\\n         tessera unpack --model M.tem [--out CKPT.bin]   back to legacy f32\\n  \\\n         tessera accel                     GPU/TPU selftest (native CUDA + native PJRT)\n  \
         tessera logo                      print the mosaic mark\\n  \\\n         tessera transplant --pool P --labels L [--kd-weight F] [--steps N]\n                                     distill teacher soft-labels into Tessera (step 3)\n  \
         tessera train --data PATH [--arch tessera|transformer] [...]   corpus LM training\n  \
         tessera parity --data PATH        parameter-matched speedup/quality comparison\n  \
         tessera generate --model PATH [--prompt TEXT] [--n N]\n\n\
         every command also accepts --help for its flags and defaults.\n\
         exit codes: 0 success | 2 usage or validation error | 1 runtime failure\n\
         unknown flags, missing values, and non-numeric values are rejected before any work starts."
}


fn usage() {
    println!("{}", usage_text());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 {
        if let Some(c0) = args.get(1) {
        if matches!(c0.as_str(), "--version" | "-V" | "version") {
            println!("{}", version_line());
            return;
        }
        }
    }
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    if cmd == "help" {
        match args.get(2) {
            None => usage(),
            Some(c) if c == "--help" || c == "-h" => usage(),
            Some(c) => match cmd_help_text(c) {
                Some(t) => println!("{t}"),
                None => {
                    eprintln!("{}: unknown command '{c}'", ui::red("error"));
                    if let Some(s) = suggest(c) {
                        eprintln!("\n  did you mean {}?", ui::cyan(&format!("tessera {s}")));
                    }
                    eprintln!("\n{}", usage_text());
                    std::process::exit(2);
                }
            },
        }
        return;
    }
    if cmd == "--help" || cmd == "-h" {
        usage();
        return;
    }
    let rest: Vec<String> = args[2..].to_vec();
    if rest.iter().any(|a| a == "--help" || a == "-h") {
        if let Some(t) = cmd_help_text(cmd) {
            println!("{t}");
            return;
        }
    }
    validate_args(&rest, cmd);
    match cmd {
        "gradcheck" => {
            println!("SIMD: {} (gradchecks run the f64 reference path)", simd_level_name());
            let rep = gradcheck::run_all();
            let mut worst = 0.0f64;
            for (name, err, th) in &rep.rows {
                worst = worst.max(*err);
                if *err >= *th {
                    println!("  FAIL {:55} rel_err {:.3e}", name, err);
                }
            }
            println!(
                "gradient checks: {} passed / {} total  (worst rel err {:.2e}, threshold {:.0e})",
                rep.rows.len() - rep.failed,
                rep.rows.len(),
                worst,
                1e-4
            );
            if rep.failed > 0 {
                std::process::exit(1);
            }
            println!("{}", ui::green("GRADCHECK PASS"));
        }
        "memtest" => memtest(&rest),
        "selftest" => selftest(),
        "niah-train" => train_niah(&rest),
        "niah-overfit" => niah_overfit(),
        "niah-eval" => cmd_niah_eval(&rest),
        "niah-gen" => cmd_niah_gen(&rest),
        "transplant" => cmd_transplant(&rest),
        "info" => cmd_info(),
        "doctor" => cmd_doctor(),
        "pack" => cmd_pack(&rest),
        "unpack" => cmd_unpack(&rest),
        "logo" => cmd_logo(),
        "accel" => cmd_accel(),
        "train" => train_lm(&rest, &arg_str(&rest, "--arch", "tessera")),
        "parity" => parity(&rest),
        "generate" => generate(&rest),
        other => {
            eprintln!("{}: unknown command '{other}'", ui::red("error"));
            if let Some(s) = suggest(other) {
                eprintln!("\n  did you mean {}?", ui::cyan(&format!("tessera {s}")));
            }
            eprintln!("\n{}", usage_text());
            std::process::exit(2);
        }
    }
}
