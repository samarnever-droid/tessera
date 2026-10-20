# TESSERA-Q — session status

Last updated: 2026-09-18 (hardening pass)

## Acceptance criteria

| # | criterion | status |
|---|---|---|
| 1 | FD gradchecks < 1e-4, all 9 pillars | ✅ 691/691 PASS, worst rel err 4.41e-6 |
| 2 | O(1) memory ≈145 KB | ✅ 153,960 B @64 == @100k (substrate 147,456 B = 144 KiB) |
| 3 | NIAH 16/16 @ 8192 (k4/v4, seed 999123) | ⏳ pending — run `bash scripts/ladder_v6.sh` (~3 h, gated, resumable) |
| 4 | enwik8 parity + >2× speedup | ✅ BPC within 0.15; throughput@8192 3.23× (621 vs 192 tok/s) |

## Hardening pass (this session)

- **Zero build warnings** (was 40): dead code removed/annotated, unused muts, dP→dpv,
  strict arg parsing, `Corpus::from_file` empty-file guard.
- **69 new tests, all passing**: U01–U25 unit (data/rng/num/norms/optim/model),
  U26–U29 parser cores, C01–C40 CLI integration (`tests/cli.rs`, spawns the real binary).
- **CLI contract**: exit 0/2/1; unknown flags/commands, bad values, missing files,
  too-small `--seq` (NiahCfg::min_seq), `--n/--steps` 0, negative `--lr`,
  `--resume` without checkpoint — all clean `error:` + exit 2, never a panic.
- **ACCEL selftest (`tessera accel`)** — native accelerator support:
  * CUDA: the CUDA **Driver API** loaded at runtime via dlopen (zero deps) —
    cuInit/device probe (name, sm_xx, SMs, GB), bit-exact 1 MiB H2D/D2H
    roundtrip, then two **embedded PTX kernels** (vadd 8192 + 64x64x64 matmul,
    JIT-compiled at runtime with error-log capture) verified against
    Rust-f64 references (vadd must be exact; matmul < 5e-4).
  * TPU: **native PJRT C API handshake** (`src/pjrt.rs`) — no Python, no
    PyTorch. dlopens the TPU runtime plugin (libtpu.so /
    pjrt_c_api_tpu_plugin.so; TESSERA_TPU_LIB overrides; probes /lib,
    /usr/lib, python dist-packages dirs as plain files), resolves
    TPU_PJRT_C_API / GetPjrtApi, then Plugin_Initialize -> Client_Create ->
    platform name/version -> process index -> addressable-device count.
    Struct layouts transcribed from pjrt_c_api.h (minor 115) and every call
    is null/error-guarded. HLO buffer/execute = phase-2.
  * Behavior: absent hardware is [skip] (never fails), present-but-broken
    hardware is [FAIL] + exit 1. Test C60; 60 CLI tests total.
- **`.tem` container (TESSERA Embedded Model, v1)**: `src/tem.rs`, magic
  `TEMST8V1`. 8-bit symmetric quantization in 128-element groups (one f32
  scale per group), self-describing header (full ModelCfg incl. float fields
  TSS1 dropped + named tensor directory), FNV-1a64 over the payload, one
  read + one linear dequant pass on load (no entropy decoder). ~3.9x smaller
  than the f32 dump; pack ~5 ms, verify-load ~17 ms for 760K params.
  Commands: `pack --model m.bin [--out m.tem]`, `unpack` (back to TSS1),
  `logo`; every `--model` flag (niah-eval/generate/unpack) auto-detects the
  format by magic, and `--out x.tem` on niah-train/transplant writes .tem
  directly. Hostile-input covered: corrupt payload fails the checksum loudly,
  wrong-format files get friendly errors (tests C56-C59, 59 CLI tests).
- **Brand**: hand-authored SVG mark + lockup in `assets/` — nine mosaic
  pillars around the amber Hopfield core, wrapped by the O(1) orbit ring
  (one dashed ring, one state dot). `tessera logo` prints the same mark in
  ANSI truecolor on a terminal.
- **CLI v0.2 (friendly surface)**: per-command `--help`/`help <cmd>` (15 pages,
  all with usage + flags + defaults + exit codes), `version|--version|-V`,
  `info` (build+model table), `doctor` (5 env checks, exit 1 on hard fail),
  typo suggestions ("did you mean tessera niah-train?"), cargo-style ANSI color
  (auto off when piped; NO_COLOR wins; CLICOLOR_FORCE=1 forces), zero new deps.
  Tests C41–C55 cover all of it (55 CLI tests total).
- **TRANSPLANT pipeline (the LoLCATs-style weight transplant, CLI edition)**:
  `tessera niah-gen` exports a `TSPOOLv1` pool (header n/seq/kl/vl/distract/seed
  + prompts + answers + offsets; deterministic per seed) → Gemma soft-labels it
  (`TSLABLv1`: n×vl×32 f32 probs, Section X of tessera_vs_gemma.py) →
  `tessera transplant --pool --labels [--kd-weight 0.7 --ans-w 5 --batch 16]`
  distills via `Model::bwd_seq_kd` (teacher KL on the 32 alphabet logits seeded
  through the same analytical BPTT; softcap chain identical to CE). Validated:
  512-pool, 1500 steps × batch 16 × lr 5e-4 → 16/16 mid-train evals on fresh
  prompts; checkpoint loads in the standard `niah-eval`.
- **REAL BUG #2 (data)**: `NiahCfg::min_seq` undercounted the needle (" . " is
  3 B not 2 B) and ignored the head/noise split, so tiny --seq truncated the
  ANSWER out of the prompt (answer.len()==0, mask on a target-less position).
  Correct min: 2*needle + tail - 1. niah-gen now also sample-verifies and the
  transplant loader range-checks answer_start.
- **REAL BUG #3 (recipe)**: single-sample steps cannot form the retrieval
  circuit at short seq (direct niah-train k1/v1@64–128: 0–3/32 even at 3k steps
  — the validated Python port used batch 16). `transplant` therefore does
  gradient accumulation over --batch 16 pool samples per optimizer step
  (gemv_dw accumulates; same pattern as train --batch). Short-seq niah-train
  has no --batch yet (ladder always runs seq>=512 where single steps work).
- **REAL BUG #1 (kernel)**: `gemv_dw_f32_wide` AVX tail loop wrote
  `dy[o]·(Σ_i x[i])` into row 0 instead of the outer product `x[i]·dy[o]` for the
  `no % 16 ≠ 0` columns → every f32 training step used wrong weight gradients for
  the last columns of every matrix (ff=733 → 13 cols each). Fixed; f64 gradcheck
  path unaffected; acceptance #4 re-verified after the fix.
- **REAL BUG #2 (task)**: NIAH `tail_len` off-by-one — " KEY ? VAL" undercounted by
  1 byte ⇒ the last answer char was truncated in EVERY sample (v1 answers were
  empty ⇒ old "16/16" for val-len 1 was vacuous). Fixed: full val_len answers.
- **Robustness**: `sample()` saturates budgets (no underflow panic on tiny seq);
  empty answers never count as recall; save/load bit-exact incl. fresh-stream
  inference equality (U25).

## Curriculum laws for the corrected task (empirical)

- Emission circuits form only at lr ≤ 1e-3; hot restarts (≥2e-3) destroy them.
- Answer positions need `--ans-w ≥ 5` to outrank needle-loss noise targets.
- **Validated phase 1 (k1/v1): lr 5e-4 × 2500 + ans-w 5 + needle-loss → 16/16 REAL
  recall** (checkpoints/smoke_k1_low.bin recipe; hot lr 2e-3 → 1/16).
- Key ramp + value ramp schedules in `scripts/ladder_v6.sh` (v-namespace, fresh).
- `niah_eval_model` shares the training Rng — keep `--eval-every/--warm` defaults.
- `niah-overfit` passes on current code (loss 0.21, RECALL OK).

## Peak quality (new goal, 2026-09-19)

- Held-out BPC harness in `train` (flags above). Tuned defaults per arch.
- tessera 3.520 vs transformer 4.647 at 800x1024 equal steps (each at best lr):
  **1.13 nats/char better**, plus 3.23x long-context throughput. Seed-stable.
- Laws: lr 3e-3 (tessera) / 5e-4 (transformer), wd 0.1, mtp-alpha 0.1; zloss spec-pinned.
- Batching at equal tokens is a loss (more steps > less noise at this scale).

## Next steps

1. `bash scripts/ladder_v6.sh` → acceptance #3 (R8 final gate = the test's exact config).
2. `cargo test --release --no-fail-fast` → expect 74 pass / 1 fail (only #3 until R8 lands).
3. Optional: niah-eval --n 64 for the stronger 64-sample check after R8.
