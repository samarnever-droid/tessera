# TESSERA-Q

An autoregressive sequence model built from bare-metal Rust (zero dependencies)
with AVX-512/AVX2 SIMD kernels, designed around the LLM trilemma:

| Axis | TESSERA-Q | Transformer |
|---|---|---|
| Working memory | **O(1)** — 144 KiB substrate at any context length | O(T) KV cache |
| Long-term store | **on-chip Modern Hopfield memory** (exact BPTT) | none (or external DB) |
| Attention noise | **Differential Attention** (λ-cancellation) | plain softmax |

## The 9 pillars (strict order in every layer/step)

1. **Affine RMSNorm** — pre-attention, learnable gamma.
2. **1-D causal depthwise highway conv** k=4: `x_conv_t = Σ_j W_conv[j]·x_{t−j}`;
   `h_local = x_conv ⊙ σ(W_gate·x_t)`.
3. **Per-head QK-RMSNorm** — bounds dot products in [−d, d].
4. **Adaptive RoPE banding** — `θ_i = 10000^(−2i/d_k)·2σ(η_i)`, learnable `η_i`.
5. **Differential Attention** — `(softmax(A₁) − λ_eff·softmax(A₂))V` with
   `λ_init(l) = 0.8 − 0.6e^(−0.3(l−1))`, `λ_eff = max(0, e^{a} − e^{b} + λ_init)`.
6. **ResFormer value residual** — `V_mixed = w·V_l + (1−w)·V₀`, `w = σ(gate)`.
7. **SwiGLU 6× FFN + parallel low-rank adapter** (r=8).
8. **MRM-v2 Hopfield memory** — 128 fine slots + 16 coarse centroids (144 KiB),
   3-tier state machine keyed on `s = cos(k_write, k_slot)`:
   - `s ≥ t_overwrite(d) = min(0.9999, 8.014/√d)` → hard overwrite;
   - `t_merge(d) = min(t_overwrite − 1e-4, 6.50/√d) ≤ s < t_overwrite` → soft merge
     with Poincaré-ball Riemannian EMA (`∇_R = ((1−‖c‖²)²/4)·∇_E`);
   - `s < t_merge` → insert, evicting `argmin 2·hits[i] + S_i`, `S_i = ‖v_w − ctx‖`.
   Writes are gated (`σ(glin) ≤ 0.5` → skip). An `MrmSnapshot` is saved **per causal
   step** and the backward pass walks the tape in reverse, restoring state exactly.
9. **Soft-capped logits** (`30·tanh`) + **dual-head MTP** (aux head predicts t+2) +
   **aux Z-loss** `10⁻⁴·log²(Σe^z)`.

Training: AVX-accelerated AdamW, WSD schedule (10% linear warmup, 70% stable,
20% cosine decay to 1% of peak), safe pre-norm `v/max(‖v‖₂, 10⁻⁸)`.

## Layout

```
src/num.rs        SIMD kernels (gemv/gemv_dx/gemv_dw/dot/softmax), f32 AVX-512 + f64 reference
src/norms.rs      RMSNorm fwd/bwd, safe_inv_norm
src/rope.rs       adaptive RoPE (learnable per-frequency banding)
src/conv.rs       causal depthwise highway conv k=4 (streaming, O(1) state)
src/attention.rs  differential attention w/ QK-norm + RoPE
src/ffn.rs        SwiGLU + parallel LoRA
src/mrm.rs        Modern Hopfield memory: 3-tier writes, exact per-step snapshot BPTT
src/layer.rs      Stage = 9 pillars per step (fwd + bwd)
src/model.rs      multi-layer model, heads, loss, save/load
src/optim.rs      AdamW + grad clipping
src/data.rs       enwik8 corpus, NIAH generator, greedy recall eval
src/baseline.rs   parameter-matched causal Transformer (blocked attention, O(T) KV cache)
src/gradcheck.rs  finite-difference verification suite (666 checks)
```

## Peak quality (enwik8, held-out BPC, equal-step budget, each arch at its best lr)

`train` now ships a quality harness: `--mtp-alpha --zloss-w --cap --wd --batch
--holdout-bytes --bpc-every` plus a streaming held-out BPC evaluator (O(1), main head).

Tuned recipe (now the CLI defaults): **tessera lr 3e-3, wd 0.1, mtp-alpha 0.1**;
**transformer lr 5e-4, wd 0.1** (its stability limit without QK-norm).

| steps @1024 | tessera | transformer | gap |
|---|---|---|---|
| 200 | 4.046 | 5.317 | −1.27 |
| 400 | 3.765 | — | — |
| 800 | **3.520** (seed-12: 3.554) | **4.647** | **−1.13 nats/char** |

The gap is structural: pillar-3 QK-RMSNorm lets tessera train at 6x the
transformer's stable learning rate, and the highway conv + MRM add real
sequence-mixing capacity per parameter. Z-loss weight stays at the spec value
1e-4 (setting it to 0 measured 0.02 better at 200 steps but is spec-pinned).
Batching at equal tokens LOSES to more optimizer steps at this scale (batch-4
x 200 steps = 3.76 vs batch-1 x 800 steps = 3.55) — small-batch many-steps
wins; `--batch` remains available for gradient-accumulation on small RAM.

## CLI contract

Exit codes: `0` success | `2` usage or validation error | `1` runtime failure.

- Unknown flags, unknown subcommands, missing flag values, non-numeric values, and
  stray positional arguments are **rejected before any work starts** (exit 2, with the
  offending token named on stderr).
- Guards: `--seq` too small for the NIAH task (`NiahCfg::min_seq()`), `--steps/--n` < 1,
  `--lr <= 0`, missing model/data files, data smaller than `seq+1`, `--resume` without
  an existing checkpoint — all produce `error: ...` messages, never panics.
- Determinism: same seed + same flags ⇒ bit-identical checkpoints (tested).
- `tessera help` (or `--help`, `-h`, or bare invocation) prints usage and exits 0.

## Correctness notes (hardening pass)

- `gemv_dw` AVX tail loop (f32): the scalar tail once accumulated
  `dy[o]·(Σx)` into row 0 instead of the outer product `x[i]·dy[o]` for
  `no % 16 ≠ 0` columns — caught by the unit-test pass, fixed, and now covered by
  `u16_gemv_dw_accumulates_outer_product` (runs on the real SIMD path).
- NIAH `tail_len` off-by-one: the query tail " KEY ? VAL" was undercounted by one
  byte, so the last answer character was silently truncated in every sample
  (v1 answers were even empty). Fixed: answers now carry the full `val_len`
  characters; `NiahCfg::min_seq()` guards the CLI.
- Checkpoint save/load is bit-exact for every parameter and reproduces bit-identical
  inference on a fresh token stream (`u25_save_load_bitexact_small_cfg`).

## Acceptance criteria (all verified by tests in `tests/acceptance.rs`)

Run `cargo test --release` (needs the enwik8 corpus at `../data/enwik8` and a
trained NIAH checkpoint — see below).

1. **Gradients** — `tessera gradcheck`: 666 finite-difference checks across all 9
   pillars, the MRM snapshot tape, the streaming conv, cross-time stage assembly and
   the full model; worst relative error ≈ 4e-6 (threshold 1e-4).
2. **O(1) memory** — `tessera memtest --steps 100000`: identical working state
   (153,960 B) at T=64 and T=100,000; substrate 147,456 B ≈ 144 KiB (~145 KB spec).
3. **Long-context recall** — a needle-in-a-haystack curriculum trains the write-gate/
   key-binding circuit at short range first, then extends:
   ```
   # binding ladder (grow the key space 1 -> 2 -> 4 chars, then the range)
   tessera niah-train --seq 128  --steps 2000 --lr 5e-3 --key-len 1 --val-len 1 --needle-loss --out checkpoints/niah_k1.bin
   tessera niah-train --seq 128  --steps 3000 --lr 3e-3 --key-len 2 --val-len 2 --needle-loss --resume --out checkpoints/niah_k1.bin
   tessera niah-train --seq 128  --steps 4000 --lr 3e-3 --key-len 4 --val-len 4 --needle-loss --resume --out checkpoints/niah_k1.bin
   tessera niah-train --seq 512  --steps 2000 --lr 2e-3 --needle-loss --resume --out checkpoints/niah_k1.bin
   tessera niah-train --seq 2048 --steps 1200 --lr 1.5e-3 --needle-loss --resume --out checkpoints/niah_k1.bin
   tessera niah-train --seq 8192 --steps 400  --lr 1e-3 --needle-loss --resume --eval-seq 8192 --out checkpoints/niah.bin
   tessera niah-eval --seq 8192 --n 64     # greedy exact-match recall
   ```
   `--needle-loss` adds auxiliary supervision on the needle spans themselves; the
   single-sample feasibility probe is `tessera niah-overfit`.
4. **Parity + speed** — `tessera parity --data /path/to/enwik8`: BPC parity vs a
   parameter-matched Transformer and > 2× training throughput at 8k contexts
   (measured 3.2× on this host).

## Usage

```
cargo build --release
./target/release/tessera gradcheck          # FD verification (666 checks)
./target/release/tessera memtest            # O(1) memory proof to 100k tokens
./target/release/tessera selftest           # overfit smoke test
./target/release/tessera niah-overfit       # single-sample recall feasibility
./target/release/tessera niah-train --seq 8192 --steps 3000 --warm 300
./target/release/tessera niah-eval --seq 8192 --n 64
./target/release/tessera train --data /path/to/enwik8 --arch tessera --steps 5000
./target/release/tessera parity --data /path/to/enwik8 --steps 200
```

Toolchain note: Rust lives under `$HOME/.cache` on this host —
`export RUSTUP_HOME=$HOME/.cache/rustup CARGO_HOME=$HOME/.cache/cargo PATH=$HOME/.cache/cargo/bin:$PATH`.


## License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
