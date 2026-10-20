#!/bin/bash
# NIAH ladder v5 — idempotent, gated, resumable.
# Path: finish value-ramp at k1 (K3@15/16) -> key-ramp k2,k4 at v2 (hot, painless)
#       -> v4 gentle at k4 -> length ramp 512/2048/8192 -> FINAL 16/16 @8192.
# Lessons baked in (do not regress):
#   * value-len growth needs LOW lr (<=1e-3) + patience; hot destroys emission
#   * key-len growth is painless at hot lr (probe A: 16/16 in 2000 steps)
#   * niah_eval_model shares the training Rng -> keep --eval-every/--warm defaults
#   * --ans-w upweights answer positions vs needle-loss noise (keep 3.0 on gentle rungs)
#   * each rung saves a checkpoint + ok-marker -> rerunning this script resumes
set -e
cd /home/user/tessera
BIN=./target/release/tessera
CK=checkpoints
SEED=1234
mkdir -p $CK

gate() { # $1=model $2=seq $3=kl $4=vl $5=min -> hard gate, writes ok-marker on pass
  local score
  score=$($BIN niah-eval --model "$1" --seq "$2" --key-len "$3" --val-len "$4" --n 16 \
            | grep -oE '[0-9]+/16' | head -1 | cut -d/ -f1)
  echo "== GATE $1 seq=$2 kl=$3 vl=$4 => ${score}/16 (min $5) =="
  if [ "$score" -ge "$5" ]; then touch "$1.ok"; else exit 1; fi
}
softgate() {
  local score
  score=$($BIN niah-eval --model "$1" --seq "$2" --key-len "$3" --val-len "$4" --n 16 \
            | grep -oE '[0-9]+/16' | head -1 | cut -d/ -f1)
  echo "== SOFTGATE $1 seq=$2 kl=$3 vl=$4 => ${score}/16 (continuing) =="
}
rung() { # skip if marker exists
  [ -f "$1.ok" ] && { echo "== RUNG $1 ALREADY PASSED, SKIP =="; return 0; }
  return 1
}

echo "== LADDER v5 START $(date) =="

# ---------- R1: finish v2 at k1 (input = probeK3, 15/16) ----------
if ! rung $CK/niah_k1v2; then
  cp -f probes/probeK3.bin $CK/niah_k1v2.bin
  $BIN niah-train --seq 128 --key-len 1 --val-len 2 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 3.0 --out $CK/niah_k1v2.bin --resume
  gate $CK/niah_k1v2.bin 128 1 2 14
fi

# ---------- R2: key ramp k1->k2 at v2 (GENTLE + mix: young v2 circuit is fragile;
#          hot 2e-3 destroyed it (0/16). Key binding must form at low lr too.) ----------
if ! rung $CK/niah_k2v2; then
  cp -f $CK/niah_k1v2.bin $CK/niah_k2v2.bin
  $BIN niah-train --seq 128 --key-len 2 --val-len 2 --steps 4000 --lr 1e-3 --seed $SEED \
    --needle-loss --ans-w 3.0 --mix-kl 1 --mix-vl 2 --mix-p 0.4 --out $CK/niah_k2v2.bin --resume
  gate $CK/niah_k2v2.bin 128 2 2 12
  $BIN niah-train --seq 128 --key-len 2 --val-len 2 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 3.0 --mix-kl 1 --mix-vl 2 --mix-p 0.2 --out $CK/niah_k2v2.bin --resume
  gate $CK/niah_k2v2.bin 128 2 2 14
fi

# ---------- R3: key ramp k2->k4 at v2 (same gentle+mix shape) ----------
if ! rung $CK/niah_k4v2; then
  cp -f $CK/niah_k2v2.bin $CK/niah_k4v2.bin
  $BIN niah-train --seq 128 --key-len 4 --val-len 2 --steps 4000 --lr 1e-3 --seed $SEED \
    --needle-loss --ans-w 3.0 --mix-kl 2 --mix-vl 2 --mix-p 0.4 --out $CK/niah_k4v2.bin --resume
  gate $CK/niah_k4v2.bin 128 4 2 12
  $BIN niah-train --seq 128 --key-len 4 --val-len 2 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 3.0 --mix-kl 2 --mix-vl 2 --mix-p 0.2 --out $CK/niah_k4v2.bin --resume
  gate $CK/niah_k4v2.bin 128 4 2 14
fi

# ---------- R4: value ramp v2->v4 at k4 (gentle, two cycles) ----------
if ! rung $CK/niah_k4v4; then
  cp -f $CK/niah_k4v2.bin $CK/niah_k4v4.bin
  $BIN niah-train --seq 128 --key-len 4 --val-len 4 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 3.0 --out $CK/niah_k4v4.bin --resume
  gate $CK/niah_k4v4.bin 128 4 4 12
  $BIN niah-train --seq 128 --key-len 4 --val-len 4 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 3.0 --out $CK/niah_k4v4.bin --resume
  gate $CK/niah_k4v4.bin 128 4 4 14
fi

# ---------- R5: length 512 ----------
if ! rung $CK/niah_s512; then
  cp -f $CK/niah_k4v4.bin $CK/niah_s512.bin
  $BIN niah-train --seq 512 --key-len 4 --val-len 4 --steps 2000 --lr 1e-3 --seed $SEED \
    --needle-loss --out $CK/niah_s512.bin --resume
  softgate $CK/niah_s512.bin 512 4 4 12
fi

# ---------- R6: length 2048 ----------
if ! rung $CK/niah_s2048; then
  cp -f $CK/niah_s512.bin $CK/niah_s2048.bin
  $BIN niah-train --seq 2048 --key-len 4 --val-len 4 --steps 1200 --lr 1e-3 --seed $SEED \
    --needle-loss --out $CK/niah_s2048.bin --resume
  softgate $CK/niah_s2048.bin 2048 4 4 12
fi

# ---------- R7: length 8192 + FINAL ----------
if ! rung $CK/niah; then
  cp -f $CK/niah_s2048.bin $CK/niah.bin
  $BIN niah-train --seq 8192 --key-len 4 --val-len 4 --steps 400 --lr 1e-3 --seed $SEED \
    --needle-loss --out $CK/niah.bin --resume
  gate $CK/niah.bin 8192 4 4 16
fi

echo "== LADDER v5 DONE $(date) =="
