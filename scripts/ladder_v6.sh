#!/bin/bash
# NIAH ladder v6 — for the CORRECTED task (full-length answers, off-by-one fixed).
# Old-task checkpoints are incompatible: answers gained their final character.
# Recipe laws (established empirically, do not regress):
#   * emission circuits form ONLY at low lr (<=1e-3); hot restarts destroy them
#   * answer signal needs --ans-w >= 5 to outrank needle-loss noise
#     (validated: k1/v1 16/16 REAL recall with lr 5e-4 + ans-w 5 + needle-loss)
#   * key-length growth is also gentle now that eval is honest
#   * keep --eval-every/--warm defaults (eval shares the training RNG)
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
rung() { [ -f "$1.ok" ] && { echo "== RUNG $1 ALREADY PASSED, SKIP =="; return 0; }; return 1; }

echo "== LADDER v6 START $(date) =="

# ---------- R1: k1/v1 — VALIDATED: 16/16 real recall ----------
if ! rung $CK/v6_k1; then
  $BIN niah-train --seq 128 --key-len 1 --val-len 1 --steps 2500 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 5.0 --warm 0 --out $CK/v6_k1.bin
  gate $CK/v6_k1.bin 128 1 1 14
fi

# ---------- R2: k1/v2 — value ramp, gentle cycles ----------
if ! rung $CK/v6_k1v2; then
  cp -f $CK/v6_k1.bin $CK/v6_k1v2.bin
  $BIN niah-train --seq 128 --key-len 1 --val-len 2 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 5.0 --warm 0 --out $CK/v6_k1v2.bin --resume
  gate $CK/v6_k1v2.bin 128 1 2 12
  $BIN niah-train --seq 128 --key-len 1 --val-len 2 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 5.0 --warm 0 --out $CK/v6_k1v2.bin --resume
  gate $CK/v6_k1v2.bin 128 1 2 14
fi

# ---------- R3: k2/v2 — key ramp, gentle + mix rehearsal ----------
if ! rung $CK/v6_k2v2; then
  cp -f $CK/v6_k1v2.bin $CK/v6_k2v2.bin
  $BIN niah-train --seq 128 --key-len 2 --val-len 2 --steps 4000 --lr 1e-3 --seed $SEED \
    --needle-loss --ans-w 5.0 --warm 0 --mix-kl 1 --mix-vl 2 --mix-p 0.4 \
    --out $CK/v6_k2v2.bin --resume
  gate $CK/v6_k2v2.bin 128 2 2 12
  $BIN niah-train --seq 128 --key-len 2 --val-len 2 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 5.0 --warm 0 --mix-kl 1 --mix-vl 2 --mix-p 0.2 \
    --out $CK/v6_k2v2.bin --resume
  gate $CK/v6_k2v2.bin 128 2 2 14
fi

# ---------- R4: k4/v2 — key ramp continues ----------
if ! rung $CK/v6_k4v2; then
  cp -f $CK/v6_k2v2.bin $CK/v6_k4v2.bin
  $BIN niah-train --seq 128 --key-len 4 --val-len 2 --steps 4000 --lr 1e-3 --seed $SEED \
    --needle-loss --ans-w 5.0 --warm 0 --mix-kl 2 --mix-vl 2 --mix-p 0.4 \
    --out $CK/v6_k4v2.bin --resume
  gate $CK/v6_k4v2.bin 128 4 2 12
  $BIN niah-train --seq 128 --key-len 4 --val-len 2 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 5.0 --warm 0 --mix-kl 2 --mix-vl 2 --mix-p 0.2 \
    --out $CK/v6_k4v2.bin --resume
  gate $CK/v6_k4v2.bin 128 4 2 14
fi

# ---------- R5: k4/v4 — final value ramp, two gentle cycles ----------
if ! rung $CK/v6_k4v4; then
  cp -f $CK/v6_k4v2.bin $CK/v6_k4v4.bin
  $BIN niah-train --seq 128 --key-len 4 --val-len 4 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 5.0 --warm 0 --out $CK/v6_k4v4.bin --resume
  gate $CK/v6_k4v4.bin 128 4 4 12
  $BIN niah-train --seq 128 --key-len 4 --val-len 4 --steps 4000 --lr 5e-4 --seed $SEED \
    --needle-loss --ans-w 5.0 --warm 0 --out $CK/v6_k4v4.bin --resume
  gate $CK/v6_k4v4.bin 128 4 4 14
fi

# ---------- R6: length 512 ----------
if ! rung $CK/v6_s512; then
  cp -f $CK/v6_k4v4.bin $CK/v6_s512.bin
  $BIN niah-train --seq 512 --key-len 4 --val-len 4 --steps 2000 --lr 1e-3 --seed $SEED \
    --needle-loss --ans-w 5.0 --out $CK/v6_s512.bin --resume
  softgate $CK/v6_s512.bin 512 4 4 12
fi

# ---------- R7: length 2048 ----------
if ! rung $CK/v6_s2048; then
  cp -f $CK/v6_s512.bin $CK/v6_s2048.bin
  $BIN niah-train --seq 2048 --key-len 4 --val-len 4 --steps 1200 --lr 1e-3 --seed $SEED \
    --needle-loss --ans-w 5.0 --out $CK/v6_s2048.bin --resume
  softgate $CK/v6_s2048.bin 2048 4 4 12
fi

# ---------- R8: length 8192 + FINAL (acceptance #3: k4/v4 seed 999123) ----------
if ! rung $CK/niah; then
  cp -f $CK/v6_s2048.bin $CK/niah.bin
  $BIN niah-train --seq 8192 --key-len 4 --val-len 4 --steps 400 --lr 1e-3 --seed $SEED \
    --needle-loss --ans-w 5.0 --out $CK/niah.bin --resume
  gate $CK/niah.bin 8192 4 4 16
fi

echo "== LADDER v6 DONE $(date) =="
