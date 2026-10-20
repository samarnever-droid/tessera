#!/bin/bash
# NIAH curriculum v3 — REPRODUCTION of the known-good v2 recipe.
# LESSON (do not regress): niah_eval_model draws from the SAME Rng as training
# samples, and --warm shifts the first samples' lengths. Any change to
# --eval-every / --warm changes the training data stream; the k1 task is
# trajectory-chaotic and plateaued at alphabet entropy when flags were tweaked.
# => use trainer defaults (--warm 30, --eval-every 25) and seed 1234, exactly
#    like the previously successful run (16/16 @ k1 seq 128).
set -e
cd /home/user/tessera
BIN=./target/release/tessera
CK=checkpoints
SEED=1234

gate() { # $1=model $2=seq $3=keylen $4=min  -> hard gate
  local score
  score=$($BIN niah-eval --model "$1" --seq "$2" --key-len "$3" --n 16 \
            | grep -oE '[0-9]+/16' | head -1 | cut -d/ -f1)
  echo "== GATE $1 seq=$2 kl=$3 => ${score}/16 (min $4) =="
  [ "$score" -ge "$4" ]
}
softgate() { # same but warn only (uncharted territory)
  local score
  score=$($BIN niah-eval --model "$1" --seq "$2" --key-len "$3" --n 16 \
            | grep -oE '[0-9]+/16' | head -1 | cut -d/ -f1)
  echo "== SOFTGATE $1 seq=$2 kl=$3 => ${score}/16 (continuing) =="
}

echo "== LADDER v3 START $(date) =="

echo "== PHASE 1/6: seq=128 key-len=1 steps=2000 lr=5e-3 (defaults) =="
$BIN niah-train --seq 128 --key-len 1 --steps 2000 --lr 5e-3 --seed $SEED \
  --needle-loss --out $CK/niah_k1.bin
gate $CK/niah_k1.bin 128 1 14

echo "== PHASE 2/6: seq=128 key-len=2 steps=3000 lr=3e-3 =="
cp -f $CK/niah_k1.bin $CK/niah_k2.bin
$BIN niah-train --seq 128 --key-len 2 --steps 3000 --lr 3e-3 --seed $SEED \
  --needle-loss --out $CK/niah_k2.bin --resume
gate $CK/niah_k2.bin 128 2 14

echo "== PHASE 3/6: seq=128 key-len=4 steps=4000 lr=3e-3 =="
cp -f $CK/niah_k2.bin $CK/niah_k4.bin
$BIN niah-train --seq 128 --key-len 4 --steps 4000 --lr 3e-3 --seed $SEED \
  --needle-loss --out $CK/niah_k4.bin --resume
gate $CK/niah_k4.bin 128 4 14

echo "== PHASE 4/6: seq=512 key-len=4 steps=2000 lr=2e-3 =="
cp -f $CK/niah_k4.bin $CK/niah_s512.bin
$BIN niah-train --seq 512 --key-len 4 --steps 2000 --lr 2e-3 --seed $SEED \
  --needle-loss --out $CK/niah_s512.bin --resume
softgate $CK/niah_s512.bin 512 4 12

echo "== PHASE 5/6: seq=2048 key-len=4 steps=1200 lr=1.5e-3 =="
cp -f $CK/niah_s512.bin $CK/niah_s2048.bin
$BIN niah-train --seq 2048 --key-len 4 --steps 1200 --lr 1.5e-3 --seed $SEED \
  --needle-loss --out $CK/niah_s2048.bin --resume
softgate $CK/niah_s2048.bin 2048 4 10

echo "== PHASE 6/6: seq=8192 key-len=4 steps=400 lr=1e-3 =="
cp -f $CK/niah_s2048.bin $CK/niah.bin
$BIN niah-train --seq 8192 --key-len 4 --steps 400 --lr 1e-3 --seed $SEED \
  --needle-loss --out $CK/niah.bin --resume

echo "== FINAL GATE =="
gate $CK/niah.bin 8192 4 16

echo "== LADDER v3 DONE $(date) =="
