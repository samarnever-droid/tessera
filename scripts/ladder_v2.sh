#!/bin/bash
# NIAH curriculum v2 — full 6-phase ladder (restart; no checkpoints survived snapshot)
# 128/k1 -> 128/k2 -> 128/k4 -> 512/k4 -> 2048/k4 -> 8192/k4 -> checkpoints/niah.bin
# --resume loads from --out, so each chained phase first copies the previous phase's weights.
set -e
cd /home/user/tessera
BIN=./target/release/tessera
SEED=1234
CK=checkpoints

echo "== LADDER START $(date) =="

echo "== PHASE 1/6: seq=128 key-len=1 steps=2000 lr=5e-3 =="
$BIN niah-train --seq 128 --key-len 1 --steps 2000 --lr 5e-3 --seed $SEED \
  --needle-loss --warm 0 --eval-every 100 --out $CK/niah_k1.bin

echo "== PHASE 2/6: seq=128 key-len=2 steps=3000 lr=3e-3 =="
cp -f $CK/niah_k1.bin $CK/niah_k2.bin
$BIN niah-train --seq 128 --key-len 2 --steps 3000 --lr 3e-3 --seed $SEED \
  --needle-loss --warm 0 --eval-every 100 --out $CK/niah_k2.bin --resume

echo "== PHASE 3/6: seq=128 key-len=4 steps=4000 lr=3e-3 =="
cp -f $CK/niah_k2.bin $CK/niah_k4.bin
$BIN niah-train --seq 128 --key-len 4 --steps 4000 --lr 3e-3 --seed $SEED \
  --needle-loss --warm 0 --eval-every 100 --out $CK/niah_k4.bin --resume

echo "== PHASE 4/6: seq=512 key-len=4 steps=2000 lr=2e-3 =="
cp -f $CK/niah_k4.bin $CK/niah_s512.bin
$BIN niah-train --seq 512 --key-len 4 --steps 2000 --lr 2e-3 --seed $SEED \
  --needle-loss --eval-every 100 --out $CK/niah_s512.bin --resume

echo "== PHASE 5/6: seq=2048 key-len=4 steps=1200 lr=1.5e-3 =="
cp -f $CK/niah_s512.bin $CK/niah_s2048.bin
$BIN niah-train --seq 2048 --key-len 4 --steps 1200 --lr 1.5e-3 --seed $SEED \
  --needle-loss --eval-every 100 --out $CK/niah_s2048.bin --resume

echo "== PHASE 6/6: seq=8192 key-len=4 steps=400 lr=1e-3 =="
cp -f $CK/niah_s2048.bin $CK/niah.bin
$BIN niah-train --seq 8192 --key-len 4 --steps 400 --lr 1e-3 --seed $SEED \
  --needle-loss --eval-every 100 --out $CK/niah.bin --resume

echo "== LADDER DONE $(date) =="
