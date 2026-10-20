#!/bin/bash
# Quality sweep phase 1: cheap loss/schedule knobs, held-out BPC at fixed budget.
# Paired comparison: identical data/seed/holdout across configs.
cd /home/user/tessera
BIN=./target/release/tessera
DATA=/home/user/data/enwik8
COMMON="--data $DATA --max-bytes 6000000 --seq 1024 --steps 200 --holdout-bytes 32768 --bpc-every 0 --seed 11"
RES=quality_results.txt
: > $RES

run() { # $1=tag $2...=extra args
  local tag=$1; shift
  local out outbpc
  out=$($BIN train $COMMON "$@" 2>&1)
  outbpc=$(echo "$out" | grep "FINAL holdout BPC" | grep -oE '[0-9.]+$')
  echo -e "$tag\t$outbpc" | tee -a $RES
}

# two at a time (2 cores)
run A0-baseline &
run A1-alpha0  --mtp-alpha 0.0 &
wait
run A2-alpha01 --mtp-alpha 0.1 &
run A3-alpha06 --mtp-alpha 0.6 &
wait
run Z1-zloss0  --zloss-w 0.0 &
run Z2-zloss1e3 --zloss-w 1e-3 &
wait
run L1-lr5e4   --lr 5e-4 &
run L2-lr2e3   --lr 2e-3 &
wait
run W1-wd005   --wd 0.05 &
run W2-wd01    --wd 0.1 &
wait

echo "== SWEEP PHASE 1 DONE =="
sort -t$'\t' -k2 -n $RES | awk -F'\t' '{printf "%-14s %s\n", $1, $2}'
