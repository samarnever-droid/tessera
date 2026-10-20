#!/bin/bash
# Quality sweep phase 2: winner combos + equal-token batch experiment.
# Batch note: batch=4 accumulates grads over 4 chunks per step -> 4x tokens/step.
# Equal-token compare: batch4 x 200 steps (819K tok) vs batch1 x 800 steps (819K tok).
cd /home/user/tessera
BIN=./target/release/tessera
DATA=/home/user/data/enwik8
RES=quality_results2.txt
: > $RES

run() { # $1=tag $2=steps $3=batch $4...=extra args
  local tag=$1 steps=$2 batch=$3; shift 3
  local out outbpc
  out=$($BIN train --data $DATA --max-bytes 6000000 --seq 1024 --steps $steps --batch $batch \
          --holdout-bytes 32768 --bpc-every 0 --seed 11 "$@" 2>&1)
  outbpc=$(echo "$out" | grep "FINAL holdout BPC" | grep -oE '[0-9.]+$')
  echo -e "$tag\t$outbpc" | tee -a $RES
}

run C1-lr2-a01-wd01 200 1 --lr 2e-3 --mtp-alpha 0.1 --wd 0.1 &
run C2-lr3-a01-wd01 200 1 --lr 3e-3 --mtp-alpha 0.1 --wd 0.1 &
wait
run C3-lr2-a005-wd01 200 1 --lr 2e-3 --mtp-alpha 0.05 --wd 0.1 &
run C4-lr2-a01-wd02 200 1 --lr 2e-3 --mtp-alpha 0.1 --wd 0.2 &
wait
run B4a-batch4-lr2 200 4 --lr 2e-3 --mtp-alpha 0.1 --wd 0.1 &
run B1x800-equaltok 800 1 --lr 2e-3 --mtp-alpha 0.1 --wd 0.1 &
wait

echo "== SWEEP PHASE 2 DONE =="
sort -t$'\t' -k2 -n $RES | awk -F'\t' '{printf "%-20s %s\n", $1, $2}'
