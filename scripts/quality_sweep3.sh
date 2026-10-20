#!/bin/bash
# Quality phase 3: head-to-head with the tuned recipe + longer-budget trajectory.
cd /home/user/tessera
BIN=./target/release/tessera
DATA=/home/user/data/enwik8
RES=quality_results3.txt
: > $RES

trun() { # $1=tag $2=steps $3=arch $4...=extra
  local tag=$1 steps=$2 arch=$3; shift 3
  local out outbpc
  out=$($BIN train --data $DATA --max-bytes 6000000 --seq 1024 --steps $steps --batch 1 \
          --holdout-bytes 32768 --bpc-every 0 --seed 11 --arch $arch --lr 3e-3 --wd 0.1 "$@" 2>&1)
  outbpc=$(echo "$out" | grep "FINAL holdout BPC" | grep -oE '[0-9.]+$')
  echo -e "$tag\t$outbpc" | tee -a $RES
}

trun H1-tessera-400 400 tessera --mtp-alpha 0.1 &
trun H2-transformer-400 400 transformer &
wait
trun H3-tessera-lr4e3-200 200 tessera --lr 4e-3 --mtp-alpha 0.1 &
trun H4-tessera-800 800 tessera --mtp-alpha 0.1 &
wait

echo "== PHASE 3 DONE =="
sort -t$'\t' -k2 -n $RES | awk -F'\t' '{printf "%-24s %s\n", $1, $2}'
