#!/bin/bash
# Gate 4, the lessons: the connectome against shuffled, frozen and MLP controls, three seeds each,
# plus a reversal run. Constants come from the environment so the receipted configuration is one
# line; the defaults are the configuration the pilots selected (see receipts/g4_lessons.json).
#   GAIN=0.02 TEMP=0.3 CAP=3 BALANCE=1 ETA=10 DECISIONS=4000 ./tools/campaign_odour.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/Users/muellerberndt/Projects/oph-meta/cadence/.venv/bin/python}
OUT=${OUT:-runs/learn_odour}
GAIN=${GAIN:-0.02}; TEMP=${TEMP:-0.3}; CAP=${CAP:-3}; ETA=${ETA:-10}; DECISIONS=${DECISIONS:-4000}; BALANCE=${BALANCE:-1}; JOBS=${JOBS:-4}; OMP=${OMP:-2}
BAL=""; [ "$BALANCE" = "1" ] && BAL="--balance"
COMMON="--gain $GAIN --temperature $TEMP --scale-cap $CAP --eta $ETA --decisions $DECISIONS $BAL --report 250 --eval 128 --eval-batch 64"
mkdir -p "$OUT"
run() { # name, then the arguments
  local name=$1; shift
  echo "OMP_NUM_THREADS=$OMP $PY tools/learn_odour.py $* --out $OUT/$name.json --checkpoint $OUT/$name.npz > $OUT/$name.log 2>&1"
}
{
  for s in 0 1 2; do run connectome_s$s --kind connectome --seed $s $COMMON; done
  for s in 0 1 2; do run shuffled_s$s --kind shuffled --seed $s $COMMON; done
  run frozen_s0 --kind frozen --seed 0 $COMMON
  for s in 0 1 2; do run mlp_s$s --kind mlp --seed $s --decisions $DECISIONS --report 250 --eval 128 --eval-batch 64; done
  run connectome_swap_s0 --kind connectome --seed 0 $COMMON --decisions $((2 * DECISIONS)) --swap-at $DECISIONS
} > "$OUT/queue.txt"
echo "$(wc -l < "$OUT/queue.txt") runs queued at $JOBS in parallel (OMP $OMP); logs in $OUT"
xargs -P "$JOBS" -I CMD bash -c CMD < "$OUT/queue.txt"
echo "campaign done"
