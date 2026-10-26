#!/usr/bin/env bash
# Multi-class study (thesis Sec. 4.5): one model trained on the pool of all 15 categories,
# evaluated per category, on real data and on the distilled sets (best iteration, as in the thesis).
# Needs 01_single_class.sh for the distilled sets.  Usage: bash scripts/03_multiclass.sh
# Output: outputs/multiclass/results_multiclass_3det.csv (RD4AD, Dinomaly), results_multiclass.csv (STFPM)
set -uo pipefail
source "$(dirname "$0")/config.sh"

LOG=$OUT/multiclass/logs; mkdir -p "$LOG"
TPL="$OUT/single_class/images/{det}_{cat}_{res}.pt"

# RD4AD @64 and Dinomaly @224, runs 0..4
for SETUP in "rd4ad 64" "dinomaly 224"; do
  set -- $SETUP
  for TYPE in distilled real; do
    for RUN in $(seq 0 $((N_RUNS - 1))); do
      python -u evaluate_multiclass.py --detector $1 --type $TYPE --res $2 --run $RUN \
        --pt_template "$TPL" --out_csv "$OUT/multiclass/results_multiclass_3det.csv" \
        > "$LOG/mc_$1_$2_${TYPE}_r$RUN.log" 2>&1
    done
  done
done

# STFPM @64 (one script does both types and all runs)
python -u evaluate_multiclass_stfpm.py --types distilled real --runs $N_RUNS --epochs 400 --res 64 \
  --dist_template "$OUT/single_class/images/stfpm_{cat}_{res}.pt" \
  --csv "$OUT/multiclass/results_multiclass.csv" > "$LOG/mc_stfpm_64.log" 2>&1
