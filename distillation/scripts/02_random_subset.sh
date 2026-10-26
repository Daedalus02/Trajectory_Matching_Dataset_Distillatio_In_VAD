#!/usr/bin/env bash
# Equal-budget baselines (thesis Sec. 4.1, STFPM @64): for each category, 5 runs of
#   initial10   = images_0.pt (the real images the distillation starts from)
#   distilled10 = last distillation iteration
#   real10      = 10 random real images (new draw per run)
# Needs 01_single_class.sh stfpm (64) first.  Usage: bash scripts/02_random_subset.sh [cat ...]
# Output: outputs/random_subset/results.csv
set -uo pipefail
source "$(dirname "$0")/config.sh"

CATS=("$@"); [ ${#CATS[@]} -eq 0 ] && CATS=("${ALL_CATS[@]}")
RES=64; EP=200
LOG=$OUT/random_subset/logs; SUB=$OUT/random_subset/subsets; mkdir -p "$LOG" "$SUB"
CSV=$OUT/random_subset/results.csv
csv_init "$CSV" "category,setting,res,run,image_auroc,pixel_auroc,image_f1max,pixel_f1max"

ev() {  # setting train_set run category
  local L=$LOG/$4_$1_r$3.log
  python -u evaluate_pixel.py --category $4 --img_size $RES --epochs $EP --lr 0.1 --train_dir "$2" > "$L" 2>&1
  echo "$4,$1,$RES,$3,$(metrics "$L")" >> "$CSV"
}

for CAT in "${CATS[@]}"; do
  DIR=$OUT/distilled/$CAT/$IPC/real/full$RES
  LAST=$(last_ckpt "$DIR")
  [ -f "$DIR/images_0.pt" ] && [ -n "$LAST" ] || { echo "skip $CAT (run 01_single_class.sh stfpm first)"; continue; }
  for R in $(seq 1 $N_RUNS); do
    ev initial10   "$DIR/images_0.pt" $R $CAT
    ev distilled10 "$LAST" $R $CAT
    python make_real_subset.py --src "$DATA/$CAT/train/good" --n 10 --res $RES --seed $R --out "$SUB/${CAT}_r$R.pt"
    ev real10 "$SUB/${CAT}_r$R.pt" $R $CAT
  done
  echo "done $CAT"
done
