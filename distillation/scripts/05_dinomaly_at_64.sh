#!/usr/bin/env bash
# Backbone-vs-resolution ablation (thesis Sec. 4.4): real and distilled training images are
# downsampled to 64 and upsampled back to 224, then Dinomaly is trained on them (image AUROC).
# Uses the best-iteration distilled sets, as in the thesis. Needs 01_single_class.sh dinomaly.
# Usage: bash scripts/05_dinomaly_at_64.sh [cat ...]      Output: outputs/dino64/results.csv
set -uo pipefail
source "$(dirname "$0")/config.sh"

CATS=("$@"); [ ${#CATS[@]} -eq 0 ] && CATS=("${ALL_CATS[@]}")
D=$OUT/dino64; mkdir -p "$D/logs"
CSV=$D/results.csv
csv_init "$CSV" "category,type,image_auroc"

for CAT in "${CATS[@]}"; do
  python degrade_res.py --src "$DATA/$CAT/train/good" --down 64 --up 224 --out "$D/${CAT}_real.pt"
  python degrade_res.py --src "$OUT/distilled_dinomaly/$CAT/$IPC/real/full224/images_best.pt" \
    --down 64 --up 224 --out "$D/${CAT}_dist.pt"
  for T in real dist; do
    L=$D/logs/${CAT}_$T.log
    python -u evaluate_dinomaly.py --category $CAT --img_size 224 --epochs 200 \
      --train_dir "$D/${CAT}_$T.pt" --out_dir "$D/eval_$T" > "$L" 2>&1
    echo "$CAT,$T,$(val "$L" 'Image-level AUROC')" >> "$CSV"
  done
done
