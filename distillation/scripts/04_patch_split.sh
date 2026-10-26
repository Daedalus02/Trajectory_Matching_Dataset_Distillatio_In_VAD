#!/usr/bin/env bash
# Patch-split study (thesis Sec. 4.9, STFPM @224): each image is split into four 112x112
# quadrants; one buffer + distillation (150 it.) per quadrant, then
#   (a) four separate detectors, maps tiled back to 224   (evaluate_pixel_4models.py)
#   (b) one detector on the pooled quadrants               (evaluate_pixel_1model_quadrants.py)
# Usage: bash scripts/04_patch_split.sh [cat ...]
# Output: outputs/patch/results_4models.csv, outputs/patch/results_1model.csv
set -uo pipefail
source "$(dirname "$0")/config.sh"

CATS=("$@"); [ ${#CATS[@]} -eq 0 ] && CATS=("${ALL_CATS[@]}")
LOG=$OUT/patch/logs; mkdir -p "$LOG"
CSV4=$OUT/patch/results_4models.csv; CSV1=$OUT/patch/results_1model.csv
csv_init "$CSV4" "category,image_auroc,pixel_auroc"
csv_init "$CSV1" "category,image_auroc,pixel_auroc"

for CAT in "${CATS[@]}"; do
  for Q in 0 1 2 3; do
    [ -f "$OUT/distilled/$CAT/$IPC/real/q${Q}_224/images_best.pt" ] && continue   # already distilled
    BUF=$OUT/buffer_patch_${CAT}_q$Q
    [ -f "$BUF/MVTec_feature/ResNet18/replay_buffer_0.pt" ] || python -u generate_buffer.py --category $CAT \
      --num_experts 10 --train_epochs 20 --optimizer sgd --res 224 --quadrant $Q \
      --buffer_path "$BUF" > "$LOG/buf_${CAT}_q$Q.log" 2>&1
    python -u distill.py --category $CAT \
      --ipc $IPC --batch_syn $IPC --Iteration 150 --eval_it 50 --eval_epochs 100 --syn_steps 15 \
      --expert_epochs 2 --lr_img 0.01 --max_start_epoch 5 --res 224 --quadrant $Q \
      --buffer_path "$BUF" > "$LOG/dist_${CAT}_q$Q.log" 2>&1
  done
  # best-iteration sets, as in the thesis
  PT="$OUT/distilled/$CAT/$IPC/real/q{q}_224/images_best.pt"

  L=$LOG/eval4_$CAT.log
  python -u evaluate_pixel_4models.py --category $CAT --pt_template "$PT" --epochs 200 \
    --img_size 224 --norm none > "$L" 2>&1
  echo "$CAT,$(val "$L" 'MERGE norm=none] Image-AUROC'),$(val "$L" 'MERGE norm=none] Pixel-AUROC')" >> "$CSV4"

  L=$LOG/eval1_$CAT.log
  python -u evaluate_pixel_1model_quadrants.py --category $CAT --pt_template "$PT" --epochs 200 \
    --lr 0.1 --norm none > "$L" 2>&1
  echo "$CAT,$(val "$L" 'MERGE norm=none] Image-AUROC'),$(val "$L" 'MERGE norm=none] Pixel-AUROC')" >> "$CSV1"
  echo "done $CAT"
done
