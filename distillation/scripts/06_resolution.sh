#!/usr/bin/env bash
# Resolution studies (thesis Tab. "res224"):
#   - STFPM and RD4AD trained on all real images at 224
#   - RD4AD distilled at 224 (and 112, which fails by construction, see thesis Appendix A)
# Usage: bash scripts/06_resolution.sh [cat ...]
# Output: outputs/resolution/real224.csv, outputs/resolution/rd4ad_distilled.csv
set -uo pipefail
source "$(dirname "$0")/config.sh"

CATS=("$@"); [ ${#CATS[@]} -eq 0 ] && CATS=("${ALL_CATS[@]}")
D=$OUT/resolution; LOG=$D/logs; mkdir -p "$LOG"
csv_init "$D/real224.csv" "category,detector,image_auroc,pixel_auroc,image_f1max,pixel_f1max"
csv_init "$D/rd4ad_distilled.csv" "category,res,run,image_auroc,pixel_auroc,image_f1max,pixel_f1max"

for CAT in "${CATS[@]}"; do
  # real baselines at 224
  L=$LOG/real_stfpm_$CAT.log
  python -u evaluate_pixel.py --category $CAT --img_size 224 --epochs 400 --lr 0.1 > "$L" 2>&1
  echo "$CAT,STFPM,$(metrics "$L")" >> "$D/real224.csv"
  L=$LOG/real_rd4ad_$CAT.log
  python -u evaluate_pixel_rd4ad.py --category $CAT --img_size 224 --epochs 800 > "$L" 2>&1
  echo "$CAT,RD4AD,$(metrics "$L")" >> "$D/real224.csv"

  # RD4AD distilled at 112 and 224 (2 runs, as in the thesis)
  for RES in 112 224; do
    BUF=$OUT/buffer_rd4ad_${CAT}_$RES
    DIR=$OUT/distilled_rd4ad/$CAT/$IPC/real/full$RES
    if [ -z "$(last_ckpt "$DIR")" ]; then
      [ -d "$BUF" ] || python -u generate_buffer_rd4ad.py --category $CAT --num_experts 10 --train_epochs 20 \
        --optimizer sgd --res $RES --half_buffer --buffer_path "$BUF" > "$LOG/buf_${CAT}_$RES.log" 2>&1
      python -u distill_rd4ad.py --category $CAT --ipc $IPC --Iteration 300 --syn_steps 15 --expert_epochs 2 \
        --lr_img 0.01 --max_start_epoch 5 --res $RES --buffer_path "$BUF" --pix_init real --offline \
        > "$LOG/dist_${CAT}_$RES.log" 2>&1
    fi
    LAST=$(last_ckpt "$DIR")
    [ -n "$LAST" ] || { echo "$CAT,$RES,NA,NA,NA,NA,NA" >> "$D/rd4ad_distilled.csv"; continue; }   # 112 fails here
    for R in 1 2; do
      L=$LOG/ev_${CAT}_${RES}_r$R.log
      python -u evaluate_pixel_rd4ad.py --category $CAT --img_size $RES --epochs 800 --train_dir "$LAST" > "$L" 2>&1
      echo "$CAT,$RES,$R,$(metrics "$L")" >> "$D/rd4ad_distilled.csv"
    done
  done
done
