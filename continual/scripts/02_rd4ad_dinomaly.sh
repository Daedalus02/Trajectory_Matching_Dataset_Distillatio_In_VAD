#!/usr/bin/env bash
# Continual RD4AD (WideResNet50-2) and Dinomaly (ViT-S): fine-tuning, real replay and
# distilled replay (last-iteration sets), 3 seeds, 30 epochs per task.
# Needs distillation/scripts/01_single_class.sh rd4ad and dinomaly.
# Usage: bash scripts/02_rd4ad_dinomaly.sh
set -uo pipefail
source "$(dirname "$0")/config.sh"

R="--backbone wide_resnet50_2 --epochs 30 --batch_size 16"
D="--epochs 30 --batch_size 16"
for s in "${SEEDS[@]}"; do
  run main_rd4ad_clad.py rd4ad FT_s$s        --model FT_rd4ad $R --seed $s
  run main_rd4ad_clad.py rd4ad real_s$s      --model real_replay_rd4ad --mem_per_task 10 $R --seed $s
  run main_rd4ad_clad.py rd4ad distilled_s$s --model distilled_replay_rd4ad --mem_per_task 10 $R --seed $s \
      --distilled_dir "$DIST/outputs/distilled_rd4ad" --distilled_template "{cat}/10/real/full128/images_300.pt"

  run main_dinomaly_clad.py dinomaly FT_s$s        --model FT_dino $D --seed $s
  run main_dinomaly_clad.py dinomaly real_s$s      --model real_replay_dino --mem_per_task 10 $D --seed $s
  run main_dinomaly_clad.py dinomaly distilled_s$s --model distilled_replay_dino --mem_per_task 10 $D --seed $s \
      --distilled_dir "$DIST/outputs/distilled_dinomaly" --distilled_template "{cat}/10/real/full224/images_300.pt"
done
