#!/usr/bin/env bash
# Continual STFPM (thesis Ch. 6): fine-tuning, real replay, distilled replay (last-iteration
# sets at 128) and the joint model, 3 seeds, 20 epochs per task.
# Needs distillation/scripts/01_single_class.sh stfpm.   Usage: bash scripts/01_stfpm.sh
set -uo pipefail
source "$(dirname "$0")/config.sh"

A="--backbone resnet18 --epochs 20 --batch_size 8"
for s in "${SEEDS[@]}"; do
  run main_stfpm_clad.py stfpm FT_s$s          --model FT_stfpm $A --seed $s
  run main_stfpm_clad.py stfpm real_s$s        --model real_replay_stfpm --mem_per_task 10 $A --seed $s
  run main_stfpm_clad.py stfpm distilled_s$s   --model distilled_replay_stfpm --mem_per_task 10 $A --seed $s \
      --distilled_dir "$DIST/outputs/distilled" --distilled_template "{cat}/10/real/full128/images_300.pt"
  run main_stfpm_clad.py stfpm joint_s$s       --model MT_stfpm $A --seed $s
done
