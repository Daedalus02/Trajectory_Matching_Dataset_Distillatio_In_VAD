#!/usr/bin/env bash
# Replay at equal memory (thesis Sec. 6.3, STFPM): real replay with 1, 2, 3, 5, 20 images
# per task, and 10 real images per task stored at 128x128 (same footprint as the distilled buffer).
# Usage: bash scripts/03_memory.sh
set -uo pipefail
source "$(dirname "$0")/config.sh"

A="--backbone resnet18 --epochs 20 --batch_size 8"
for s in "${SEEDS[@]}"; do
  for M in 1 2 3 5 20; do
    run main_stfpm_clad.py memory real_m${M}_s$s --model real_replay_stfpm --mem_per_task $M $A --seed $s
  done
  run main_stfpm_clad.py memory lowres128_s$s --model lowres_replay_stfpm --mem_per_task 10 --store_res 128 $A --seed $s
done
