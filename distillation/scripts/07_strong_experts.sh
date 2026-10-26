#!/usr/bin/env bash
# Strong STFPM experts for the final runs (thesis Sec. 2.6): plain SGD (no momentum, no weight decay,
# constant lr), so that the parameter displacement depends on the data only; 10 experts, a snapshot every
# K epochs; then the AUROC of the snapshots of the first 3 experts (expert_auroc.py).
# Usage: [RES=64 LR=0.4 EP=100 K=5] bash scripts/07_strong_experts.sh [cat ...]
# Output: outputs/buffer_s<RES>strong_<cat>/, outputs/experts/snapshots_<RES>.csv
set -uo pipefail
source "$(dirname "$0")/config.sh"

CATS=("$@"); [ ${#CATS[@]} -eq 0 ] && CATS=("${ALL_CATS[@]}")
RES=${RES:-64}; LR=${LR:-0.4}; EP=${EP:-100}; K=${K:-5}
LOG=$OUT/experts/logs; mkdir -p "$LOG"

for CAT in "${CATS[@]}"; do
  BUF=$OUT/buffer_s${RES}strong_$CAT
  [ -f "$BUF/MVTec_feature/ResNet18/replay_buffer_0.pt" ] || python -u generate_buffer.py --category $CAT \
    --num_experts 10 --train_epochs $EP --snap_every $K --optimizer sgd --lr_teacher $LR --mom 0 --l2 0 \
    --res $RES --buffer_path "$BUF" > "$LOG/buf_${CAT}_$RES.log" 2>&1
  python -u expert_auroc.py --category $CAT --buffer_path "$BUF" --res $RES \
    --out "$OUT/experts/snapshots_$RES.csv" > "$LOG/auroc_${CAT}_$RES.log" 2>&1
  echo "done $CAT"
done
