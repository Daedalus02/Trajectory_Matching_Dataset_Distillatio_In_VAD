#!/usr/bin/env bash
# Main single-class experiment (thesis Ch. 4): for each category and resolution
#   expert buffer -> distillation (300 it.) -> 5 evaluations on the distilled set
#   (last iteration, no test selection) and 5 on the full real training set.
# Usage: bash scripts/01_single_class.sh <stfpm|rd4ad|dinomaly> [cat ...]
# Output: outputs/single_class/results.csv
set -uo pipefail
source "$(dirname "$0")/config.sh"

DET=$1; shift
CATS=("$@"); [ ${#CATS[@]} -eq 0 ] && CATS=("${ALL_CATS[@]}")

case "$DET" in
  stfpm)    RESES=(64 128); BUFGEN=generate_buffer.py;          DISTILL=distill.py;          EVAL=evaluate_pixel.py
            DDIR=distilled;          EP=400; EVAL_EXTRA="--lr 0.1"; BUF_EXTRA=""
            DIST_EXTRA="--batch_syn $IPC --eval_it 100 --eval_epochs 100" ;;
  rd4ad)    RESES=(64 128); BUFGEN=generate_buffer_rd4ad.py;    DISTILL=distill_rd4ad.py;    EVAL=evaluate_pixel_rd4ad.py
            DDIR=distilled_rd4ad;    EP=800; EVAL_EXTRA="";         BUF_EXTRA="--half_buffer"; DIST_EXTRA="--pix_init real --offline" ;;
  dinomaly) RESES=(224);    BUFGEN=generate_buffer_dinomaly.py; DISTILL=distill_dinomaly.py; EVAL=evaluate_pixel_dinomaly.py
            DDIR=distilled_dinomaly; EP=50;  EVAL_EXTRA="";         BUF_EXTRA="--half_buffer"; DIST_EXTRA="--pix_init real --offline" ;;
  *) echo "unknown detector: $DET"; exit 1 ;;
esac
RESES=(${RES_LIST:-${RESES[@]}})   # optional override, e.g. RES_LIST=64

LOG=$OUT/single_class/logs; mkdir -p "$LOG" "$OUT/single_class/images"
CSV=$OUT/single_class/results.csv
csv_init "$CSV" "category,detector,res,type,run,image_auroc,pixel_auroc,image_f1max,pixel_f1max,ckpt"

for CAT in "${CATS[@]}"; do
  for RES in "${RESES[@]}"; do
    TAG=${DET}_${CAT}_${RES}
    BUF=$OUT/buffer_${TAG}
    DIR=$OUT/$DDIR/$CAT/$IPC/real/full$RES

    # skipped if the distilled set already exists (e.g. copied from distilled_sets/)
    if [ -z "$(last_ckpt "$DIR")" ]; then
      # 1) expert trajectories: 10 experts x 20 epochs
      [ -d "$BUF" ] || python -u $BUFGEN --category $CAT --num_experts 10 --train_epochs 20 \
        --optimizer sgd --res $RES $BUF_EXTRA --buffer_path "$BUF" > "$LOG/buf_$TAG.log" 2>&1
      # 2) distillation: IPC=10, 300 iterations, real-image init
      python -u $DISTILL --category $CAT --ipc $IPC $DIST_EXTRA \
        --Iteration 300 --syn_steps 15 --expert_epochs 2 --lr_img 0.01 --max_start_epoch 5 \
        --res $RES --buffer_path "$BUF" > "$LOG/dist_$TAG.log" 2>&1
    fi

    LAST=$(last_ckpt "$DIR")
    [ -n "$LAST" ] || { echo "!! no distilled set for $TAG"; continue; }
    # best-iteration copy, used only by the multi-class study (as in the thesis)
    cp "$DIR/images_best.pt" "$OUT/single_class/images/$TAG.pt" 2>/dev/null

    # 3) evaluation: detector trained from scratch, N_RUNS times
    for R in $(seq 1 $N_RUNS); do
      L=$LOG/ev_${TAG}_dist_r$R.log
      python -u $EVAL --category $CAT --img_size $RES --epochs $EP $EVAL_EXTRA --train_dir "$LAST" > "$L" 2>&1
      echo "$CAT,${DET^^},$RES,distilled,$R,$(metrics "$L"),$(basename "$LAST")" >> "$CSV"

      L=$LOG/ev_${TAG}_real_r$R.log
      python -u $EVAL --category $CAT --img_size $RES --epochs $EP $EVAL_EXTRA > "$L" 2>&1
      echo "$CAT,${DET^^},$RES,real,$R,$(metrics "$L"),full" >> "$CSV"
    done
    echo "done $TAG"
  done
done
