#!/usr/bin/env bash
# Final trajectory-matching runs for STFPM (thesis Ch. 5). For each category and starting set:
#   1) distillation against the strong experts of 07_strong_experts.sh: ZCA space, first-order
#      meta-gradient, Adam with cosine decay, no skipped updates, loss on 50 fixed pairs every 100 it.,
#      checkpoints at 500/1000/1500;
#   2) control checks of every checkpoint (check_set.py);
#   3) validation half, 3 paired seeds: starting set and every checkpoint;
#   4) selection (select_checkpoint.py) and test half, 10 paired seeds: starting set, selected
#      checkpoint, Real-10, coreset, all real images.
# Inner loop: snapshots LO..HI-1 as start, EE snapshots ahead, ST steps at lr LRT, NP pairs per iteration.
# Usage: [RES=64 NP=2 LRT=0.4 ST=40 EE=2 LO=10 HI=18 STARTS="r0 r1 core"] bash scripts/08_trajectory_matching.sh [cat ...]
# Output: outputs/tm/<cat>/{checks.csv,val/paired.csv,test/paired.csv,selection.txt}
set -uo pipefail
source "$(dirname "$0")/config.sh"

CATS=("$@"); [ ${#CATS[@]} -eq 0 ] && CATS=("${ALL_CATS[@]}")
RES=${RES:-64}; NP=${NP:-2}; LRT=${LRT:-0.4}; ST=${ST:-40}; EE=${EE:-2}; LO=${LO:-10}; HI=${HI:-18}
STARTS=${STARTS:-r0 r1 core}

for CAT in "${CATS[@]}"; do
  BUF=$OUT/buffer_s${RES}strong_$CAT
  [ -d "$BUF" ] || { echo "skip $CAT (run 07_strong_experts.sh first)"; continue; }
  S=$(make_starts $CAT $RES)
  W=$OUT/tm/$CAT; mkdir -p "$W/logs"
  csv_init "$W/checks.csv" "category,tag,it,mad255,band_lo,band_hi,struct,unchanged,fixed0,fixed_it,ratio"
  VAL_SPECS=""
  for I in $STARTS; do
    DIR=$OUT/distilled/$CAT/$IPC/real/full${RES}_tm_$I
    [ -f "$DIR/images_1500.pt" ] || python -u distill.py --category $CAT --res $RES --ipc $IPC --batch_syn $IPC \
      --init_path "$S/$I.pt" --buffer_path "$BUF" --zca --zca_eps 0.005 --fo --opt_img adam --lr_img 3e-3 --img_cos \
      --n_pairs $NP --min_start_epoch $LO --max_start_epoch $HI --expert_epochs $EE --lr_teacher $LRT --syn_steps $ST \
      --threshold 1000 --Iteration 1500 --eval_it 500 --eval_epochs 0 --eval_pairs 50 --fixed_eval_every 100 \
      --save_tag _tm_$I > "$W/logs/distill_$I.log" 2>&1
    grep -q "^$CAT,$I," "$W/checks.csv" || python check_set.py --category $CAT --run_dir "$DIR" --res $RES --tag $I >> "$W/checks.csv"
    VAL_SPECS="$VAL_SPECS $I=$S/$I.pt"
    for IT in 500 1000 1500; do VAL_SPECS="$VAL_SPECS ${I}_it$IT=$DIR/images_$IT.pt"; done
  done
  paired_eval val 3 $CAT "$W/val" $VAL_SPECS

  TEST_SPECS="real10=auto core=$S/core.pt all_real=$DATA/$CAT/train/good"
  : > "$W/selection.txt"
  for I in $STARTS; do
    SEL=$(python select_checkpoint.py --checks "$W/checks.csv" --val "$W/val/paired.csv" --tag $I)
    echo "$I $SEL" >> "$W/selection.txt"
    TEST_SPECS="$TEST_SPECS $I=$S/$I.pt"
    [ "$SEL" = none ] || TEST_SPECS="$TEST_SPECS tm_$I=$OUT/distilled/$CAT/$IPC/real/full${RES}_tm_$I/images_${SEL##*_it}.pt"
  done
  paired_eval test 10 $CAT "$W/test" $TEST_SPECS
  echo "done $CAT"
done
