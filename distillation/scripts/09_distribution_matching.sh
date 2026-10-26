#!/usr/bin/env bash
# Distribution matching in the STFPM teacher feature space (thesis Ch. 7), at 64x64.
#   stage val : for each category and starting set, MMD with coverage weight 0, 1 and 3 (2000 it.),
#               then the validation half with 3 paired seeds;
#   stage test: the coverage weight with the largest mean validation gain over the starting set,
#               pooled over all categories, starting sets and seeds (select_dm.py), on the test half
#               with 10 paired seeds against the starting set, Real-10, coreset and all real images.
# Usage: bash scripts/09_distribution_matching.sh <val|test> [cat ...]   (run val on all categories first)
# Output: outputs/dm/<cat>/{val,test}/paired.csv, outputs/dm/selection.txt
set -uo pipefail
source "$(dirname "$0")/config.sh"

STAGE=$1; shift
CATS=("$@"); [ ${#CATS[@]} -eq 0 ] && CATS=("${ALL_CATS[@]}")
RES=64; STARTS=${STARTS:-r0 r1 r2 core}
[ "$STAGE" = test ] && { python select_dm.py --root "$OUT/dm" > "$OUT/dm/selection.txt" || exit 1; }

for CAT in "${CATS[@]}"; do
  S=$(make_starts $CAT $RES)
  W=$OUT/dm/$CAT; mkdir -p "$W/logs"
  SPECS=""
  for I in $STARTS; do
    SPECS="$SPECS $I=$S/$I.pt"
    if [ "$STAGE" = val ]; then
      for C in 0 1 3; do
        N=dm_cov${C}_$I
        [ -f "$OUT/distilled/$CAT/$IPC/real/$N/images_2000.pt" ] || python -u distill_dm.py --category $CAT --res $RES \
          --init "$S/$I.pt" --cov $C --name $N > "$W/logs/$N.log" 2>&1
        SPECS="$SPECS cov${C}_$I=$OUT/distilled/$CAT/$IPC/real/$N/images_2000.pt"
      done
    else
      C=$(tail -1 "$OUT/dm/selection.txt")
      SPECS="$SPECS dm_$I=$OUT/distilled/$CAT/$IPC/real/dm_cov${C}_$I/images_2000.pt"
    fi
  done
  if [ "$STAGE" = val ]; then
    paired_eval val 3 $CAT "$W/val" $SPECS
  else
    paired_eval test 10 $CAT "$W/test" $SPECS real10=auto all_real=$DATA/$CAT/train/good
  fi
  echo "done $STAGE $CAT"
done
