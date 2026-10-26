# Shared settings, sourced by every script in this folder.
# Dataset: distillation/data/mvtec_anomaly_detection   Outputs: distillation/outputs
# Override from the shell, e.g.  GPU=1 N_RUNS=2 bash scripts/01_single_class.sh stfpm

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"
DATA="$ROOT/data/mvtec_anomaly_detection"
OUT="$ROOT/outputs"
GPU="${GPU:-0}"
N_RUNS="${N_RUNS:-5}"          # evaluation runs per setting
IPC=10

ALL_CATS=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)

# Optional virtualenv
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi

[ -d "$DATA" ] || { echo "MVTec AD not found in $DATA"; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$SRC"   # the Python scripts use paths relative to src/ (../data, ../outputs)

# Last value printed after a label, e.g. "Image-level AUROC: 0.91"
val() { grep -i "$2" "$1" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1; }
# image AUROC, pixel AUROC, image F1max, pixel F1max
metrics() { echo "$(val "$1" 'Image-level AUROC'),$(val "$1" 'Pixel-level AUROC'),$(val "$1" 'Image-level F1max'),$(val "$1" 'Pixel-level F1max')"; }
# Last saved iteration of a distillation run (images_300.pt)
last_ckpt() { ls "$1"/images_*.pt 2>/dev/null | grep -E "images_[0-9]+\.pt$" | sort -V | tail -1; }
# Create a CSV with its header if missing
csv_init() { [ -f "$1" ] || echo "$2" > "$1"; }

# Paired evaluation of STFPM (thesis Sec. 2.8): every set is trained with the same seeds 1..N on one
# half of the test set; "auto" draws ten new random real images per seed (Real-10).
# Usage: paired_eval <val|test> <n_seeds> <category> <out_dir> name=set.pt [name=set.pt ...]
paired_eval() {
  local split=$1 n=$2 cat=$3 dir=$4; shift 4
  mkdir -p "$dir/logs" "$dir/sets"
  csv_init "$dir/paired.csv" "category,set,seed,image_auroc,pixel_auroc"
  for s in $(seq 1 "$n"); do for spec in "$@"; do
    local name=${spec%%=*} set=${spec#*=}
    grep -q "^$cat,$name,$s," "$dir/paired.csv" && continue
    if [ "$set" = auto ]; then
      set=$dir/sets/real10_s$s.pt
      [ -f "$set" ] || python make_real_subset.py --src "$DATA/$cat/train/good" --n 10 --res ${RES:-64} --seed $s --out "$set" > /dev/null
    fi
    local L=$dir/logs/${name}_s$s.log
    python -u evaluate_pixel.py --category $cat --img_size ${RES:-64} --epochs 200 --lr 0.1 --train_dir "$set" \
      --split $split --seed $s > "$L" 2>&1
    echo "$cat,$name,$s,$(val "$L" 'Image-level AUROC'),$(val "$L" 'Pixel-level AUROC')" >> "$dir/paired.csv"
  done; done
}

# Starting sets shared by trajectory and distribution matching (thesis Sec. 2.8):
# r0 = the ten images distill.py draws with seed 0, r1/r2 = further random draws, core = coverage coreset.
make_starts() {  # category res
  local d=$OUT/starts/$1; mkdir -p "$d"
  [ -f "$d/r0.pt" ] || python make_real_subset.py --src "$DATA/$1/train/good" --n 10 --res $2 --seed 0 --like_distill --out "$d/r0.pt" > /dev/null
  [ -f "$d/r1.pt" ] || python make_real_subset.py --src "$DATA/$1/train/good" --n 10 --res $2 --seed 1001 --out "$d/r1.pt" > /dev/null
  [ -f "$d/r2.pt" ] || python make_real_subset.py --src "$DATA/$1/train/good" --n 10 --res $2 --seed 1002 --out "$d/r2.pt" > /dev/null
  [ -f "$d/core.pt" ] || python coreset.py --category $1 --res $2 --out "$d/core.pt" > /dev/null
  echo "$d"
}
