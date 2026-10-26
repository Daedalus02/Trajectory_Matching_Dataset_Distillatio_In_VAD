# Shared settings for the continual experiments (run setup.sh first).
# Override from the shell, e.g.  GPU=1 bash scripts/01_stfpm.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # continual/
DIST="$(cd "$HERE/../distillation" && pwd)"                 # distillation/ (dataset + distilled sets)
DATA="$DIST/data/mvtec_anomaly_detection"
GPU="${GPU:-0}"
SEEDS=(${SEEDS:-0 1 2})
OUT="$HERE/results"; LOG="$HERE/logs"; mkdir -p "$OUT" "$LOG"

cd "$HERE/moviad" || { echo "run setup.sh first"; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=src WANDB_MODE=offline

# run <script> <out_subdir> <log_name> <args...>
run() {
  local script=$1 sub=$2 name=$3; shift 3
  mkdir -p "$OUT/$sub"
  uv run python experiments/$script --device 0 --dataset_path "$DATA" --out_dir "$OUT/$sub" "$@" \
    > "$LOG/$name.log" 2>&1 && echo "ok   $name" || echo "FAIL $name (see logs/$name.log)"
}
