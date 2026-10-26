# Trajectory-matching dataset distillation for visual anomaly detection

Code, scripts and raw results to reproduce the experiments of the master thesis
*Dataset distillation for continual learning in visual anomaly detection*
(Mattia Scantamburlo, University of Padova, A.Y. 2025/2026).

Ten synthetic images per MVTec AD category are distilled with trajectory matching (DATM-style,
reformulated for feature matching) for three detectors (**STFPM**, **RD4AD** and **Dinomaly**)
and then used as training sets and as replay memories for continual anomaly detection.
A second objective, distribution matching in the feature space of the STFPM teacher, is also provided.

The work has two phases, and both can be reproduced:
- **main sweep** (thesis Ch. 4, scripts `01_`-`06_`): the original pipeline. Its image learning rate
  leaves the synthetic images practically equal to their real initialisation, so its "distilled" results
  measure ten real images;
- **final runs** (thesis Ch. 5 and 7, scripts `07_`-`09_`): the images are optimised in ZCA space with a
  first-order meta-gradient and Adam, against strong experts, and every run passes control checks
  (decreasing loss on fixed pairs, size and structure of the change) before any AUROC is read.

```
distillation/     distillation + single-class / multi-class / patch / resolution experiments
  lib/            detectors, losses, trajectory-matching helpers
  src/            entry points (experts, distillation, evaluation, checks, figures)
  scripts/        one script per experiment of the thesis (01_ ... 09_)
continual/        continual experiments, built on MoViAD (github.com/AMCO-UniPD/moviad)
  setup.sh        clones MoViAD at the thesis commit and applies the changes
  overlay/        modified and new MoViAD files (replay strategies, trainer)
  experiments/    main_{stfpm,rd4ad,dinomaly}_clad.py
  scripts/        01_ ... 03_
distilled_sets/   the distilled images used in the thesis (not in git, 817 MB: see below)
results/          raw per-run results + scripts that rebuild the thesis tables
```

## 1. Setup

**Hardware.** All runs used one NVIDIA RTX A6000 (48 GB). Distillation peaks at about 25 GB
(Dinomaly, RD4AD at 128); evaluation needs 1-2 GB.

**Dataset.** Download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place
it (or a symlink) at `distillation/data/mvtec_anomaly_detection/<category>/{train,test,ground_truth}`.

**Distillation environment** (Python 3.10):
```bash
cd distillation
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
The DINOv2 ViT-S/14 (reg4) checkpoint is downloaded automatically on first use
(`dl.fbaipublicfiles.com`, cached in the torch hub folder).

**Continual environment** (MoViAD, uses [uv](https://docs.astral.sh/uv/)):
```bash
cd continual && bash setup.sh        # clone MoViAD @ ed856ec, copy overlay/ and experiments/, uv sync
```

## 2. Reproducing the experiments

All distillation scripts run from anywhere and read the dataset from `distillation/data`, writing to
`distillation/outputs`. Common options: `GPU=<id>`, `N_RUNS=<n>` (default 5), `VENV=<path>`; a list of
categories after the arguments restricts the run (default: all 15).

| Step | Command | Thesis | Measured GPU time |
|---|---|---|---|
| Single-class | `bash scripts/01_single_class.sh stfpm` (and `rd4ad`, `dinomaly`) | Ch. 4, `macro_3det` and per-category tables | ~1.5 h per category and resolution (buffer, distillation, 10 evaluations) |
| Equal-budget baselines | `bash scripts/02_random_subset.sh` | Sec. 4.1, `table_random_subset` | ~1 min per evaluation, 15 per category |
| Multi-class | `bash scripts/03_multiclass.sh` | Sec. 4.5, `table_multiclass*` | -- |
| Patch-split | `bash scripts/04_patch_split.sh` | Sec. 4.9, `table_patch` | -- |
| Dinomaly at 64 | `bash scripts/05_dinomaly_at_64.sh` | Sec. 4.4 | -- |
| Resolution 224 | `bash scripts/06_resolution.sh` | `table_res224` | -- |
| Strong experts | `bash scripts/07_strong_experts.sh` (`RES=64 LR=0.4 EP=100 K=5`) | Sec. 2.6 | ~1 h per category at 64 |
| Trajectory matching, final | `bash scripts/08_trajectory_matching.sh` (inner-loop settings as env variables, see the script) | Ch. 5 | ~2 h per category and starting set at 64 |
| Distribution matching | `bash scripts/09_distribution_matching.sh val`, then `... test` | Ch. 7 | ~5 min per set + evaluations |
| Continual STFPM | `cd continual && bash scripts/01_stfpm.sh` | Ch. 6, `table_continual` | 1.3-1.7 h per run, 12 runs |
| Continual RD4AD / Dinomaly | `bash scripts/02_rd4ad_dinomaly.sh` | `table_continual_3det` | 0.5-0.9 h per run, 18 runs |
| Replay at equal memory | `bash scripts/03_memory.sh` | Sec. 6.3, `table_cl_memory` | ~1.5 h per run, 18 runs |

Scripts 02-06 and the continual distilled-replay runs need the distilled sets of script 01;
script 08 needs the experts of script 07. Scripts 08 and 09 share the starting sets
(`outputs/starts/<category>/{r0,r1,r2,core}.pt`), so the two objectives are compared on the same images.

**Skipping the distillation.** Distillation is the expensive part (expert buffers of 2-6 GB per
category, see thesis Tab. "cost"). The sets used in the thesis (`distilled_sets/`, 817 MB, distributed as a separate archive on the
Releases page of this repository because of its size) can be used instead: copy its
content into `distillation/outputs/` and scripts 01-06 skip buffer and distillation and only run the
evaluations (01 also needs `outputs/single_class/images/`, which it fills from `images_best.pt`).

## 3. Final runs: what the scripts check

Every final run is accepted or rejected by rules fixed before the runs (thesis Sec. 2.7):

| Check | Script | Rule |
|---|---|---|
| matched segment is worth matching | `expert_auroc.py` | snapshots better than ten real images |
| matching problem is solvable | `inner_diag.py` | loss of the ten starting images < 1, all real images clearly lower |
| objective decreases | `distill.py --eval_pairs 50` | loss on 50 fixed pairs starts < 1 and drops by >= 10% |
| images change | `check_set.py` | mean absolute change >= nearest-real distance of the category |
| change is structured | `check_set.py` | share of the change surviving a 3x3 blur >= 0.3 (pixel noise: 0.14) |
| no skipped update | `check_set.py` | no unchanged image (`--threshold 1000`) |
| selection | `select_checkpoint.py` | largest gain over the starting set on the validation half, 3 seeds |
| result | `paired_summary.py` | test half, 10 paired seeds, Wilcoxon test against the starting set |

The evaluation uses `evaluate_pixel.py --seed s --split val|test`: the seed fixes the student
initialisation and the batch order, so different training sets are compared run by run.

The final RD4AD and Dinomaly runs will be added with the same structure.

## 4. Protocol notes

- **No selection on the test set.** Distillation also stores `images_best.pt` (best test AUROC during
  distillation), but every single-class and continual result uses the **last iteration**
  (`images_300.pt`; `images_150.pt` for RD4AD *bottle* at 64). Only three studies use the
  best-iteration sets, as stated in the thesis: multi-class, patch-split and Dinomaly at 64.
- **Evaluation epochs:** STFPM 400 (SGD, lr 0.1), RD4AD 800, Dinomaly 50; 5 runs per setting.
  The equal-budget baselines use STFPM at 200 epochs.
- **Feature extractor.** The distillation code uses torchvision backbones (MoViAD's extractor is
  optional and was not used); the continual code uses MoViAD's own models.
- **Joint (multi-task) continual model.** The original runs scored it with one AUROC on the pooled
  test set of the 10 categories; `main_stfpm_clad.py` now also reports the per-task average, which is
  the value comparable with the other strategies.
- **RD4AD at 112** fails by construction (encoder 28x28 vs decoder 32x32 feature maps, thesis Appendix).
- **Non-determinism.** Evaluation runs are not bit-identical across machines; expect run-to-run
  differences of the size of the reported standard deviations.

## 5. Rebuilding the tables from the raw results

`results/raw/` holds one row (CSV) or one file (JSON) per run, exactly as produced for the thesis.
```bash
cd results
python build_single_class.py                                   # merge real + last-iteration rows
python make_tables.py --sweep merged --tables tables --data data   # single-class tables, identical to the thesis
python aggregate_continual.py                                  # continual tables (mean +- std over seeds)
```

| Raw file | Content | Thesis |
|---|---|---|
| `single_class/results_raw.clean.csv` | real-data rows (and old best-iteration rows, not used) | Ch. 4 tables |
| `single_class/results_lastckpt_f1*.csv` | distilled rows, last iteration, with F1-max | Ch. 4 tables |
| `random_subset/real10_initial10.csv` | Initial-10 / Distilled-10 / Real-10 | `table_random_subset` |
| `multiclass/*.csv` | multi-class STFPM, RD4AD, Dinomaly | `table_multiclass*` |
| `patch/*.csv` | patch-split, 4 models and 1 model | `table_patch` |
| `dino64/*.csv` | Dinomaly at 64 | Sec. 4.4 |
| `resolution/*.csv` | real at 224, RD4AD distilled at 112/224 | `table_res224` |
| `continual/*/*.json` | per-run continual results (task matrices, final average, forgetting) | Ch. 6 tables and figures |
