#!/usr/bin/env python3
"""Continual figures: average AUROC vs task and forgetting matrix, averaged over seeds.

Reads the JSON files written by main_*_clad.py.
Example: python plot_continual.py --json_dir <dir> --tag stfpm --out_dir <dir>
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATS = ["distilled_replay", "real_replay", "replay", "FT", "MT"]  # longest-first
LABEL = {"FT": "Fine-tuning", "MT": "Joint (ref.)", "real_replay": "Real replay",
         "distilled_replay": "Distilled replay", "replay": "Replay (large mem.)"}
PRETTY = {"img_roc_auc": "Image-AUROC", "pxl_roc_auc": "Pixel-AUROC",
          "img_f1": "Image-F1", "pxl_f1": "Pixel-F1", "pxl_au_pro": "Pixel-AUPRO",
          "img_avg_prec": "Image-AP", "pxl_avg_prec": "Pixel-AP"}
# palette accessibile (Okabe-Ito)
COLOR = {"FT": "#999999", "MT": "#000000", "real_replay": "#0072B2",
         "distilled_replay": "#D55E00", "replay": "#009E73"}
ORDER = ["FT", "real_replay", "distilled_replay", "MT"]


def strat_of(fname):
    base = os.path.basename(fname)
    for s in STRATS:
        if base.startswith(s + "_"):
            return s
    return None


def load_matrix(d, metric):
    m = d.get("matrices", {}).get(metric)
    if m is None:
        return None
    return np.array([[np.nan if v is None else float(v) for v in row] for row in m], dtype=float)


def collect(json_dir, metric, exclude=None):
    """ritorna {strat: {'mats':[TxT...], 'final':[...]}} aggregato sui seed."""
    exclude = exclude or []
    out = {}
    for f in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        base = os.path.basename(f)
        if any(x in base for x in exclude):
            continue
        s = strat_of(f)
        if s is None:
            continue
        d = json.load(open(f))
        out.setdefault(s, {"mats": [], "final": [], "cats": d.get("categories")})
        M = load_matrix(d, metric)
        if M is not None and M.size:
            out[s]["mats"].append(M)
        fa = d.get("final_average", {}).get(metric)
        if fa is not None:
            out[s]["final"].append(float(fa))
    return out


def running_avg_per_task(M):
    """media sui task visti (0..i) dopo il task i -> vettore lungo T."""
    T = M.shape[0]
    return np.array([np.nanmean(M[i, :i + 1]) for i in range(T)])


def fig_auroc_vs_task(agg, metric, tag, out_dir):
    plt.figure(figsize=(7, 4.3))
    T = None
    for s in ORDER:
        if s not in agg:
            continue
        mats = agg[s]["mats"]
        if mats:
            T = mats[0].shape[0]
            curves = np.stack([running_avg_per_task(M) for M in mats])  # [seeds, T]
            mean, std = np.nanmean(curves, 0), np.nanstd(curves, 0)
            x = np.arange(1, T + 1)
            if s == "MT":
                continue  # joint (MT) is drawn as a horizontal line
            plt.plot(x, mean, "-o", ms=4, color=COLOR[s], label=LABEL[s])
            plt.fill_between(x, mean - std, mean + std, color=COLOR[s], alpha=0.15)
    # joint (MT) reference line
    if "MT" in agg and agg["MT"]["final"]:
        mt = float(np.mean(agg["MT"]["final"]))
        plt.axhline(mt, ls="--", color=COLOR["MT"], lw=1, label=f"{LABEL['MT']} = {mt:.3f}")
    pm = PRETTY.get(metric, metric)
    plt.xlabel("Seen tasks (current task index)")
    plt.ylabel(f"Mean {pm} over seen tasks")
    plt.title(f"Continual VAD — {tag.upper()} — {pm}")
    if T:
        plt.xticks(range(1, T + 1))
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    p = os.path.join(out_dir, f"continual_auroc_vs_task_{tag}_{metric}.png")
    plt.savefig(p, dpi=160); plt.close()
    return p


def fig_forgetting_matrix(agg, metric, tag, out_dir, strat="distilled_replay"):
    if strat not in agg or not agg[strat]["mats"]:
        return None
    M = np.nanmean(np.stack(agg[strat]["mats"]), 0)   # media seed
    cats = agg[strat]["cats"] or [str(i) for i in range(M.shape[0])]
    T = M.shape[0]
    plt.figure(figsize=(6.6, 5.6))
    im = plt.imshow(M, cmap="viridis", vmin=np.nanmin(M), vmax=1.0, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=PRETTY.get(metric, metric))
    for i in range(T):
        for j in range(T):
            if not np.isnan(M[i, j]):
                plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                         fontsize=6.5, color="white" if M[i, j] < 0.75 else "black")
    plt.xticks(range(T), [c[:6] for c in cats], rotation=60, ha="right", fontsize=7)
    plt.yticks(range(T), [c[:6] for c in cats], fontsize=7)
    plt.xlabel("Evaluated on task $j$")
    plt.ylabel("After training on task $i$")
    plt.title(f"Forgetting matrix — {tag.upper()} — {LABEL.get(strat,strat)} — {PRETTY.get(metric,metric)}")
    plt.tight_layout()
    p = os.path.join(out_dir, f"continual_forgetting_{tag}_{strat}_{metric}.png")
    plt.savefig(p, dpi=160); plt.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--metrics", nargs="+", default=["img_roc_auc", "pxl_roc_auc"])
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--exclude", nargs="*", default=["_5ep_"], help="salta i JSON che contengono questi substring")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for metric in a.metrics:
        agg = collect(a.json_dir, metric, a.exclude)
        if not agg:
            print(f"[{metric}] nessun JSON con matrici in {a.json_dir}")
            continue
        strat_counts = {s: len(agg[s]["mats"]) for s in agg}
        print(f"[{metric}] strategie/seed trovati: {strat_counts}")
        p1 = fig_auroc_vs_task(agg, metric, a.tag, a.out_dir)
        print("  ->", p1)
        for strat in ("distilled_replay", "real_replay"):
            p2 = fig_forgetting_matrix(agg, metric, a.tag, a.out_dir, strat)
            if p2:
                print("  ->", p2)


if __name__ == "__main__":
    main()
