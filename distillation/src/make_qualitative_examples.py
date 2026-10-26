#!/usr/bin/env python3
"""Figure of MVTec AD examples: one row per category, [good | anomalous | anomalous + mask].
"""
import os, glob, argparse, numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def first_good(cat, root):
    fs = sorted(glob.glob(os.path.join(root, cat, "train", "good", "*.png")))
    return fs[0] if fs else None

def first_anomaly(cat, root):
    for d in sorted(glob.glob(os.path.join(root, cat, "test", "*"))):
        defect = os.path.basename(d)
        if defect == "good": continue
        for f in sorted(glob.glob(os.path.join(d, "*.png"))):
            base = os.path.splitext(os.path.basename(f))[0]
            gt = os.path.join(root, cat, "ground_truth", defect, base + "_mask.png")
            if os.path.exists(gt): return f, gt, defect
    return None, None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["bottle", "carpet", "metal_nut", "screw"])
    ap.add_argument("--data_root", default="../data/mvtec_anomaly_detection")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="../outputs/mvtec_examples.png")
    args = ap.parse_args()

    cats = args.categories; S = args.size
    fig, ax = plt.subplots(len(cats), 3, figsize=(8, 2.7 * len(cats)))
    if len(cats) == 1: ax = ax[None, :]
    for r, cat in enumerate(cats):
        fg = first_good(cat, args.data_root)
        fa, fgt, defect = first_anomaly(cat, args.data_root)
        good = Image.open(fg).convert("RGB").resize((S, S)) if fg else Image.new("RGB", (S, S))
        anom = Image.open(fa).convert("RGB").resize((S, S)) if fa else Image.new("RGB", (S, S))
        mask = (np.asarray(Image.open(fgt).convert("L").resize((S, S), Image.NEAREST)) > 0) if fgt else np.zeros((S, S), bool)
        ax[r, 0].imshow(good); ax[r, 0].set_ylabel(cat, fontsize=11)
        ax[r, 1].imshow(anom)
        ax[r, 2].imshow(anom)
        overlay = np.zeros((S, S, 4)); overlay[mask] = [1, 0, 0, 0.45]   # rosso semitrasparente
        ax[r, 2].imshow(overlay)
        if r == 0:
            ax[r, 0].set_title("good"); ax[r, 1].set_title("anomalous"); ax[r, 2].set_title("GT overlay")
        for c in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, 1].set_xlabel(defect or "", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight"); plt.close()
    print("salvato", args.out)

if __name__ == "__main__":
    main()
