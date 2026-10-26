#!/usr/bin/env python3
"""Control checks of a distilled run, computed before any AUROC (thesis Sec. 2.7).

For every checkpoint images_<it>.pt of a run folder, against images_0.pt:
  MAD       mean absolute change, in units of 1/255
  band      [mean distance of each real image to its nearest real image, median real-real distance]
  struct    structured share: energy of the change surviving a 3x3 binomial blur (pixel noise ~0.14)
  unchanged number of images that did not change at all (skipped updates)
  fixed     loss on the fixed pairs at iteration 0 and at the checkpoint (fixed_eval.csv), and their ratio
One CSV row per checkpoint on stdout.
Usage: python check_set.py --category screw --run_dir ../outputs/distilled/screw/10/real/full64_x [--res 64]
"""

import argparse
import csv
import glob
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.feature_matching import load_images_tensor


def blur(x):
    k = torch.tensor([1., 2., 1.])
    k = (k[:, None] * k[None, :]) / 16
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k.expand(3, 1, 3, 3), groups=3)


def real_band(real):
    d = torch.cdist(real.flatten(1), real.flatten(1), p=1) / real[0].numel() * 255
    d.fill_diagonal_(float("inf"))
    lo = d.min(1).values.mean().item()
    d.fill_diagonal_(float("nan"))
    return lo, d.nanmedian().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    ap.add_argument("--tag", default="", help="label written in the second column")
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()

    real = load_images_tensor(os.path.join(args.data_path, args.category, "train", "good"), args.res, verbose=False).float()
    lo, hi = real_band(real)

    fixed = []
    p = os.path.join(args.run_dir, "fixed_eval.csv")
    if os.path.exists(p):
        fixed = [(int(r["iter"]), float(r["fixed_loss"])) for r in csv.DictReader(open(p))]

    if args.header:
        print("category,tag,it,mad255,band_lo,band_hi,struct,unchanged,fixed0,fixed_it,ratio")
    x0 = torch.load(os.path.join(args.run_dir, "images_0.pt")).float()
    its = sorted(int(re.findall(r"images_(\d+)\.pt$", f)[0]) for f in glob.glob(os.path.join(args.run_dir, "images_*.pt"))
                 if re.search(r"images_\d+\.pt$", f))
    for it in [i for i in its if i > 0]:
        d = torch.load(os.path.join(args.run_dir, "images_{}.pt".format(it))).float() - x0
        share = (blur(d).pow(2).sum() / d.pow(2).sum().clamp_min(1e-12)).item()
        unchanged = int((d.flatten(1).abs().max(1).values == 0).sum())
        f0 = fixed[0][1] if fixed else float("nan")
        upto = [v for i, v in fixed if i <= it]
        f1 = upto[-1] if upto else float("nan")
        print("%s,%s,%d,%.2f,%.2f,%.2f,%.3f,%d,%.4f,%.4f,%.3f" % (args.category, args.tag, it, d.abs().mean() * 255,
                                                               lo, hi, share, unchanged, f0, f1, f1 / f0))


if __name__ == "__main__":
    main()
