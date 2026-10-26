#!/usr/bin/env python3
"""Paired comparison of training sets evaluated with the same seeds (thesis Sec. 2.8).
For each paired.csv (columns category,set,seed,image_auroc,...): mean AUROC of every set and, against
the reference set, mean paired difference, number of seeds in which the set wins, Wilcoxon p-value.
Usage: python paired_summary.py [--ref initial] paired.csv [...]
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--ref", default="initial", help="reference set name")
args = ap.parse_args()

for f in args.files:
    auc, cat = defaultdict(dict), None
    for r in csv.DictReader(open(f)):
        cat = r["category"]
        if r["image_auroc"]:
            auc[r["set"]][int(r["seed"])] = float(r["image_auroc"])
    print("== %s (%s)" % (cat, f))
    ref = auc.get(args.ref, {})
    for name, v in auc.items():
        line = "  %-22s n=%2d  AUROC %.3f +- %.3f" % (name, len(v), np.mean(list(v.values())), np.std(list(v.values())))
        seeds = sorted(set(v) & set(ref))
        if name != args.ref and len(seeds) >= 3:
            d = np.array([v[s] - ref[s] for s in seeds])
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            line += "   vs %s: %+.3f  wins %d/%d  p=%.3f" % (args.ref, d.mean(), (d > 0).sum(), len(d), p)
        print(line)
