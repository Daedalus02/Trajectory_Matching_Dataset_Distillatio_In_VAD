#!/usr/bin/env python3
"""Selection rule of distribution matching, fixed before the runs (thesis Sec. 7.4): the coverage weight
with the largest mean validation gain over its own starting set, pooled over all categories, starting
sets and seeds. Reads <root>/<cat>/val/paired.csv (sets cov<w>_<start> and <start>); prints the
gain of every weight and, on the last line, the selected weight.
Usage: python select_dm.py --root ../outputs/dm
"""

import argparse
import csv
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
args = ap.parse_args()

auc = {}
for f in glob.glob(os.path.join(args.root, "*", "val", "paired.csv")):
    for r in csv.DictReader(open(f)):
        if r["image_auroc"]:
            auc[(r["category"], r["set"], r["seed"])] = float(r["image_auroc"])

gains = {}
for (cat, name, seed), v in auc.items():
    if name.startswith("cov"):
        weight, start = name[3:].split("_", 1)
        ref = auc.get((cat, start, seed))
        if ref is not None:
            gains.setdefault(weight, []).append(v - ref)
if not gains:
    raise SystemExit("no validation results in " + args.root)
for w, g in sorted(gains.items()):
    print("cov weight %s: mean validation gain %+.4f (n=%d, wins %d)" % (w, sum(g) / len(g), len(g), sum(x > 0 for x in g)))
print(max(gains, key=lambda w: sum(gains[w]) / len(gains[w])))
