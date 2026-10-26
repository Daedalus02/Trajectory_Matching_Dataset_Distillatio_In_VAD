#!/usr/bin/env python3
"""Selection rule of the final runs, fixed before the runs (thesis Sec. 2.7).

Among the checkpoints of one run that pass the control checks (check_set.py: fixed-pair loss starting
below 1 and lower by at least --min_drop, MAD at least the lower end of the band, structured share at
least --min_struct, no unchanged image), pick the one with the largest mean gain over its own starting
set on the validation half. Prints the set name of the chosen checkpoint, or "none".
Usage: python select_checkpoint.py --checks checks.csv --val paired.csv --tag r0
       (the validation sets are named <tag> for the start and <tag>_it<it> for the checkpoints)
"""

import argparse
import csv
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--checks", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--min_drop", type=float, default=0.10)
ap.add_argument("--min_struct", type=float, default=0.30)
args = ap.parse_args()

ok = []
for r in csv.DictReader(open(args.checks)):
    if r["tag"] != args.tag:
        continue
    passed = (float(r["fixed0"]) < 1 and float(r["ratio"]) <= 1 - args.min_drop
              and float(r["mad255"]) >= float(r["band_lo"]) and float(r["struct"]) >= args.min_struct
              and int(r["unchanged"]) == 0)
    if passed:
        ok.append("{}_it{}".format(args.tag, r["it"]))

auc = defaultdict(dict)
for r in csv.DictReader(open(args.val)):
    if r["image_auroc"]:
        auc[r["set"]][r["seed"]] = float(r["image_auroc"])
start = auc.get(args.tag, {})
best, best_gain = "none", None
for name in ok:
    seeds = set(auc.get(name, {})) & set(start)
    if seeds:
        gain = sum(auc[name][s] - start[s] for s in seeds) / len(seeds)
        if best_gain is None or gain > best_gain:
            best, best_gain = name, gain
print(best)
