#!/usr/bin/env python3
"""Macro-averages of the multi-class runs vs the per-category runs (standard library only).
"""

import argparse
import csv
from collections import defaultdict
from statistics import mean, stdev


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def macro(rows, metric):
    """Media sulle categorie delle medie per categoria; nan se non ci sono righe."""
    per_cat = defaultdict(list)
    for r in rows:
        if r.get(metric) not in (None, "", "NA"):
            per_cat[r["category"]].append(float(r[metric]))
    if not per_cat:
        return float("nan")
    return mean(mean(v) for v in per_cat.values())


def main(args):
    mc = read_rows(args.mc)
    pc = read_rows(args.percat)
    groups = sorted({(r["detector"].replace("_MC", ""), r["res"]) for r in mc})

    for det, res in groups:
        g = [r for r in mc if r["detector"].replace("_MC", "") == det and r["res"] == res]
        ref = [r for r in pc if r["detector"] == det and r["res"] == res]
        runs = sorted({int(r["run"]) for r in g})
        print("\n=== {} @ {}  (run MC: {}) ===".format(det, res, runs))
        for metric in ("image_auroc", "pixel_auroc"):
            vals = {}
            for t in ("distilled", "real"):
                p = macro([r for r in ref if r["type"] == t], metric)
                m = macro([r for r in g if r["type"] == t], metric)
                vals[t] = (p, m)
                print("  {:<12s} {:<9s} per-cat {:.3f}  multi {:.3f}  delta {:+.3f}".format(
                    metric, t, p, m, m - p))
            print("  {:<12s} gap real-dist: per-cat {:+.3f}  multi {:+.3f}".format(
                metric, vals["real"][0] - vals["distilled"][0],
                vals["real"][1] - vals["distilled"][1]))
        for metric in ("image_f1max", "pixel_f1max"):
            for t in ("distilled", "real"):
                m = macro([r for r in g if r["type"] == t], metric)
                print("  {:<12s} {:<9s} multi {:.3f}".format(metric, t, m))

    if args.percat_table:
        metrics = ("image_auroc", "pixel_auroc", "image_f1max", "pixel_f1max")
        cells = defaultdict(lambda: defaultdict(list))
        for r in mc:
            key = (r["detector"], r["res"], r["type"], r["category"])
            for m in metrics:
                cells[key][m].append(float(r[m]))
        with open(args.percat_table, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["detector", "res", "type", "category", "n"]
                       + [f"{m}_{s}" for m in metrics for s in ("mean", "std")])
            for key in sorted(cells):
                v = cells[key]
                n = len(v[metrics[0]])
                w.writerow(list(key) + [n] + [
                    "{:.4f}".format(x) for m in metrics
                    for x in (mean(v[m]), stdev(v[m]) if n > 1 else 0.0)])
        print("\nTabella per-categoria (media/std sui run, ddof=1) in {}".format(args.percat_table))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", default="../outputs/multiclass/results_multiclass_3det.csv")
    ap.add_argument("--percat", default="../outputs/thesis_sweep/results_raw.clean.csv")
    ap.add_argument("--percat_table", default="../outputs/multiclass/summary_percat.csv")
    main(ap.parse_args())
