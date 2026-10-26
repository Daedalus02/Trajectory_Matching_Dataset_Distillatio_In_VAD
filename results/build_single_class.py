"""Merge the single-class runs into one CSV for make_tables.py.

Real rows come from the main sweep; distilled rows from the re-evaluation of the last
distillation iteration (no test-set selection), which also has the F1-max.
STFPM distilled rows use the 400-epoch re-evaluation (same budget as the real rows).

Usage:  python build_single_class.py   ->  merged/results_raw.csv, merged/summary.csv
"""
import csv
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw' / 'single_class'
DST = HERE / 'merged'
COLS = ['category', 'detector', 'res', 'type', 'run', 'image_auroc', 'pixel_auroc', 'image_f1max', 'pixel_f1max']


def main():
    DST.mkdir(exist_ok=True)
    rows = [{k: r.get(k, '') for k in COLS}
            for r in csv.DictReader(open(RAW / 'results_raw.clean.csv')) if r['type'] == 'real']
    n_real = len(rows)
    # RD4AD and Dinomaly: last-iteration re-evaluation
    for r in csv.DictReader(open(RAW / 'results_lastckpt_f1.csv')):
        if r['detector'] != 'STFPM':
            rows.append({k: dict(r, type='distilled').get(k, '') for k in COLS})
    # STFPM: 400-epoch re-evaluation
    for r in csv.DictReader(open(RAW / 'results_lastckpt_f1_stfpm_ep400.csv')):
        rows.append({k: dict(r, type='distilled').get(k, '') for k in COLS})
    with open(DST / 'results_raw.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    # summary.csv only fills the RD4AD/bottle real cell, missing from results_raw.csv
    shutil.copy(RAW / 'summary.csv', DST / 'summary.csv')
    print('real rows: %d, distilled rows: %d -> %s' % (n_real, len(rows) - n_real, DST))


if __name__ == '__main__':
    main()
