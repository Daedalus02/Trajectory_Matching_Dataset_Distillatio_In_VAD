"""Mean +- std over seeds of the continual results (thesis Ch. 5 tables).

Reads raw/continual/<dir>/*.json (one file per run, written by main_*_clad.py) and prints, for
each configuration: final image/pixel AUROC, final image/pixel F1 and image forgetting.
Usage:  python aggregate_continual.py
"""
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw', 'continual')


def ms(xs):
    return '%.3f+-%.3f' % (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


groups = {}
for f in sorted(glob.glob(os.path.join(RAW, '*', '*.json'))):
    j = json.load(open(f))
    if 'final_average' not in j:          # old joint runs: no per-task metrics stored
        continue
    key = os.path.basename(os.path.dirname(f)) + '/' + re.sub(r'_s\d\.json$', '', os.path.basename(f))
    groups.setdefault(key, []).append(j)

print('%-75s %s  %-12s %-12s %-12s %-12s %s' % ('run', 'n', 'img_final', 'pix_final', 'img_F1', 'pix_F1', 'forget_img'))
for k, js in groups.items():
    fa = lambda m: [j['final_average'][m] for j in js]
    print('%-75s %d  %s %s %s %s %s' % (k, len(js), ms(fa('img_roc_auc')), ms(fa('pxl_roc_auc')),
          ms(fa('img_f1')), ms(fa('pxl_f1')), ms([j['forgetting']['img_roc_auc'] for j in js])))
