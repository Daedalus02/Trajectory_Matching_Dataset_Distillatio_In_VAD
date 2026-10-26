"""Regenerate the single-class tables (tables/*.tex) and plot data (data/*.dat) of the thesis.

Duplicated (category, detector, res, type, run) rows keep the last occurrence; macro-averages
use only categories with all 5 runs. Usage:
    python build_single_class.py
    python make_tables.py --sweep merged --tables tables --data data
"""
import argparse
import csv
import statistics as st
from collections import OrderedDict
from pathlib import Path

from scipy.stats import spearmanr

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
        'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
TEXTURES = {'carpet', 'grid', 'leather', 'tile', 'wood'}
RES = [64, 128]
TAG = {'STFPM': 'Stfpm', 'RD4AD': 'Rd'}  # macro names: \rhoStfpmImg, \rhoRdPix, ...
FULL = 5
# mean fraction of anomalous pixels in the GT masks (%), from the anomaly-size analysis
# (results_tables.md, Tabella 6) -- computed on the MVTec test masks.
AREA = {'screw': 0.34, 'leather': 0.87, 'grid': 0.95, 'capsule': 1.11, 'carpet': 2.11,
        'toothbrush': 2.14, 'zipper': 2.62, 'hazelnut': 3.35, 'pill': 3.98, 'cable': 4.68,
        'wood': 5.09, 'bottle': 7.62, 'tile': 9.80, 'transistor': 11.98, 'metal_nut': 14.49}


def load(sweep):
    runs = OrderedDict()
    dups = 0
    src = Path(sweep, 'results_raw.clean.csv')  # keep-last deduplicated copy of results_raw.csv
    if not src.exists():
        src = Path(sweep, 'results_raw.csv')
    print('source:', src)
    for r in csv.DictReader(open(src)):
        key = (r['detector'], r['category'], int(r['res']), r['type'], int(r['run']))
        if key in runs:
            dups += 1
            del runs[key]  # keep the last occurrence
        f1 = lambda k: float(r[k]) if r.get(k) not in (None, '', 'NA') else None
        runs[key] = (float(r['image_auroc']), float(r['pixel_auroc']), f1('image_f1max'), f1('pixel_f1max'))
    cells = {}
    for (det, cat, res, typ, _), vals in runs.items():
        cells.setdefault((det, cat, res, typ), []).append(vals)
    stats = {}
    for k, v in cells.items():
        im = [a[0] for a in v]
        px = [a[1] for a in v]
        sd = (lambda a: st.stdev(a) if len(a) > 1 else 0.0)
        stats[k] = dict(n=len(v), img=st.mean(im), img_s=sd(im), pix=st.mean(px), pix_s=sd(px))
        fi = [a[2] for a in v if a[2] is not None]
        fp = [a[3] for a in v if a[3] is not None]
        if len(fi) == len(v):  # F1-max only where every run has it (leakage-free distilled runs)
            stats[k].update(f1i=st.mean(fi), f1i_s=sd(fi), f1p=st.mean(fp), f1p_s=sd(fp))
    added = []
    for r in csv.DictReader(open(Path(sweep, 'summary.csv'))):
        k = (r['detector'], r['category'], int(r['res']), r['type'])
        if k not in stats and r['detector'] == 'RD4AD':
            stats[k] = dict(n=int(r['n']), img=float(r['image_mean']), img_s=float(r['image_std']),
                            pix=float(r['pixel_mean']), pix_s=float(r['pixel_std']))
            added.append(k)
    print('duplicated rows dropped (kept last):', dups)
    print('cells taken from summary.csv:', added)
    return stats


def fmt(s, metric):
    if s is None:
        return '--'
    flag = '' if s['n'] >= FULL else '$^{*}$'
    return '%.3f$\\pm$%.3f%s' % (s[metric], s[metric + '_s'], flag)


def complete(stats, det, cat, res):
    return all(stats.get((det, cat, res, t), {}).get('n', 0) >= FULL for t in ('distilled', 'real'))


def macro(stats, det, res, metric, cats):
    d = st.mean(stats[(det, c, res, 'distilled')][metric] for c in cats)
    r = st.mean(stats[(det, c, res, 'real')][metric] for c in cats)
    return d, r


def per_category_table(stats, det, metric):
    lines = []
    for c in CATS:
        row = [c.replace('_', '\\_')]
        for res in RES:
            for t in ('distilled', 'real'):
                row.append(fmt(stats.get((det, c, res, t)), metric))
        lines.append(' & '.join(row) + ' \\\\')
    lines.append('\\midrule')
    row = ['Macro-average']
    counts = []
    for res in RES:
        cats = [c for c in CATS if complete(stats, det, c, res)]
        counts.append(len(cats))
        if cats:
            d, r = macro(stats, det, res, metric, cats)
            row += ['%.3f' % d, '%.3f' % r]
        else:
            row += ['--', '--']
    lines.append(' & '.join(row) + ' \\\\')
    lines.append('\\multicolumn{5}{l}{\\footnotesize Categories in the macro-average: %d at 64, %d at 128.} \\\\'
                 % tuple(counts))
    return '\n'.join(lines) + '\n'


def gap_table(stats):
    lines = []
    for det in ('STFPM', 'RD4AD'):
        for metric, name in (('img', 'Image'), ('pix', 'Pixel')):
            for res in RES:
                cats = [c for c in CATS if complete(stats, det, c, res)]
                if not cats:
                    continue
                d, r = macro(stats, det, res, metric, cats)
                lines.append('\\gls{%s} & %s-\\gls{auroc} & %d & %d & %.3f & %.3f & $%+.3f$ \\\\'
                             % (det.lower(), name, res, len(cats), d, r, d - r))
        lines.append('\\midrule')
    # common categories (both detectors complete) -- the fair two-detector comparison
    for res in RES:
        common = [c for c in CATS if complete(stats, 'STFPM', c, res) and complete(stats, 'RD4AD', c, res)]
        for metric, name in (('img', 'Image'), ('pix', 'Pixel')):
            for det in ('STFPM', 'RD4AD'):
                d, r = macro(stats, det, res, metric, common)
                lines.append('\\gls{%s} (common) & %s-\\gls{auroc} & %d & %d & %.3f & %.3f & $%+.3f$ \\\\'
                             % (det.lower(), name, res, len(common), d, r, d - r))
    return '\n'.join(lines) + '\n', None


def bottle_table(stats):
    lines = []
    for res in RES:
        for det in ('STFPM', 'RD4AD'):
            for t, lab in (('distilled', 'Dist'), ('real', 'Real')):
                s = stats.get((det, 'bottle', res, t))
                lines.append('%d & \\gls{%s} & %s & %s & %s \\\\' % (res, det.lower(), lab, fmt(s, 'img'), fmt(s, 'pix')))
        if res != RES[-1]:
            lines.append('\\midrule')
    return '\n'.join(lines) + '\n'


def macro3_table(stats):
    """Macro comparison of the three detectors (tables/macro_3det.tex), with the F1-max of the
    distilled detector when the sweep carries it."""
    rows = []
    for det, res in (('STFPM', 64), ('STFPM', 128), (None, None), ('RD4AD', 64), ('RD4AD', 128),
                     ('MID', None), ('DINOMALY', 224)):
        if det is None:
            rows.append('\\addlinespace')
            continue
        if det == 'MID':
            rows.append('\\midrule')
            continue
        cats = [c for c in CATS if complete(stats, det, c, res)]
        di, ri = macro(stats, det, res, 'img', cats)
        dp, rp = macro(stats, det, res, 'pix', cats)
        dist = [stats[(det, c, res, 'distilled')] for c in cats]
        f1 = ['$%.3f$' % st.mean(s[m] for s in dist) if all(m in s for s in dist) else '--' for m in ('f1i', 'f1p')]
        name = 'Dinomaly' if det == 'DINOMALY' else det
        rows.append('%s & $%d$ & $%.3f$ & $%.3f$ & $%+.3f$ & $%.3f$ & $%.3f$ & $%+.3f$ & %s & %s \\\\'
                    % (name, res, di, ri, di - ri, dp, rp, dp - rp, f1[0], f1[1]))
    return r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Macro-average comparison of the three detectors on MVTec AD (15 categories,
IPC${=}10$, mean over 5 runs). The distilled set is the one of the \emph{last} distillation
iteration (no selection on the test split). Gap${=}$Dist${-}$Real (negative when the distilled
detector is below the real one); F1 is the F1-max of the distilled detector. STFPM and RD4AD use a
ResNet backbone at $64/128$; Dinomaly uses a DINOv2 ViT-S at its native $224$ (a separate block,
not comparable in resolution).}
\label{tab:macro-3det}
\begin{tabular}{ll ccc ccc cc}
\toprule
 & & \multicolumn{3}{c}{\textbf{Image-AUROC}} & \multicolumn{3}{c}{\textbf{Pixel-AUROC}} & \multicolumn{2}{c}{\textbf{Dist F1-max}} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}
Detector & Res. & Dist & Real & Gap & Dist & Real & Gap & Image & Pixel \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def dinomaly_table(stats):
    """Per-category Dinomaly table (tables/dinomaly_percat.tex)."""
    det, res = 'DINOMALY', 224
    sc = lambda s, m: '$%.3f{\\scriptstyle\\pm%.3f}$' % (s[m], s[m + '_s'])
    lines = []
    for c in CATS:
        d, r = stats[(det, c, res, 'distilled')], stats[(det, c, res, 'real')]
        lines.append('%s & %s & %s & & %s & %s \\\\' % (c.replace('_', '\\_'), sc(d, 'img'), sc(r, 'img'), sc(d, 'pix'), sc(r, 'pix')))
    di, ri = macro(stats, det, res, 'img', CATS)
    dp, rp = macro(stats, det, res, 'pix', CATS)
    return r"""\begin{sidewaystable}
\centering
\footnotesize
\setlength{\tabcolsep}{6pt}
\caption{Dinomaly (DINOv2 ViT-S, native resolution $224$), feature-matching distillation versus
the real-data baseline, per MVTec AD category (IPC${=}10$, mean$\pm$std over 5 evaluation runs,
50 evaluation epochs; GT mask at $224$; distilled set of the last distillation iteration). Not comparable at
$64/128$ with STFPM/RD4AD because of the ViT patching.}
\label{tab:dinomaly-percat}
\begin{tabular}{l cc c cc}
\toprule
 & \multicolumn{2}{c}{\textbf{Image-AUROC}} & & \multicolumn{2}{c}{\textbf{Pixel-AUROC}} \\
\cmidrule(lr){2-3}\cmidrule(lr){5-6}
Category & Dist & Real & & Dist & Real \\
\midrule
""" + '\n'.join(lines) + r"""
\midrule
\textbf{Macro-average} & $%.3f$ & $%.3f$ & & $%.3f$ & $%.3f$ \\
\bottomrule
\end{tabular}
\end{sidewaystable}
""" % (di, ri, dp, rp)


def write_dat(path, header, rows):
    with open(path, 'w') as f:
        f.write(' '.join(header) + '\n')
        for r in rows:
            f.write(' '.join(str(x) for x in r) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', required=True)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument('--tables', default=str(root / 'tables'))
    ap.add_argument('--data', default=str(root / 'img' / 'data'))
    ap.add_argument('--outputs', default=None, help='VAD_distill/outputs (for the distillation curves)')
    args = ap.parse_args()
    stats = load(args.sweep)
    tdir, ddir = Path(args.tables), Path(args.data)
    tdir.mkdir(parents=True, exist_ok=True)
    ddir.mkdir(parents=True, exist_ok=True)
    banner = '% AUTO-GENERATED by scripts/make_tables.py from results_raw.csv -- do not edit by hand\n'

    for det in ('STFPM', 'RD4AD'):
        for metric, name in (('img', 'image'), ('pix', 'pixel')):
            (tdir / ('%s_%s.tex' % (det.lower(), name))).write_text(banner + per_category_table(stats, det, metric))
    # gap.tex (2-detector macro table) retired: replaced by tables/macro_3det.tex (3 detectors)
    (tdir / 'bottle_detectors.tex').write_text(banner + bottle_table(stats))
    if all(complete(stats, 'DINOMALY', c, 224) for c in CATS):
        (tdir / 'macro_3det.tex').write_text(banner + macro3_table(stats))
        (tdir / 'dinomaly_percat.tex').write_text(banner + dinomaly_table(stats))

    # --- per-category gap (Dist - Real), for the bar charts -------------------
    for det in ('STFPM', 'RD4AD'):
        rows = []
        for i, c in enumerate(CATS):
            row = [i, c.replace('_', '-')]
            for res in RES:
                for metric in ('img', 'pix'):
                    if complete(stats, det, c, res):
                        row.append('%.4f' % (stats[(det, c, res, 'distilled')][metric] - stats[(det, c, res, 'real')][metric]))
                    else:
                        row.append('nan')
            rows.append(row)
        write_dat(ddir / ('gap_%s.dat' % det.lower()), ['idx', 'cat', 'img64', 'pix64', 'img128', 'pix128'], rows)

    # --- AUROC sorted by anomaly size (STFPM, pixel and image, 64) ------------
    for det in ('STFPM', 'RD4AD'):
        rows = []
        for i, c in enumerate(sorted(CATS, key=AREA.get)):
            row = [i, c.replace('_', '-'), AREA[c]]
            for res in RES:
                for t in ('distilled', 'real'):
                    s = stats.get((det, c, res, t))
                    ok = complete(stats, det, c, res)
                    row += ['%.4f' % s['img'] if ok else 'nan', '%.4f' % s['pix'] if ok else 'nan']
            rows.append(row)
        write_dat(ddir / ('bysize_%s.dat' % det.lower()),
                  ['idx', 'cat', 'area', 'img64d', 'pix64d', 'img64r', 'pix64r', 'img128d', 'pix128d', 'img128r', 'pix128r'], rows)

    # --- resolution gain vs anomaly size (real baseline) + Spearman -----------
    size_lines = []
    for det in ('STFPM', 'RD4AD'):
        rows, a, di, dp = [], [], [], []
        for c in sorted(CATS, key=AREA.get):
            if not (complete(stats, det, c, 64) and complete(stats, det, c, 128)):
                continue
            gi = stats[(det, c, 128, 'real')]['img'] - stats[(det, c, 64, 'real')]['img']
            gp = stats[(det, c, 128, 'real')]['pix'] - stats[(det, c, 64, 'real')]['pix']
            rows.append([c.replace('_', '-'), AREA[c], '%.4f' % gi, '%.4f' % gp])
            a.append(AREA[c]); di.append(gi); dp.append(gp)
            if det == 'STFPM':
                size_lines.append('%s & %.2f & $%+.3f$ & $%+.3f$ \\\\' % (c.replace('_', '\\_'), AREA[c], gi, gp))
        write_dat(ddir / ('resgain_%s.dat' % det.lower()), ['cat', 'area', 'dimg', 'dpix'], rows)
        if len(a) > 2:
            ri, pi = spearmanr(a, di)
            rp, pp = spearmanr(a, dp)
            print('%s resolution gain vs area: n=%d  Spearman image rho=%+.2f (p=%.2f)  pixel rho=%+.2f (p=%.2f)'
                  % (det, len(a), ri, pi, rp, pp))
            if det == 'STFPM':
                size_lines.append('\\midrule')
                size_lines.append('\\multicolumn{2}{l}{Spearman $\\rho$ (area, $\\Delta$), $n=%d$} & $%+.2f$ & $%+.2f$ \\\\'
                                  % (len(a), ri, rp))
            with open(tdir / ('spearman_%s.tex' % det.lower()), 'w') as f:
                f.write(banner)
                f.write('\\newcommand{\\rho%sImg}{%+.2f}\\newcommand{\\rho%sPix}{%+.2f}\\newcommand{\\n%sSize}{%d}\n'
                        % (TAG[det], ri, TAG[det], rp,
                           TAG[det], len(a)))
    (tdir / 'size_stfpm.tex').write_text(banner + '\n'.join(size_lines) + '\n')

    # --- distillation curves (grand_loss vs iteration), mean over categories ---
    if args.outputs:
        out = Path(args.outputs)
        win = 20
        cols, header = [], ['iter']
        for det, sub in (('stfpm', 'distilled'), ('rd4ad', 'distilled_rd4ad')):
            for res in RES:
                curves = []
                for c in CATS:
                    h = out / sub / c / '10' / 'real' / ('full%d' % res) / 'history.csv'
                    if not h.exists():
                        continue
                    loss = [float(r['grand_loss']) for r in csv.DictReader(open(h))]
                    if len(loss) >= 301:
                        curves.append(loss[:301])
                if not curves:
                    continue
                mean = [st.mean(v) for v in zip(*curves)]
                smooth = [st.mean(mean[max(0, i - win + 1):i + 1]) for i in range(len(mean))]
                cols.append(smooth)
                header.append('%s%d' % (det, res))
                print('distillation curve %s@%d: %d categories' % (det, res, len(curves)))
        rows = [[i] + ['%.4f' % col[i] for col in cols] for i in range(0, 301, 5)]
        write_dat(ddir / 'distcurve.dat', header, rows)

    # --- console summary of completeness -------------------------------------
    for det in ('STFPM', 'RD4AD'):
        for res in RES:
            miss = [c for c in CATS if not complete(stats, det, c, res)]
            print('%s @%d incomplete/missing: %s' % (det, res, miss or 'none'))


if __name__ == '__main__':
    main()
