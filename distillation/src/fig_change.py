#!/usr/bin/env python3
"""Before / after / difference figure of distilled sets (thesis Figs. 2.x, 5.x, 7.x).
Usage: python fig_change.py out.png 0,3,7 "label=initial.pt=final.pt" ["label2=a.pt=b.pt" ...]
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

out, idx, specs = sys.argv[1], [int(i) for i in sys.argv[2].split(",")], sys.argv[3:]
n, H = len(idx), len(specs)
fig, ax = plt.subplots(3 * H, n, figsize=(2.2 * n, 2.4 * 3 * H), squeeze=False)
titles = []
for b, spec in enumerate(specs):
    label, a, f = spec.split("=")
    x0, x1 = torch.load(a).float(), torch.load(f).float()
    for j, i in enumerate(idx):
        d = x1[i] - x0[i]
        g = 0.5 / max(d.abs().max().item(), 1e-8)  # amplify the difference to the full grey range
        rows = [(x0[i], "initial #%d" % i), (x1[i].clamp(0, 1), "optimised #%d" % i),
                ((0.5 + g * d).clamp(0, 1), "difference (x%.0f)" % (2 * g))]
        for r, (im, t) in enumerate(rows):
            ax[3 * b + r, j].imshow(im.permute(1, 2, 0).numpy())
            ax[3 * b + r, j].set_title(t, fontsize=8)
            ax[3 * b + r, j].axis("off")
    titles.append((b, "%s  (MAD %.1f/255)" % (label, (x1 - x0).abs().mean().item() * 255)))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.01, top=1 - 0.06 / H, hspace=0.25, wspace=0.05)
for b, t in titles:
    fig.text(0.5, 1 - b / H - 0.025 / H, t, ha="center", fontsize=10, fontweight="bold")
    if b:
        for j in range(n):
            p = ax[3 * b, j].get_position()
            ax[3 * b, j].set_position([p.x0, p.y0 - 0.02 / H, p.width, p.height])
plt.savefig(out, dpi=130)
print("Saved", out)
