#!/usr/bin/env python3
"""Is the matched segment worth matching? (thesis Sec. 2.6-2.7)

Scores every snapshot of the first experts of a buffer as an STFPM detector: image AUROC on the whole
test set and on its validation half, pixel AUROC with masks at 224. BatchNorm as in the evaluation
(no running statistics). One CSV row per (expert, snapshot).
Usage: python expert_auroc.py --category cable --buffer_path ../outputs/buffer_s64strong_cable [--res 64] [--out x.csv]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.feature_matching import anomaly_scores, build_student, build_teacher, get_expert_dir
from distill import load_buffer_files

MASK = 224


def load_test(root, res):
    tf = transforms.Compose([transforms.Resize((res, res), antialias=True), transforms.ToTensor()])
    paths = sorted(Path(root, "test").rglob("*.png"))
    x, y, m = [], [], []
    for q in paths:
        x.append(tf(Image.open(q).convert("RGB")))
        good = q.parent.name == "good"
        y.append(0 if good else 1)
        if good:
            m.append(np.zeros((MASK, MASK), np.uint8))
        else:
            g = Image.open(Path(root, "ground_truth", q.parent.name, q.stem + "_mask.png")).convert("L")
            m.append((np.array(g.resize((MASK, MASK), Image.NEAREST)) > 0).astype(np.uint8))
    val = np.zeros(len(paths), bool)  # same halves as evaluate_pixel.py --split val
    for dt in sorted({q.parent.name for q in paths}):
        idx = [i for i, q in enumerate(paths) if q.parent.name == dt]
        val[idx[0::2]] = True
    return torch.stack(x), np.array(y), np.stack(m), val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--buffer_path", required=True)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--n_experts", type=int, default=3)
    ap.add_argument("--out", default="", help="append rows to this CSV")
    ap.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    args = ap.parse_args()

    dev = "cuda"
    teacher = build_teacher(device=dev).eval()
    student = build_student(device=dev, bn_track_stats=False, verbose=False).eval()
    x, y, m, val = load_test(os.path.join(args.data_path, args.category), args.res)
    x = x.to(dev)

    rows = []
    files = load_buffer_files(get_expert_dir(args.buffer_path, "MVTec_feature", "ResNet18"))
    for k, f in enumerate(files[:args.n_experts]):
        for e, snap in enumerate(torch.load(f, map_location="cpu")[0]):
            for p, s in zip(student.parameters(), snap):
                p.data.copy_(s.to(dev).reshape(p.shape))
            with torch.no_grad():
                maps = torch.cat([anomaly_scores(teacher, student, x[i:i + 1])[0] for i in range(len(x))])
            s_img = maps.flatten(1).max(1).values.cpu().numpy()
            up = F.interpolate(maps, size=(MASK, MASK), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            r = (args.category, k, e, roc_auc_score(y, s_img), roc_auc_score(y[val], s_img[val]), roc_auc_score(m.ravel(), up.ravel()))
            rows.append(r)
            print("%s expert %d snapshot %2d  image %.3f  image(val) %.3f  pixel %.3f" % r, flush=True)
    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, "a") as fh:
            if new:
                fh.write("category,expert,snapshot,image_auroc,image_auroc_val,pixel_auroc\n")
            for r in rows:
                fh.write("%s,%d,%d,%.4f,%.4f,%.4f\n" % r)


if __name__ == "__main__":
    main()
