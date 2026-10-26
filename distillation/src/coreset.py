#!/usr/bin/env python3
"""Coverage coreset: n real images chosen greedily (facility location) so that every real patch
(teacher layer2) is close to a patch of the selected images, as in the PatchCore coreset.
Usage: python coreset.py --category cable --out ../outputs/inits/cable/core.pt [--n 10] [--res 64]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.feature_matching import build_teacher, load_images_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    args = ap.parse_args()

    dev = "cuda"
    teacher = build_teacher(device=dev, prefer_moviad=False)
    r = load_images_tensor(os.path.join(args.data_path, args.category, "train", "good"), args.res, verbose=False).float()
    with torch.no_grad():
        p = torch.cat([teacher(r[i:i + 64].to(dev))[1] for i in range(0, len(r), 64)])
        ppi = p.shape[2] * p.shape[3]
        p = p.permute(0, 2, 3, 1).reshape(-1, p.shape[1])
        # d[j, i] = distance of real patch j to the closest patch of image i
        d = torch.stack([torch.cdist(p, p[i * ppi:(i + 1) * ppi]).min(1).values for i in range(len(r))], 1)
        best = torch.full((len(p),), float("inf"), device=dev)
        sel = []
        for _ in range(args.n):
            cand = torch.minimum(best[:, None], d).mean(0)
            cand[sel] = float("inf")
            i = int(cand.argmin())
            sel.append(i)
            best = torch.minimum(best, d[:, i])
            print("pick %3d  mean distance to the closest selected patch %.4f" % (i, best.mean().item()))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(r[sel].clone(), args.out)
    print("Saved", args.out, tuple(r[sel].shape), "indices", sel)


if __name__ == "__main__":
    main()
