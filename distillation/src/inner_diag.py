#!/usr/bin/env python3
"""Is the matching problem solvable? (thesis Sec. 2.7)

Runs the inner loop from fixed expert snapshots on given image sets and prints the normalised
matching loss for every (expert_epochs, inner lr, inner steps). A usable setting has the loss of the
ten starting images below 1 and the loss of all the real images clearly lower.
Sets: real10 (the ten images of the seed-0 permutation, i.e. the default starting set), all_real, or name=path.pt.
Usage: python inner_diag.py --category carpet --buffer_path ../outputs/buffer_s64strong_carpet \
           --lo 10 --hi 18 --lrs 0.1 0.2 0.4 --steps 20 40 --ees 1 2 --sets real10 all_real
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.reparam_module import ReparamModule
from lib.feature_matching import (build_student, build_teacher, check_trajectory_layout, feature_mse_loss,
                                  flatten_snapshot, get_expert_dir, load_images_tensor)
from distill import load_buffer_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--buffer_path", required=True)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--lo", type=int, required=True, help="first start snapshot")
    ap.add_argument("--hi", type=int, required=True, help="last start snapshot (excluded)")
    ap.add_argument("--lrs", type=float, nargs="+", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--ees", type=int, nargs="+", default=[2], help="snapshots between start and target")
    ap.add_argument("--sets", nargs="+", default=["real10", "all_real"])
    ap.add_argument("--n_traj", type=int, default=6)
    ap.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    teacher = build_teacher(device=dev)
    net = ReparamModule(build_student(device=dev, bn_track_stats=False, verbose=False))
    trajs = []
    for f in load_buffer_files(get_expert_dir(args.buffer_path, "MVTec_feature", "ResNet18")):
        trajs += torch.load(f, map_location="cpu")
        if len(trajs) >= args.n_traj:
            break
    trajs = trajs[:args.n_traj]
    check_trajectory_layout(net, trajs[0][0])

    real = load_images_tensor(os.path.join(args.data_path, args.category, "train", "good"), args.res, verbose=False).float()
    np.random.seed(0)
    idx = np.random.permutation(len(real))
    sets = {}
    for spec in args.sets:
        if spec == "real10":
            sets[spec] = real[idx[:10]]
        elif spec == "all_real":
            sets[spec] = real
        else:
            name, path = spec.split("=", 1)
            sets[name] = torch.load(path, map_location="cpu").float()

    for ee in args.ees:
        pairs = [(t, e) for t in range(len(trajs)) for e in range(args.lo, min(args.hi, len(trajs[0]) - ee))]
        for name, x in sets.items():
            x = x.to(dev)
            for lr in args.lrs:
                for steps in args.steps:
                    losses = []
                    for ti, e in pairs:
                        s0, s1 = flatten_snapshot(trajs[ti][e], dev), flatten_snapshot(trajs[ti][e + ee], dev)
                        w = s0.clone()
                        for _ in range(steps):
                            b = x if len(x) <= 10 else x[torch.randperm(len(x), device=dev)[:32]]
                            with torch.no_grad():
                                tb = teacher(b)
                            w.requires_grad_(True)
                            g = torch.autograd.grad(feature_mse_loss(tb, net(b, flat_param=w)), w)[0]
                            w = (w - lr * g).detach()
                        losses.append((F.mse_loss(w, s1, reduction="sum") / F.mse_loss(s0, s1, reduction="sum")).item())
                    print("ee %d %-9s lr %-5g steps %3d  loss %.3f (min %.3f max %.3f)"
                          % (ee, name, lr, steps, np.mean(losses), np.min(losses), np.max(losses)), flush=True)


if __name__ == "__main__":
    main()
