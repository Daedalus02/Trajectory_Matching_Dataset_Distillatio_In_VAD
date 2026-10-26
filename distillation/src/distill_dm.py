#!/usr/bin/env python3
"""Distribution matching in the STFPM teacher feature space (thesis Ch. 7). No experts, no unrolling.

Every spatial position of the teacher layer1-3 maps, channel-normalised, is one sample ("patch").
The synthetic patches are matched to the patches of all the real train/good images of the category:
    loss = MMD^2 / MMD^2_0 + cov_weight * cov / cov_0
with a multi-bandwidth RBF MMD and the coverage term cov = mean over real patches of the squared
distance to the nearest synthetic patch (_0 = value at the first iteration).
--loss moments matches only the per-channel mean and standard deviation.

Outputs in outputs/distilled/<category>/<ipc>/real/<name>/: images_<it>.pt (pixels in [0,1]), history.csv.
"""

import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.feature_matching import DEFAULT_LAYERS, build_teacher, load_images_tensor


def patches(feats):
    """List of [B,C,H,W] maps -> list of [B*H*W, C] channel-normalised patch features."""
    return [F.normalize(f, dim=1).permute(0, 2, 3, 1).reshape(-1, f.shape[1]) for f in feats]


def mmd2(x, y, sigmas2):
    """Biased multi-bandwidth RBF MMD^2 between x [n,C] and y [m,C]."""
    xx, yy, xy = torch.cdist(x, x).pow(2), torch.cdist(y, y).pow(2), torch.cdist(x, y).pow(2)
    k = 0.0
    for s2 in sigmas2:
        k = k + torch.exp(-xx / (2 * s2)).mean() + torch.exp(-yy / (2 * s2)).mean() - 2 * torch.exp(-xy / (2 * s2)).mean()
    return k


def moments(x, y):
    return (x.mean(0) - y.mean(0)).pow(2).sum() + (x.std(0) - y.std(0)).pow(2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    ap.add_argument("--init", required=True, help="starting set, .pt [ipc,3,res,res] in [0,1]")
    ap.add_argument("--name", required=True, help="output folder name")
    ap.add_argument("--ipc", type=int, default=10)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--loss", choices=["mmd", "moments"], default="mmd")
    ap.add_argument("--cov", type=float, default=0.0, help="weight of the coverage term")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.01, help="Adam lr on the pixels, cosine decay")
    ap.add_argument("--n_real", type=int, default=4096, help="real patches sampled per layer and iteration")
    ap.add_argument("--n_syn", type=int, default=4096, help="max synthetic patches per layer and iteration")
    ap.add_argument("--sigmas2", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--save_its", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join("..", "outputs", "distilled", args.category, str(args.ipc), "real", args.name)
    os.makedirs(out, exist_ok=True)

    teacher = build_teacher(layers=DEFAULT_LAYERS, device=dev)
    real = load_images_tensor(os.path.join(args.data_path, args.category, "train", "good"), img_size=args.res)
    with torch.no_grad():
        real_p = [torch.cat(z) for z in zip(*[patches(teacher(real[i:i + 32].to(dev))) for i in range(0, len(real), 32)])]

    x0 = torch.load(args.init, map_location="cpu").float().clamp(0, 1)
    assert x0.shape[0] == args.ipc and x0.shape[-1] == args.res, x0.shape
    x = x0.clone().to(dev).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    torch.save(x0, os.path.join(out, "images_0.pt"))

    with open(os.path.join(out, "history.csv"), "w", newline="") as hist:
        w = csv.writer(hist)
        w.writerow(["iter", "loss", "dist", "cov", "mad255"])
        for it in range(1, args.iters + 1):
            dist, cov = 0.0, 0.0
            for sp, rp in zip(patches(teacher(x)), real_p):
                rp = rp[torch.randint(len(rp), (min(args.n_real, len(rp)),), device=dev)]
                if args.cov > 0:
                    cov = cov + torch.cdist(rp, sp).pow(2).min(1).values.mean()
                if len(sp) > args.n_syn:
                    sp = sp[torch.randperm(len(sp), device=dev)[:args.n_syn]]
                dist = dist + (mmd2(sp, rp, args.sigmas2) if args.loss == "mmd" else moments(sp, rp))
            if it == 1:
                d0, c0 = dist.item(), (cov.item() if args.cov > 0 else 1.0)
            loss = dist / d0 + (args.cov * cov / c0 if args.cov > 0 else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                x.clamp_(0, 1)
            if it % 50 == 0 or it == 1:
                mad = (x.detach().cpu() - x0).abs().mean().item() * 255
                cv = cov.item() if args.cov > 0 else float("nan")
                w.writerow([it, "%.6f" % loss.item(), "%.6f" % dist.item(), "%.6f" % cv, "%.3f" % mad])
                hist.flush()
                print("iter %04d | loss %.4f | dist %.6f | cov %.5f | MAD %.2f" % (it, loss.item(), dist.item(), cv, mad), flush=True)
            if it in args.save_its:
                torch.save(x.detach().cpu().clone(), os.path.join(out, "images_{}.pt".format(it)))
    print("Saved to", out)


if __name__ == "__main__":
    main()
