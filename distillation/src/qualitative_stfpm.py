#!/usr/bin/env python3
"""Qualitative STFPM figure: [input | ground truth | map (distilled) | map (real)].
"""
import os, glob, argparse, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import sys; sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from lib.feature_matching import build_teacher, build_student, feature_mse_loss, DEFAULT_LAYERS

# ---------------------------------------------------------------- dati (input in [0,1])
def load_img(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1)   # [3,H,W] in [0,1]

def load_mask(path, size):
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 0).astype(np.float32)

def train_good_images(cat, root, size):
    files = sorted(glob.glob(os.path.join(root, cat, "train", "good", "*.png")))
    return torch.stack([load_img(f, size) for f in files])

def test_anomalous(cat, root, size, n=4):
    items = []
    for d in sorted(glob.glob(os.path.join(root, cat, "test", "*"))):
        defect = os.path.basename(d)
        if defect == "good": continue
        for f in sorted(glob.glob(os.path.join(d, "*.png"))):
            base = os.path.splitext(os.path.basename(f))[0]
            gt = os.path.join(root, cat, "ground_truth", defect, base + "_mask.png")
            if os.path.exists(gt): items.append((f, gt, defect))
    if len(items) > n: items = items[:: max(1, len(items)//n)][:n]
    return items

# train one student
def train_student(train, epochs, lr, dev):
    teacher = build_teacher(backbone="resnet18", layers=DEFAULT_LAYERS, device=dev,
                            prefer_moviad=True)
    student = build_student(backbone="resnet18", layers=DEFAULT_LAYERS, device=dev,
                            bn_track_stats=False, prefer_moviad=True)
    student.train()
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.1)
    bs = min(32, train.size(0))
    for ep in range(epochs):
        perm = torch.randperm(train.size(0), device=dev)
        for i in range(0, train.size(0), bs):
            xb = train[perm[i:i+bs]]
            with torch.no_grad(): ft = teacher(xb)
            fs = student(xb)
            loss = feature_mse_loss(ft, fs)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep+1) % 100 == 0: print(f"  epoch {ep+1}/{epochs}  loss {loss.item():.4f}")
    student.eval()
    return teacher, student

# ---------------------------------------------------------------- anomaly map STFPM (Eq. 1.4)
@torch.no_grad()
def anomaly_map(teacher, student, x, out_size):
    ft = teacher(x); fs = student(x)
    amap = torch.ones(x.size(0), 1, *out_size, device=x.device)        # prodotto sui livelli
    for t, s in zip(ft, fs):
        a = F.normalize(t, dim=1); b = F.normalize(s, dim=1)
        d = 0.5 * ((a - b) ** 2).sum(1, keepdim=True)
        d = F.interpolate(d, size=out_size, mode="bilinear", align_corners=False)
        amap = amap * d
    return amap.squeeze(1).cpu().numpy()

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--data_root", default="../data/mvtec_anomaly_detection")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--mask_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--train_dir", required=True, help=".pt immagini distillate")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="../outputs/qual_stfpm")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device); os.makedirs(args.out, exist_ok=True)

    real = train_good_images(args.category, args.data_root, args.img_size).to(dev)
    dist = torch.load(args.train_dir).float().to(dev)                  # già in [0,1]
    print(f"real {tuple(real.shape)} | dist {tuple(dist.shape)}")

    print("== training REAL =="); t_r, s_r = train_student(real, args.epochs, args.lr, dev)
    print("== training DIST =="); t_d, s_d = train_student(dist, args.epochs, args.lr, dev)

    items = test_anomalous(args.category, args.data_root, args.img_size, args.n)
    if not items:
        print("Nessuna immagine anomala con maschera trovata."); return
    for k, (fimg, fgt, defect) in enumerate(items):
        x = load_img(fimg, args.img_size).unsqueeze(0).to(dev)
        a_d = anomaly_map(t_d, s_d, x, (args.mask_size, args.mask_size))[0]
        a_r = anomaly_map(t_r, s_r, x, (args.mask_size, args.mask_size))[0]
        gt = load_mask(fgt, args.mask_size)
        disp = Image.open(fimg).convert("RGB").resize((args.mask_size, args.mask_size))
        def norm(a): return (a - a.min()) / (a.max() - a.min() + 1e-8)
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
        ax[0].imshow(disp); ax[0].set_title(f"{args.category} / {defect}")
        ax[1].imshow(gt, cmap="gray"); ax[1].set_title("Ground truth")
        ax[2].imshow(disp); ax[2].imshow(norm(a_d), cmap="jet", alpha=0.5); ax[2].set_title("Distilled")
        ax[3].imshow(disp); ax[3].imshow(norm(a_r), cmap="jet", alpha=0.5); ax[3].set_title("Real")
        for a in ax: a.axis("off")
        out = os.path.join(args.out, f"{args.category}_{defect}_{k}.png")
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print("salvato", out)

if __name__ == "__main__":
    main()