#!/usr/bin/env python3
"""Qualitative RD4AD figure: [input | ground truth | predicted map] for a few anomalous images.
"""
import os, glob, argparse, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# project lib
import sys; sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from lib.rd4ad_matching import build_encoder, build_rd4ad_student, rd4ad_cosine_loss  # ### API ###

IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------- dati
def load_img(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1)
    for c in range(3): x[c] = (x[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return x

def load_mask(path, size):  # mask at 224, as in the evaluation
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 0).astype(np.float32)

def train_good_images(cat, root, size):
    files = sorted(glob.glob(os.path.join(root, cat, "train", "good", "*.png")))
    return torch.stack([load_img(f, size) for f in files])

def test_anomalous(cat, root, size, mask_size=224, n=4):
    items = []
    for d in sorted(glob.glob(os.path.join(root, cat, "test", "*"))):
        defect = os.path.basename(d)
        if defect == "good": continue
        for f in sorted(glob.glob(os.path.join(d, "*.png"))):
            base = os.path.splitext(os.path.basename(f))[0]
            gt = os.path.join(root, cat, "ground_truth", defect, base + "_mask.png")
            if os.path.exists(gt): items.append((f, gt, defect))
    # prendi n esempi spaziati tra i difetti
    if len(items) > n: items = items[:: max(1, len(items)//n)][:n]
    return items

# ---------------------------------------------------------------- anomaly map
@torch.no_grad()
def anomaly_map(encoder, student, x, out_size, blur_sigma=4.0):
    enc = encoder(x)                 # lista [E1,E2,E3]
    dec = student(enc)               # lista [D1,D2,D3]   (student = bn+decoder)
    amap = torch.zeros(x.size(0), 1, *out_size, device=x.device)
    for fe, fd in zip(enc, dec):
        a = F.normalize(fe, dim=1); b = F.normalize(fd, dim=1)
        m = 1.0 - (a * b).sum(1, keepdim=True)                     # per-pixel cosine
        m = F.interpolate(m, size=out_size, mode="bilinear", align_corners=False)
        amap += m
    amap = amap.squeeze(1).cpu().numpy()
    if blur_sigma > 0:
        amap = np.stack([gaussian_filter(a, sigma=blur_sigma) for a in amap])
    return amap  # [B,H,W]

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--data_root", default="../data/mvtec_anomaly_detection")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--mask_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--train_dir", default=None, help=".pt di immagini distillate; se assente usa il reale good")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="../outputs/qual_rd4ad")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device); os.makedirs(args.out, exist_ok=True)

    # training data
    if args.train_dir:
        train = torch.load(args.train_dir).float()      # [N,3,H,W] distillate
    else:
        train = train_good_images(args.category, args.data_root, args.img_size)
    train = train.to(dev)
    print(f"train set: {tuple(train.shape)}")

    # ======================= ### API ### (allinea a evaluate_pixel_rd4ad.py) =======================
    encoder = build_encoder().to(dev).eval()                 # encoder ImageNet congelato
    for p in encoder.parameters(): p.requires_grad_(False)
    student = build_rd4ad_student().to(dev).train()          # OCBE (bn) + decoder, allenabile
    opt = torch.optim.Adam(student.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bs = min(16, train.size(0))
    for ep in range(args.epochs):
        perm = torch.randperm(train.size(0), device=dev)
        for i in range(0, train.size(0), bs):
            xb = train[perm[i:i+bs]]
            with torch.no_grad(): enc = encoder(xb)          # encoder congelato (training NORMALE)
            dec = student(enc)
            loss = rd4ad_cosine_loss(enc, dec, detach_target=True)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 100 == 0: print(f"  epoch {ep+1}/{args.epochs}  loss {loss.item():.4f}")
    student.eval()
    # ===============================================================================================

    # anomalous test images and panels
    items = test_anomalous(args.category, args.data_root, args.img_size, args.mask_size, args.n)
    if not items:
        print("Nessuna immagine anomala con maschera trovata."); return
    for k, (fimg, fgt, defect) in enumerate(items):
        x = load_img(fimg, args.img_size).unsqueeze(0).to(dev)
        amap = anomaly_map(encoder, student, x, (args.mask_size, args.mask_size))[0]
        gt = load_mask(fgt, args.mask_size)
        # input image at mask_size for display
        disp = Image.open(fimg).convert("RGB").resize((args.mask_size, args.mask_size))
        amn = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
        ax[0].imshow(disp); ax[0].set_title(f"{args.category} / {defect}")
        ax[1].imshow(gt, cmap="gray"); ax[1].set_title("Ground truth")
        ax[2].imshow(disp); ax[2].imshow(amn, cmap="jet", alpha=0.5); ax[2].set_title("RD4AD anomaly map")
        for a in ax: a.axis("off")
        tag = "dist" if args.train_dir else "real"
        out = os.path.join(args.out, f"{args.category}_{defect}_{k}_{tag}.png")
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print("salvato", out)

if __name__ == "__main__":
    main()
