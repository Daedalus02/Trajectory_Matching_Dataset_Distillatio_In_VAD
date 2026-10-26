#!/usr/bin/env python3
"""Qualitative Dinomaly figure: [input | ground truth | map (distilled) | map (real)].
"""
import os, glob, argparse, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import sys; sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from lib.dinomaly_matching import (
    DEFAULT_BACKBONE, DEFAULT_RES, DEFAULT_HM_P, DEFAULT_HM_FACTOR, DEFAULT_BLUR_SIGMA,
    build_encoder, load_images_tensor, train_dinomaly, anomaly_scores,
)

# ---------------------------------------------------------------- dati (input in [0,1])
def load_train_images(path, img_size):
    """Cartella di immagini reali oppure file .pt di immagini distillate (come evaluate_dinomaly.py)."""
    if str(path).endswith(".pt"):
        im = torch.load(path, map_location="cpu").clamp(0, 1)
        if im.dim() == 3: im = im.unsqueeze(0)
        if im.shape[-1] != img_size:
            im = F.interpolate(im, size=(img_size, img_size), mode="bilinear", align_corners=False)
        print(f"  Loaded {len(im)} images from tensor {path} at {img_size}x{img_size}")
        return im
    return load_images_tensor(path, img_size=img_size)

def load_img(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1).clamp(0, 1)

def load_mask(path, size):
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 0).astype(np.float32)

def test_anomalous(cat, root, n=4):
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

def train_one(images, encoder, dev, args):
    return train_dinomaly(
        images, encoder, dev,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, momentum=args.momentum,
        backbone=args.backbone, p=args.hm_p, factor=args.hm_factor, verbose=True,
    )

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--data_root", default="../data/mvtec_anomaly_detection")
    ap.add_argument("--img_size", type=int, default=DEFAULT_RES)       # 224
    ap.add_argument("--mask_size", type=int, default=DEFAULT_RES)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--hm_p", type=float, default=DEFAULT_HM_P)
    ap.add_argument("--hm_factor", type=float, default=DEFAULT_HM_FACTOR)
    ap.add_argument("--blur_sigma", type=float, default=DEFAULT_BLUR_SIGMA)
    ap.add_argument("--train_dir", required=True, help=".pt distillato full224")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="../outputs/qual_dinomaly")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device); os.makedirs(args.out, exist_ok=True)

    # shared frozen encoder
    encoder = build_encoder(backbone=args.backbone, device=dev, verbose=True)

    real_dir = os.path.join(args.data_root, args.category, "train", "good")
    real = load_train_images(real_dir, args.img_size)
    dist = load_train_images(args.train_dir, args.img_size)
    print(f"real {tuple(real.shape)} | dist {tuple(dist.shape)}")

    print("== training REAL =="); s_r = train_one(real, encoder, dev, args)
    print("== training DIST =="); s_d = train_one(dist, encoder, dev, args)

    items = test_anomalous(args.category, args.data_root, args.n)
    if not items:
        print("Nessuna immagine anomala con maschera trovata."); return
    out_shape = (args.mask_size, args.mask_size)
    for k, (fimg, fgt, defect) in enumerate(items):
        x = load_img(fimg, args.img_size).unsqueeze(0).to(dev)
        amap_d, _ = anomaly_scores(encoder, s_d, x, out_shape=out_shape, blur_sigma=args.blur_sigma)
        amap_r, _ = anomaly_scores(encoder, s_r, x, out_shape=out_shape, blur_sigma=args.blur_sigma)
        a_d = amap_d.squeeze(1).cpu().numpy()[0]
        a_r = amap_r.squeeze(1).cpu().numpy()[0]
        gt = load_mask(fgt, args.mask_size)
        disp = Image.open(fimg).convert("RGB").resize(out_shape)
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