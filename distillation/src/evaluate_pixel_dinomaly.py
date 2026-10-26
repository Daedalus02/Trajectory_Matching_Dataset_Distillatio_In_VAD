#!/usr/bin/env python3
"""Train Dinomaly from scratch and report image/pixel AUROC and F1-max.

--train_dir: folder of images or a .pt of distilled images; default = real train/good.
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.dinomaly_matching import (
    BACKBONE_CHOICES,
    DEFAULT_BACKBONE,
    DEFAULT_BLUR_SIGMA,
    DEFAULT_HM_FACTOR,
    DEFAULT_HM_P,
    DEFAULT_RES,
    IMG_EXTENSIONS,
    anomaly_scores,
    build_encoder,
    check_res,
    gpu_peak_report,
    train_dinomaly,
)
from evaluate_dinomaly import load_train_images

warnings.filterwarnings("ignore", message="xFormers")


def load_mask(mask_path, img_size):
    """Carica una maschera binaria e la porta a img_size (0/1)."""
    m = Image.open(mask_path).convert("L").resize((img_size, img_size), Image.NEAREST)
    m = np.array(m)
    return (m > 127).astype(np.uint8)


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    check_res(args.img_size, args.backbone)
    t_start = time.time()

    if args.train_dir:
        train_dir = args.train_dir
    else:
        train_dir = os.path.join(args.data_path, args.category, "train", "good")

    test_dir = os.path.join(args.data_path, args.category, "test")
    gt_dir = os.path.join(args.data_path, args.category, "ground_truth")
    out_dir = os.path.join(args.out_dir, args.category, f"full{args.img_size}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Pixel-level evaluation (Dinomaly) — {}".format(args.category))
    print("  Train dir: {}".format(train_dir))
    print("  img_size: {}, epochs: {}, lr: {}, optimizer: sgd, backbone: {}".format(
        args.img_size, args.epochs, args.lr, args.backbone))
    print("=" * 60)

    native_res = args.img_size                                   # 224
    mask_res = args.mask_size                                    # native GT resolution

    encoder = build_encoder(backbone=args.backbone, device=args.device, verbose=True)
    images = load_train_images(train_dir, native_res)
    student = train_dinomaly(
        images, encoder, args.device, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, momentum=args.momentum, backbone=args.backbone,
        p=args.hm_p, factor=args.hm_factor, verbose=True,
    )
    student.eval()

    tf = transforms.Compose([
        transforms.Resize((native_res, native_res), antialias=True),
        transforms.ToTensor(),
    ])
    test_base = Path(test_dir)

    img_scores, img_labels = [], []
    all_pixel_preds, all_pixel_targets = [], []

    with torch.no_grad():
        for test_path in sorted(test_base.rglob("*.*")):
            if test_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            defect_type = test_path.parent.name
            is_good = (defect_type == "good")
            img_lbl = 0 if is_good else 1

            img = tf(Image.open(test_path).convert("RGB")).unsqueeze(0).to(args.device)

            # GT mask ALWAYS at native mask_res (no downscaling to the training resolution)
            if is_good:
                mask = torch.zeros((1, 1, mask_res, mask_res), device=args.device)
            else:
                mask_file = os.path.join(gt_dir, defect_type, test_path.stem + "_mask.png")
                if not os.path.exists(mask_file):
                    mask_file = os.path.join(gt_dir, defect_type, test_path.name)
                m_np = load_mask(mask_file, mask_res)
                mask = torch.tensor(m_np, dtype=torch.float32, device=args.device)[None, None]

            full_map, _ = anomaly_scores(encoder, student, img, blur_sigma=args.blur_sigma)

            # image-level score = peak of the anomaly map (as in STFPM / RD4AD)
            img_scores.append(full_map.max().item())
            img_labels.append(img_lbl)

            # pixel score: bring the map to the GT's native resolution, then compare vs full mask
            map_up = F.interpolate(full_map, size=(mask_res, mask_res),
                                   mode="bilinear", align_corners=False)
            all_pixel_preds.extend(map_up.flatten().cpu().numpy())
            all_pixel_targets.extend((mask > 0).flatten().cpu().numpy().astype(int))

    img_auroc = roc_auc_score(img_labels, img_scores)
    pixel_auroc = roc_auc_score(all_pixel_targets, all_pixel_preds)
    def _f1max(_l, _s):
        pr, rc, _ = precision_recall_curve(_l, _s)
        return float((2 * pr * rc / np.clip(pr + rc, 1e-12, None)).max())
    img_f1max = _f1max(img_labels, img_scores)
    pixel_f1max = _f1max(all_pixel_targets, all_pixel_preds)
    print("  Image-level F1max : {:.4f}".format(img_f1max))
    print("  Pixel-level F1max : {:.4f}".format(pixel_f1max))
    print(f"\n  Image-level AUROC (native {native_res}, peak of the map) : {img_auroc:.4f}")
    print(f"  Pixel-level AUROC (GT {mask_res})                      : {pixel_auroc:.4f}")

    # Salvataggio
    with open(os.path.join(out_dir, "pixel_results.txt"), "w") as f:
        f.write(f"Image-level AUROC: {img_auroc:.4f}\n")
        f.write(f"Pixel-level AUROC: {pixel_auroc:.4f}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Blur sigma: {args.blur_sigma}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Train dir: {train_dir}\n")
    print(f"\n  Risultati salvati in {os.path.join(out_dir, 'pixel_results.txt')}")
    print("  " + gpu_peak_report(args.device))
    print("  Evaluation time: {:.1f} s".format(time.time() - t_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--train_dir", default=None,
                        help="Cartella di immagini oppure file .pt")
    parser.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    parser.add_argument("--out_dir", default="../outputs/pixel_dinomaly")
    # architettura
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, choices=BACKBONE_CHOICES)
    parser.add_argument("--blur_sigma", type=float, default=DEFAULT_BLUR_SIGMA,
                        help="Smoothing gaussiano della anomaly map (0 = disattivato)")
    parser.add_argument("--hm_p", type=float, default=DEFAULT_HM_P)
    parser.add_argument("--hm_factor", type=float, default=DEFAULT_HM_FACTOR)
    # training
    parser.add_argument("--epochs", type=int, default=200,
                        help="il decoder ViT, come RD4AD, probabilmente ne vuole di piu'")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--img_size", type=int, default=DEFAULT_RES,
                        help="Native/operating resolution of the run (multiple of 14)")
    parser.add_argument("--mask_size", type=int, default=224,
                        help="Native GT-mask resolution; the anomaly map is upsampled to this")
    args = parser.parse_args()
    main(args)
