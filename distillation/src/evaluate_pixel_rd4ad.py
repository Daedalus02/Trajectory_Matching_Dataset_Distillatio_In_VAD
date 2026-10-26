#!/usr/bin/env python3
"""Train RD4AD (bn + decoder) from scratch and report image/pixel AUROC and F1-max.

--train_dir: folder of images or a .pt of distilled images; default = real train/good.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.rd4ad_matching import (
    DEFAULT_BACKBONE,
    DEFAULT_BLUR_SIGMA,
    IMG_EXTENSIONS,
    anomaly_scores,
    build_encoder,
    load_images_tensor,
    train_rd4ad,
)


def split_4_patches(tensor):
    """Splitta un tensore [..., H, W] in 4 quadranti [..., H/2, W/2] (TL, TR, BL, BR)."""
    H, W = tensor.shape[-2:]
    h2, w2 = H // 2, W // 2
    return [
        tensor[..., :h2, :w2],
        tensor[..., :h2, w2:],
        tensor[..., h2:, :w2],
        tensor[..., h2:, w2:],
    ]


def merge_4_patches(p1, p2, p3, p4):
    """Ricompatta 4 mappe nella mappa originale."""
    top = torch.cat([p1, p2], dim=-1)
    bottom = torch.cat([p3, p4], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def load_mask(mask_path, img_size):
    """Carica una maschera binaria e la porta a img_size (0/1)."""
    m = Image.open(mask_path).convert("L").resize((img_size, img_size), Image.NEAREST)
    m = np.array(m)
    return (m > 127).astype(np.uint8)


def load_train_images(train_dir_or_pt, patch_size, device, patch_split=False):
    if str(train_dir_or_pt).endswith(".pt"):
        imgs = torch.load(train_dir_or_pt, map_location="cpu").clamp(0, 1)
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.shape[-1] != patch_size:
            imgs = F.interpolate(imgs, size=(patch_size, patch_size),
                                 mode="bilinear", align_corners=False)
        print("  Loaded {} images from tensor {} at {}x{}".format(
            len(imgs), train_dir_or_pt, patch_size, patch_size))
        return imgs.to(device)

    load_res = patch_size * 2 if patch_split else patch_size
    imgs = load_images_tensor(train_dir_or_pt, img_size=load_res)

    if patch_split:
        h_mid, w_mid = imgs.shape[-2] // 2, imgs.shape[-1] // 2
        p1 = imgs[:, :, :h_mid, :w_mid]
        p2 = imgs[:, :, :h_mid, w_mid:]
        p3 = imgs[:, :, h_mid:, :w_mid]
        p4 = imgs[:, :, h_mid:, w_mid:]
        imgs = torch.cat([p1, p2, p3, p4], dim=0)

    return imgs.to(device)


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.train_dir:
        train_dir = args.train_dir
    else:
        train_dir = os.path.join(args.data_path, args.category, "train", "good")

    test_dir = os.path.join(args.data_path, args.category, "test")
    gt_dir = os.path.join(args.data_path, args.category, "ground_truth")
    tag = f"{'patch' if args.patch_split else 'full'}{args.img_size}"
    out_dir = os.path.join(args.out_dir, args.category, tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Pixel-level evaluation (RD4AD) — {}".format(args.category))
    print("  Train dir: {}".format(train_dir))
    print("  img_size: {}, epochs: {}, lr: {}, optimizer: {}".format(
        args.img_size, args.epochs, args.lr, args.optimizer))
    print("=" * 60)

    native_res = args.img_size                                   # 64, 112 or 224
    model_res = native_res // 2 if args.patch_split else native_res
    mask_res = args.mask_size                                    # native GT resolution

    encoder = build_encoder(
        backbone=args.backbone, device=args.device, pretrained=True, verbose=True
    )
    images = load_train_images(train_dir, model_res, args.device, patch_split=args.patch_split)
    student = train_rd4ad(
        images, encoder, args.device, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, optimizer=args.optimizer,
        backbone=args.backbone, verbose=True,
    )
    student.eval()

    # test image fed at the run's native resolution; patch mode splits it into model_res quadrants
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

            if args.patch_split:              # img is native_res -> 4 quadrants of model_res
                patch_out = [
                    anomaly_scores(encoder, student, p, blur_sigma=args.blur_sigma)
                    for p in split_4_patches(img)
                ]
                full_map = merge_4_patches(*[o[0] for o in patch_out])       # back to native_res
            else:
                full_map, _ = anomaly_scores(
                    encoder, student, img, blur_sigma=args.blur_sigma
                )                                                            # native_res

            # in RD4AD the image-level score IS the peak of the anomaly map (as in STFPM)
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
        f.write(f"Train dir: {train_dir}\n")
    print(f"\n  Risultati salvati in {os.path.join(out_dir, 'pixel_results.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--train_dir", default=None,
                        help="Cartella di immagini oppure file .pt")
    parser.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    parser.add_argument("--out_dir", default="../outputs/pixel_rd4ad")
    # architettura
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        choices=["resnet18", "resnet34", "resnet50", "wide_resnet50_2"])
    parser.add_argument("--blur_sigma", type=float, default=DEFAULT_BLUR_SIGMA,
                        help="Smoothing gaussiano della anomaly map (0 = disattivato)")
    # training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--img_size", type=int, default=64,
                        help="Native/operating resolution of the run")
    parser.add_argument("--mask_size", type=int, default=224,
                        help="Native GT-mask resolution; the anomaly map is upsampled to this")
    parser.add_argument("--patch_split", action="store_true",
                        help="Abilita divisione in 4 patch")
    args = parser.parse_args()
    main(args)
