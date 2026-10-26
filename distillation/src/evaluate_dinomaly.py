#!/usr/bin/env python3
"""Train Dinomaly from scratch on real or distilled images and report the image-level AUROC.

Used by the Dinomaly-at-64 ablation. Example:
    python evaluate_dinomaly.py --category bottle --train_dir <images.pt> --epochs 200
"""

import argparse
import os
import sys
import time
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.dinomaly_matching import (
    BACKBONE_CHOICES,
    DEFAULT_BACKBONE,
    DEFAULT_BLUR_SIGMA,
    DEFAULT_HM_FACTOR,
    DEFAULT_HM_P,
    DEFAULT_RES,
    build_encoder,
    check_res,
    evaluate_auroc,
    gpu_peak_report,
    load_images_tensor,
    train_dinomaly,
)

warnings.filterwarnings("ignore", message="xFormers")


def load_train_images(train_dir_or_pt, img_size):
    """Cartella di immagini oppure file .pt di immagini distillate."""
    if str(train_dir_or_pt).endswith(".pt"):
        images = torch.load(train_dir_or_pt, map_location="cpu").clamp(0, 1)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] != img_size:
            images = F.interpolate(
                images, size=(img_size, img_size), mode="bilinear", align_corners=False
            )
        print("  Loaded {} images from tensor {} at {}x{}".format(
            len(images), train_dir_or_pt, img_size, img_size))
        return images
    return load_images_tensor(train_dir_or_pt, img_size=img_size)


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    check_res(args.img_size, args.backbone)
    t_start = time.time()

    # ── Percorsi ──
    if args.train_dir:
        train_dir = args.train_dir
    else:
        train_dir = os.path.join(args.data_path, args.category, "train", "good")
    test_dir = os.path.join(args.data_path, args.category, "test")

    out_dir = os.path.join(args.out_dir, args.category)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Dinomaly evaluation — {}".format(args.category))
    print("  Train dir: {}".format(train_dir))
    print("  lr={}, epochs={}, optimizer=sgd, backbone={}".format(args.lr, args.epochs, args.backbone))
    print("=" * 60)

    # frozen encoder (target and decoder input)
    encoder = build_encoder(backbone=args.backbone, device=args.device, verbose=True)

    # training images (real or distilled)
    images = load_train_images(train_dir, args.img_size)

    # ── Allena bottleneck + decoder da zero ──
    student = train_dinomaly(
        images, encoder, args.device,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, momentum=args.momentum,
        backbone=args.backbone, p=args.hm_p, factor=args.hm_factor, verbose=True,
    )

    # ── Valutazione AUROC ──
    result = evaluate_auroc(
        student, encoder, test_dir,
        img_size=args.img_size, device=args.device,
        batch_size=args.eval_batch_size, blur_sigma=args.blur_sigma, verbose=True,
    )
    auroc = result["auroc"]

    # ── Salva risultati ──
    with open(os.path.join(out_dir, "auroc_results.txt"), "w") as f:
        f.write("Category: {}\n".format(args.category))
        f.write("AUROC: {:.4f}\n".format(auroc))
        f.write("Train dir: {}\n".format(train_dir))
        f.write("Epochs: {}\n".format(args.epochs))
        f.write("LR: {}\n".format(args.lr))
        f.write("Optimizer: sgd\n")
        f.write("Model: Dinomaly (backbone={}, blur_sigma={})\n".format(args.backbone, args.blur_sigma))
        f.write("Img size: {}\n".format(args.img_size))
        f.write("Train images: {}\n".format(len(images)))

    print("\n  Image-level AUROC {}: {:.4f} (salvato in {}/auroc_results.txt)".format(
        args.category, auroc, out_dir))
    print("  " + gpu_peak_report(args.device))
    print("  Evaluation time: {:.1f} s".format(time.time() - t_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="Categoria MVTec (es. bottle)")
    parser.add_argument("--train_dir", default=None,
                        help="Override cartella training o file .pt (immagini distillate)")
    parser.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    parser.add_argument("--out_dir", default="../outputs/dinomaly")
    # architettura
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, choices=BACKBONE_CHOICES)
    parser.add_argument("--blur_sigma", type=float, default=DEFAULT_BLUR_SIGMA,
                        help="Smoothing gaussiano della anomaly map (0 = disattivato)")
    parser.add_argument("--hm_p", type=float, default=DEFAULT_HM_P)
    parser.add_argument("--hm_factor", type=float, default=DEFAULT_HM_FACTOR)
    # training
    parser.add_argument("--img_size", type=int, default=DEFAULT_RES)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
