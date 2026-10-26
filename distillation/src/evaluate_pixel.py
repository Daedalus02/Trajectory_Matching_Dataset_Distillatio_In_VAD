#!/usr/bin/env python3
"""Train STFPM from scratch and report image/pixel AUROC and F1-max.

--train_dir: folder of images or a .pt of distilled images; default = real train/good.
"""

import argparse
import random
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.feature_matching import (
    DEFAULT_LAYERS,
    build_teacher,
    load_images_tensor,
    train_student_features,
    anomaly_scores,
)

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def split_4_patches(tensor):
    """
    Splitta un tensore [..., H, W] in 4 quadranti [..., H/2, W/2]
    nell'ordine: TL, TR, BL, BR.
    """
    H, W = tensor.shape[-2:]
    h2, w2 = H // 2, W // 2
    p1 = tensor[..., :h2, :w2]
    p2 = tensor[..., :h2, w2:]
    p3 = tensor[..., h2:, :w2]
    p4 = tensor[..., h2:, w2:]
    return [p1, p2, p3, p4]


def merge_4_patches(p1, p2, p3, p4):
    """
    Ricompatta 4 mappe 112x112 nella mappa originale 224x224.
    """
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
        # resize to patch_size if needed
        if imgs.shape[-1] != patch_size:
            imgs = F.interpolate(imgs, size=(patch_size, patch_size),
                                 mode="bilinear", align_corners=False)
        print("  Loaded {} images from tensor {} at {}x{}".format(
            len(imgs), train_dir_or_pt, patch_size, patch_size))
        return imgs.to(device)

    # Carica da cartella a 224 (se patch_split) o 112
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
    if args.seed is not None:  # paired evaluation: same student initialisation and batch order for every training set
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

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
    print("  Pixel-level evaluation — {}".format(args.category))
    print("  Train dir: {}".format(train_dir))
    print("  img_size: {}, epochs: {}, lr: {}".format(args.img_size, args.epochs, args.lr))
    print("=" * 60)

    native_res  = args.img_size                                   # 112 or 224
    student_res = native_res // 2 if args.patch_split else native_res
    mask_res    = args.mask_size                                  # ALWAYS 224 (native GT)

    teacher = build_teacher(backbone=args.backbone, layers=args.layers,
                            device=args.device, prefer_moviad=not args.no_moviad)

    images  = load_train_images(train_dir, student_res, args.device, patch_split=args.patch_split)
    student = train_student_features(
        images, teacher, args.device, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, backbone=args.backbone, layers=args.layers,
        momentum=args.momentum, weight_decay=args.wd, lr_decay=not args.no_lr_decay,
        prefer_moviad=not args.no_moviad, verbose=True,
    )
    student.eval(); teacher.eval()

    # test image fed at the run's native resolution; patch mode splits it into student_res quadrants
    tf = transforms.Compose([
        transforms.Resize((native_res, native_res), antialias=True),
        transforms.ToTensor(),
    ])
    test_base = Path(test_dir)

    img_scores, img_labels = [], []
    all_pixel_preds, all_pixel_targets = [], []

    with torch.no_grad():
        all_paths = [q for q in sorted(test_base.rglob("*.*")) if q.suffix.lower() in IMG_EXTENSIONS]
        if args.split != "all":  # fixed halves of the test set: alternate images within each defect type
            all_paths = [q for dt in sorted({q.parent.name for q in all_paths})
                         for q in [q for q in all_paths if q.parent.name == dt][(0 if args.split == "val" else 1)::2]]
        for test_path in all_paths:
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

            if args.patch_split:                       # img is native_res -> 4 quadrants of student_res
                patch_maps = [anomaly_scores(teacher, student, p)[0] for p in split_4_patches(img)]
                full_map = merge_4_patches(*patch_maps)                # back to native_res
            else:
                full_map = anomaly_scores(teacher, student, img)[0]    # native_res

            # image score on the NATIVE map (avoid interpolation smoothing the peak)
            img_scores.append(full_map.max().item())
            img_labels.append(img_lbl)

            # pixel score: bring the map to the GT's native resolution, then compare vs full mask
            map_up = F.interpolate(full_map, size=(mask_res, mask_res),
                                   mode="bilinear", align_corners=False)
            all_pixel_preds.extend(map_up.flatten().cpu().numpy())
            all_pixel_targets.extend((mask > 0).flatten().cpu().numpy().astype(int))

    img_auroc   = roc_auc_score(img_labels, img_scores)
    pixel_auroc = roc_auc_score(all_pixel_targets, all_pixel_preds)
    def _f1max(_l, _s):
        pr, rc, _ = precision_recall_curve(_l, _s)
        return float((2 * pr * rc / np.clip(pr + rc, 1e-12, None)).max())
    img_f1max = _f1max(img_labels, img_scores)
    pixel_f1max = _f1max(all_pixel_targets, all_pixel_preds)
    print("  Image-level F1max : {:.4f}".format(img_f1max))
    print("  Pixel-level F1max : {:.4f}".format(pixel_f1max))
    print(f"\n  Image-level AUROC (native {native_res}) : {img_auroc:.4f}")
    print(f"  Pixel-level AUROC (GT {mask_res})       : {pixel_auroc:.4f}")
    
    # Salvataggio
    with open(os.path.join(out_dir, "pixel_results.txt"), "w") as f:
        f.write(f"Image-level AUROC: {img_auroc:.4f}\n")
        f.write(f"Pixel-level AUROC: {pixel_auroc:.4f}\n")
    print(f"\n  Results saved to {os.path.join(out_dir, 'pixel_results.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--train_dir", default=None,
                        help="folder of images or .pt tensor; default: all real train/good images")
    parser.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    parser.add_argument("--out_dir", default="../outputs/pixel")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--no_lr_decay", action="store_true", help="constant lr (expert regime)")
    parser.add_argument("--seed", type=int, default=None, help="fix student init and batch order (paired evaluation)")
    parser.add_argument("--split", default="all", choices=["all", "val", "test"],
                        help="half of the test set: val = even images of each defect type, test = odd")
    parser.add_argument("--no_moviad", action="store_true")
    parser.add_argument("--img_size", type=int, default=112,
                    help="Native/operating resolution of the run (112 full, 224 patch or full-224)")
    parser.add_argument("--mask_size", type=int, default=224,
                    help="Native GT-mask resolution; the anomaly map is upsampled to this before scoring")
    parser.add_argument("--patch_split", action="store_true", help="split the images into four 112x112 quadrants")
    args = parser.parse_args()
    main(args)