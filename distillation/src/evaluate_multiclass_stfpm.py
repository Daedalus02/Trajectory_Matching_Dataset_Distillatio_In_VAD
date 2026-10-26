#!/usr/bin/env python3
"""Multi-class evaluation for STFPM: one student on the pool of all categories.

Same protocol as evaluate_pixel.py, evaluated per category. Appends rows to --csv.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.feature_matching import (
    DEFAULT_LAYERS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_teacher,
    load_images_tensor,
    train_student_features,
    anomaly_scores,
)

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

# candidate paths of a distilled set (first existing one is used)
DEFAULT_DIST_TEMPLATES = [
    "../outputs/distilled_stfpm/{cat}/{ipc}/real/full{res}/images_best.pt",
    "../outputs/thesis_sweep/images/{cat}/{ipc}/real/full{res}/images_best.pt",
    "../outputs/thesis_sweep/images/{cat}_{ipc}_full{res}.pt",
    "../outputs/distilled/{cat}/{ipc}/real/full{res}/images_best.pt",
]

CSV_FIELDS = ["category", "detector", "res", "type", "run", "image_auroc", "pixel_auroc"]
DETECTOR = "STFPM_MC"


def load_mask(mask_path, img_size):
    """Carica una maschera binaria e la porta a img_size (0/1). [verbatim da evaluate_pixel.py]"""
    m = Image.open(mask_path).convert("L").resize((img_size, img_size), Image.NEAREST)
    m = np.array(m)
    return (m > 127).astype(np.uint8)


def find_distilled(cat, args):
    templates = [args.dist_template] if args.dist_template else DEFAULT_DIST_TEMPLATES
    for t in templates:
        p = t.format(cat=cat, ipc=args.ipc, res=args.res)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Distillato non trovato per '{}' (ipc={}, res={}). Provati:\n  {}".format(
            cat, args.ipc, args.res,
            "\n  ".join(t.format(cat=cat, ipc=args.ipc, res=args.res) for t in templates)))


def load_distilled_pool(args):
    """Concatena i distillati di tutte le categorie -> [N_cat*IPC, 3, res, res] su CPU, in [0,1]."""
    pool = []
    for cat in args.categories:
        path = find_distilled(cat, args)
        # same loading as evaluate_pixel.py
        imgs = torch.load(path, map_location="cpu").clamp(0, 1)
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.shape[-1] != args.res:
            imgs = F.interpolate(imgs, size=(args.res, args.res),
                                 mode="bilinear", align_corners=False)
        if len(imgs) != args.ipc:
            print("  [WARN] {}: {} immagini nel .pt, attese ipc={}".format(cat, len(imgs), args.ipc))
        print("  {:<11s} {} imgs  <- {}".format(cat, len(imgs), path))
        pool.append(imgs)
    pool = torch.cat(pool, dim=0)
    print("  Pool distillato: {}".format(tuple(pool.shape)))
    return pool


def load_real_pool(args):
    """Pool del train/good reale di tutte le categorie a res x res, in [0,1]."""
    pool = []
    for cat in args.categories:
        d = os.path.join(args.data_path, cat, "train", "good")
        pool.append(load_images_tensor(d, img_size=args.res, verbose=False))
        print("  {:<11s} {} imgs  <- {}".format(cat, len(pool[-1]), d))
    pool = torch.cat(pool, dim=0)
    print("  Pool reale: {}".format(tuple(pool.shape)))
    return pool


def evaluate_category(teacher, student, cat, args, norm):
    """Loop di eval di evaluate_pixel.py (ramo non-patch), per una categoria."""
    test_dir = os.path.join(args.data_path, cat, "test")
    gt_dir = os.path.join(args.data_path, cat, "ground_truth")
    native_res = args.res
    mask_res = args.mask_size

    ops = [transforms.Resize((native_res, native_res), antialias=True), transforms.ToTensor()]
    if norm is not None:
        ops.append(norm)
    tf = transforms.Compose(ops)

    img_scores, img_labels = [], []
    all_pixel_preds, all_pixel_targets = [], []

    with torch.no_grad():
        for test_path in sorted(Path(test_dir).rglob("*.*")):
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

            full_map = anomaly_scores(teacher, student, img)[0]    # native_res

            # image score on the NATIVE map (avoid interpolation smoothing the peak)
            img_scores.append(full_map.max().item())
            img_labels.append(img_lbl)

            # pixel score: bring the map to the GT's native resolution, then compare vs full mask
            map_up = F.interpolate(full_map, size=(mask_res, mask_res),
                                   mode="bilinear", align_corners=False)
            all_pixel_preds.append(map_up.flatten().cpu().numpy())
            all_pixel_targets.append((mask > 0).flatten().cpu().numpy().astype(int))

    if not img_labels:
        raise RuntimeError("Nessuna immagine di test in {}".format(test_dir))
    img_auroc = roc_auc_score(img_labels, img_scores)
    pixel_auroc = roc_auc_score(np.concatenate(all_pixel_targets), np.concatenate(all_pixel_preds))
    print(f"\n  [{cat}]")
    print(f"  Image-level AUROC (native {native_res}) : {img_auroc:.4f}")
    print(f"  Pixel-level AUROC (GT {mask_res})       : {pixel_auroc:.4f}")
    return img_auroc, pixel_auroc


def append_row(csv_path, row):
    new = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) if args.normalize else None

    print("=" * 60)
    print("  STFPM multi-class - {} categorie, res={}, ipc={}, epochs={}, runs={}".format(
        len(args.categories), args.res, args.ipc, args.epochs, args.runs))
    print("  types={}, normalize={}, device={}".format(args.types, args.normalize, args.device))
    print("  CSV: {}".format(args.csv))
    print("=" * 60)

    teacher = build_teacher(backbone=args.backbone, layers=args.layers,
                            device=args.device, prefer_moviad=not args.no_moviad)

    t_start = time.time()
    for typ in args.types:
        print("\n### type = {}".format(typ))
        pool = load_distilled_pool(args) if typ == "distilled" else load_real_pool(args)
        if norm is not None:
            pool = norm(pool)

        for run in range(args.runs):
            torch.manual_seed(args.seed + run)
            np.random.seed(args.seed + run)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            print("\n--- {} run {}/{} ---".format(typ, run + 1, args.runs))

            # one student on the pool of all categories
            student = train_student_features(
                pool.to(args.device), teacher, args.device, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, backbone=args.backbone, layers=args.layers,
                prefer_moviad=not args.no_moviad, verbose=True,
            )
            student.eval(); teacher.eval()
            t_train = time.time() - t0

            img_list, pix_list = [], []
            for cat in args.categories:
                img_auroc, pixel_auroc = evaluate_category(teacher, student, cat, args, norm)
                img_list.append(img_auroc); pix_list.append(pixel_auroc)
                append_row(args.csv, {
                    "category": cat, "detector": DETECTOR, "res": args.res, "type": typ,
                    "run": run, "image_auroc": f"{img_auroc:.4f}", "pixel_auroc": f"{pixel_auroc:.4f}",
                })

            peak = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else float("nan")
            print("\n  === {} run {} - MACRO ({} cat) ===".format(typ, run, len(args.categories)))
            print(f"  Image-level AUROC (macro) : {np.mean(img_list):.4f}")
            print(f"  Pixel-level AUROC (macro) : {np.mean(pix_list):.4f}")
            print("  train {:.1f}s | train+eval {:.1f}s | peak GPU {:.2f} GiB".format(
                t_train, time.time() - t0, peak))
            del student
            if args.device == "cuda":
                torch.cuda.empty_cache()

    print("\nTotale: {:.1f}s - righe appese a {}".format(time.time() - t_start, args.csv))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--types", nargs="+", default=["distilled", "real"],
                        choices=["distilled", "real"])
    parser.add_argument("--categories", nargs="+", default=MVTEC_CATEGORIES)
    parser.add_argument("--dist_template", default=None,
                        help="Path del .pt con {cat},{ipc},{res}; default: prova DEFAULT_DIST_TEMPLATES")
    parser.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    parser.add_argument("--csv", default="../outputs/thesis_sweep/results_multiclass.csv")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--mask_size", type=int, default=224,
                        help="Native GT-mask resolution; the anomaly map is upsampled to this before scoring")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalizzazione ImageNet su train e test (deve combaciare con lo sweep)")
    parser.add_argument("--no_moviad", action="store_true")
    args = parser.parse_args()
    main(args)
