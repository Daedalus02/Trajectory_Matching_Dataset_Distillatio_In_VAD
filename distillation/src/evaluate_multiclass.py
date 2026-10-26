#!/usr/bin/env python3
"""Multi-class evaluation for RD4AD and Dinomaly: one model on the pool of all categories.

--type distilled: pool of the 15 distilled sets; --type real: all train/good images.
Appends per-category image/pixel AUROC and F1-max to --out_csv.
"""

import argparse
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

warnings.filterwarnings("ignore", message="xFormers")

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
CSV_HEADER = "category,detector,res,type,run,image_auroc,pixel_auroc,image_f1max,pixel_f1max"


def load_detector(name):
    """Modulo lib e funzione di training del detector scelto."""
    if name == "rd4ad":
        from lib import rd4ad_matching as lib
        return lib, lib.train_rd4ad
    from lib import dinomaly_matching as lib
    return lib, lib.train_dinomaly


def load_mask(mask_path, img_size):
    """Carica una maschera binaria e la porta a img_size (0/1)."""
    m = Image.open(mask_path).convert("L").resize((img_size, img_size), Image.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)


def load_pt(path, res):
    imgs = torch.load(path, map_location="cpu").clamp(0, 1)
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[-1] != res:
        imgs = F.interpolate(imgs, size=(res, res), mode="bilinear", align_corners=False)
    return imgs


def build_pool(args, lib):
    """Unione dei set di training delle 15 categorie (distillati o reali)."""
    pool = []
    for cat in args.categories:
        if args.type == "distilled":
            path = args.pt_template.format(det=args.detector, cat=cat, res=args.res)
            imgs = load_pt(path, args.res)
            print("  [pool] {:<11s} {:4d} distilled from {}".format(cat, len(imgs), path))
        else:
            root = os.path.join(args.data_path, cat, "train", "good")
            imgs = lib.load_images_tensor(root, img_size=args.res, verbose=False)
            print("  [pool] {:<11s} {:4d} real from {}".format(cat, len(imgs), root))
        pool.append(imgs)
    pool = torch.cat(pool, dim=0)
    print("  [pool] total {} images, shape {}".format(len(pool), tuple(pool.shape)))
    return pool


def f1_max(labels, scores):
    """F1 alla soglia ottima (F1-max)."""
    prec, rec, _ = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(f1.max())


@torch.no_grad()
def evaluate_category(args, lib, encoder, student, cat):
    test_base = Path(args.data_path) / cat / "test"
    gt_dir = os.path.join(args.data_path, cat, "ground_truth")
    mask_res = args.mask_size
    tf = transforms.Compose([
        transforms.Resize((args.res, args.res), antialias=True),
        transforms.ToTensor(),
    ])

    img_scores, img_labels, pix_preds, pix_targets = [], [], [], []
    for test_path in sorted(test_base.rglob("*.*")):
        if test_path.suffix.lower() not in lib.IMG_EXTENSIONS:
            continue
        defect_type = test_path.parent.name
        is_good = (defect_type == "good")

        # batch of 1 as in the per-category scripts (BN uses batch statistics)
        img = tf(Image.open(test_path).convert("RGB")).unsqueeze(0).to(args.device)

        if is_good:
            mask = np.zeros((mask_res, mask_res), dtype=np.uint8)
        else:
            mask_file = os.path.join(gt_dir, defect_type, test_path.stem + "_mask.png")
            if not os.path.exists(mask_file):
                mask_file = os.path.join(gt_dir, defect_type, test_path.name)
            mask = load_mask(mask_file, mask_res)

        full_map, _ = lib.anomaly_scores(encoder, student, img, blur_sigma=args.blur_sigma)
        img_scores.append(full_map.max().item())
        img_labels.append(0 if is_good else 1)

        map_up = F.interpolate(full_map, size=(mask_res, mask_res),
                               mode="bilinear", align_corners=False)
        pix_preds.append(map_up.flatten().cpu().numpy().astype(np.float32))
        pix_targets.append(mask.flatten())

    pix_preds = np.concatenate(pix_preds)
    pix_targets = np.concatenate(pix_targets)
    return {
        "image_auroc": roc_auc_score(img_labels, img_scores),
        "pixel_auroc": roc_auc_score(pix_targets, pix_preds),
        "image_f1max": f1_max(img_labels, img_scores),
        "pixel_f1max": f1_max(pix_targets, pix_preds),
    }


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.run)
    np.random.seed(args.run)
    t_start = time.time()
    lib, train_fn = load_detector(args.detector)

    # same defaults as the per-category runs
    if args.epochs is None:
        args.epochs = 800 if args.detector == "rd4ad" else 200
    if args.batch_size is None:
        args.batch_size = 32 if args.detector == "rd4ad" else 16
    if args.blur_sigma is None:
        args.blur_sigma = lib.DEFAULT_BLUR_SIGMA
    if args.detector == "dinomaly":
        lib.check_res(args.res, lib.DEFAULT_BACKBONE)
    detector_tag = {"rd4ad": "RD4AD_MC", "dinomaly": "DINOMALY_MC"}[args.detector]

    print("=" * 60)
    print("  Multi-class evaluation ({}) — type={} res={} run={}".format(
        detector_tag, args.type, args.res, args.run))
    print("=" * 60)

    if args.detector == "rd4ad":
        encoder = lib.build_encoder(backbone=lib.DEFAULT_BACKBONE, device=args.device,
                                    pretrained=True, verbose=True)
    else:
        encoder = lib.build_encoder(backbone=lib.DEFAULT_BACKBONE, device=args.device,
                                    verbose=True)

    pool = build_pool(args, lib)
    steps = args.epochs * math.ceil(len(pool) / max(1, min(args.batch_size, len(pool))))
    print("  epochs={} batch={} lr={} -> {} optimisation steps".format(
        args.epochs, args.batch_size, args.lr, steps))

    t_train = time.time()
    student = train_fn(pool, encoder, args.device, epochs=args.epochs, lr=args.lr,
                       batch_size=args.batch_size, verbose=True)
    student.eval()
    print("  Training time: {:.1f} min".format((time.time() - t_train) / 60))

    if args.save_model:
        ckpt_dir = os.path.join(os.path.dirname(args.out_csv), "models")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = os.path.join(ckpt_dir, "{}_{}_{}_r{}.pth".format(
            args.detector, args.type, args.res, args.run))
        torch.save(student.state_dict(), ckpt)
        print("  Model saved to {}".format(ckpt))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    if not os.path.exists(args.out_csv):
        with open(args.out_csv, "w") as f:
            f.write(CSV_HEADER + "\n")

    for cat in args.categories:
        r = evaluate_category(args, lib, encoder, student, cat)
        print("  {:<11s} Image-AUROC {:.4f}  Pixel-AUROC {:.4f}  Image-F1max {:.4f}  "
              "Pixel-F1max {:.4f}".format(cat, r["image_auroc"], r["pixel_auroc"],
                                          r["image_f1max"], r["pixel_f1max"]))
        # append row by row
        with open(args.out_csv, "a") as f:
            f.write("{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                cat, detector_tag, args.res, args.type, args.run, r["image_auroc"],
                r["pixel_auroc"], r["image_f1max"], r["pixel_f1max"]))

    print("\n  Results appended to {}".format(args.out_csv))
    if hasattr(lib, "gpu_peak_report"):
        print("  " + lib.gpu_peak_report(args.device))
    print("  Total time: {:.1f} min".format((time.time() - t_start) / 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True, choices=["rd4ad", "dinomaly"])
    parser.add_argument("--type", required=True, choices=["distilled", "real"])
    parser.add_argument("--res", type=int, required=True,
                        help="64 / 128 per RD4AD, 224 per Dinomaly")
    parser.add_argument("--run", type=int, default=0, help="indice del run (anche seed)")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    parser.add_argument("--pt_template",
                        default="../outputs/thesis_sweep/images/{det}_{cat}_{res}.pt",
                        help="set distillati; segnaposto {det} {cat} {res}")
    parser.add_argument("--out_csv", default="../outputs/multiclass/results_multiclass_3det.csv")
    parser.add_argument("--save_model", action="store_true")
    # training (None = per-category default of the detector)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--blur_sigma", type=float, default=None)
    parser.add_argument("--mask_size", type=int, default=224)
    args = parser.parse_args()
    main(args)
