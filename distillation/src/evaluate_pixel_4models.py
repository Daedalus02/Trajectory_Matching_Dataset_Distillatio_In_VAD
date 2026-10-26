#!/usr/bin/env python3
"""Patch-split evaluation: four students, one per 112x112 quadrant.

The four anomaly maps are tiled back into a 224 map before computing the metrics.
--norm optionally normalises each map with statistics of the real train/good images.
"""

import argparse
import os
import sys
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
    build_teacher,
    train_student_features,
    anomaly_scores,
)

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ordine quadranti: 0=TL, 1=TR, 2=BL, 3=BR
def split_4(t):
    H, W = t.shape[-2:]
    h2, w2 = H // 2, W // 2
    return [t[..., :h2, :w2], t[..., :h2, w2:], t[..., h2:, :w2], t[..., h2:, w2:]]

def merge_4(maps):
    top = torch.cat([maps[0], maps[1]], dim=-1)
    bot = torch.cat([maps[2], maps[3]], dim=-1)
    return torch.cat([top, bot], dim=-2)

def load_mask(mask_path, size):
    m = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)

NORM_MODES = ("none", "zscore", "p99", "maxnorm")

def normalize_map(m, st, mode):
    if mode == "none":
        return m
    if mode == "zscore":
        return (m - st["mu"]) / (st["sigma"] + 1e-6)
    if mode == "p99":
        return m / (st["p99"] + 1e-6)
    if mode == "maxnorm":
        return m / (st["max"] + 1e-6)
    raise ValueError(mode)

def calib_stats(teacher, student, good_imgs, q, device, bs=32):
    """Stats delle anomaly map di student_q sul quadrante q del train/good reale."""
    vals = []
    with torch.no_grad():
        for i in range(0, len(good_imgs), bs):
            x = split_4(good_imgs[i:i + bs].to(device))[q]
            vals.append(anomaly_scores(teacher, student, x)[0].flatten().cpu().numpy())
    v = np.concatenate(vals)
    return {"mu": float(v.mean()), "sigma": float(v.std()),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}

def load_pt(path, device):
    x = torch.load(path, map_location="cpu").clamp(0, 1)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x.to(device)


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dir = os.path.join(args.data_path, args.category, "test")
    gt_dir = os.path.join(args.data_path, args.category, "ground_truth")
    mask_res = args.mask_size          # 224 nativa
    native_res = args.img_size         # 224 (l'immagine intera)

    teacher = build_teacher(backbone=args.backbone, layers=args.layers,
                            device=args.device, prefer_moviad=not args.no_moviad)

    tf = transforms.Compose([
        transforms.Resize((native_res, native_res), antialias=True),
        transforms.ToTensor(),
    ])

    # real train/good for per-quadrant calibration (no test labels)
    good_dir = Path(args.data_path) / args.category / "train" / "good"
    good_imgs = torch.stack([tf(Image.open(p).convert("RGB"))
                             for p in sorted(good_dir.iterdir())
                             if p.suffix.lower() in IMG_EXTENSIONS])
    print(f"[calib] {len(good_imgs)} train/good images from {good_dir}")

    # one student per quadrant
    students, stats = [], []
    for q in range(4):
        pt_path = args.pt_template.format(q=q)
        imgs = load_pt(pt_path, args.device)   # [IPC, 3, 112, 112]
        print(f"[q{q}] train images {tuple(imgs.shape)} from {pt_path}")
        s = train_student_features(
            imgs, teacher, args.device, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, backbone=args.backbone, layers=args.layers,
            prefer_moviad=not args.no_moviad, verbose=False,
        )
        s.eval()
        teacher.eval()
        st = calib_stats(teacher, s, good_imgs, q, args.device)
        print(f"[q{q}] calib good: mu={st['mu']:.4f} sigma={st['sigma']:.4f} "
              f"p99={st['p99']:.4f} max={st['max']:.4f}")
        students.append(s)
        stats.append(st)
    teacher.eval()

    test_base = Path(test_dir)
    modes = NORM_MODES if args.norm == "all" else (args.norm,)

    img_labels, all_pixel_targets = [], []
    img_scores = {m: [] for m in modes}
    all_pixel_preds = {m: [] for m in modes}
    # per-model map statistics
    per_q_vals = [[] for _ in range(4)]
    per_q_vals_norm = {m: [[] for _ in range(4)] for m in modes}

    with torch.no_grad():
        for p in sorted(test_base.rglob("*.*")):
            if p.suffix.lower() not in IMG_EXTENSIONS:
                continue
            defect = p.parent.name
            is_good = (defect == "good")
            img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(args.device)

            if is_good:
                mask = torch.zeros((1, 1, mask_res, mask_res), device=args.device)
            else:
                mf = os.path.join(gt_dir, defect, p.stem + "_mask.png")
                if not os.path.exists(mf):
                    mf = os.path.join(gt_dir, defect, p.name)
                m = load_mask(mf, mask_res)
                mask = torch.tensor(m, dtype=torch.float32, device=args.device)[None, None]

            quads = split_4(img)                       # 4 × [1,3,112,112]
            maps = []
            for q in range(4):
                mq = anomaly_scores(teacher, students[q], quads[q])[0]  # [1,1,112,112]
                maps.append(mq)
                per_q_vals[q].append(mq.flatten().cpu().numpy())

            img_labels.append(0 if is_good else 1)
            all_pixel_targets.extend((mask > 0).flatten().cpu().numpy().astype(int))

            for m in modes:
                maps_n = [normalize_map(maps[q], stats[q], m) for q in range(4)]
                for q in range(4):
                    per_q_vals_norm[m][q].append(maps_n[q].flatten().cpu().numpy())
                full_map = merge_4(maps_n)             # [1,1,224,224]

                img_scores[m].append(full_map.max().item())

                map_up = F.interpolate(full_map, size=(mask_res, mask_res),
                                       mode="bilinear", align_corners=False)
                all_pixel_preds[m].extend(map_up.flatten().cpu().numpy())

    print("\n=== Statistiche mappe per-modello (PRIMA del merge) ===")
    for q in range(4):
        v = np.concatenate(per_q_vals[q])
        print(f"  q{q}: min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}  std={v.std():.4f}")
    print("  -> se min/max/mean divergono molto tra i 4, serve normalizzare per-modello\n")

    for m in modes:
        if m != "none":
            print(f"=== Statistiche mappe per-modello (DOPO --norm {m}) ===")
            for q in range(4):
                v = np.concatenate(per_q_vals_norm[m][q])
                print(f"  q{q}: min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}  std={v.std():.4f}")

        g = [s for s,l in zip(img_scores[m],img_labels) if l==0]
        a = [s for s,l in zip(img_scores[m],img_labels) if l==1]
        print(f"  [{m}] good : mean={np.mean(g):.4f} min={np.min(g):.4f} max={np.max(g):.4f}")
        print(f"  [{m}] anom : mean={np.mean(a):.4f} min={np.min(a):.4f} max={np.max(a):.4f}")

        img_auroc = roc_auc_score(img_labels, img_scores[m])
        pixel_auroc = roc_auc_score(all_pixel_targets, all_pixel_preds[m])
        print(f"  [4 MODELLI - MERGE norm={m}] Image-AUROC (224): {img_auroc:.4f}")
        print(f"  [4 MODELLI - MERGE norm={m}] Pixel-AUROC (224): {pixel_auroc:.4f}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--pt_template", required=True,
                    help="path con {q}, es: ../outputs/distilled/bottle/10/real/q{q}_224/images_best.pt")
    ap.add_argument("--data_path", default="../data/mvtec_anomaly_detection")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--no_moviad", action="store_true")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--mask_size", type=int, default=224)
    ap.add_argument("--norm", default="zscore", choices=NORM_MODES + ("all",),
                    help="normalizzazione per-quadrante prima del merge (all = tutte, stessi student)")
    args = ap.parse_args()
    main(args)