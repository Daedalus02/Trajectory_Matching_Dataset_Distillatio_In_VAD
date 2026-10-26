#!/usr/bin/env python3
"""Downsample training images (real folder or distilled .pt) to --down and back to --up.

Used for the Dinomaly-at-64 ablation: 224x224 input with only 64x64 detail.
Example: python degrade_res.py --src <train/good or .pt> --down 64 --up 224 --out out.pt
"""
import os
import sys
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lib.dinomaly_matching import load_images_tensor  # same loader as the evaluation


def load_any(src, up):
    """.pt distillato (gia' tensore in [0,1]) oppure cartella di PNG reali."""
    if str(src).endswith(".pt"):
        im = torch.load(src, map_location="cpu").clamp(0, 1)
        if im.dim() == 3:
            im = im.unsqueeze(0)
        # resize to 'up' if needed
        if im.shape[-1] != up:
            im = F.interpolate(im, size=(up, up), mode="bilinear", align_corners=False).clamp(0, 1)
        return im
    # cartella reale: caricata a 'up' px, identico al baseline
    return load_images_tensor(src, img_size=up).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="cartella train/good oppure .pt distillato")
    ap.add_argument("--down", type=int, default=64, help="risoluzione a cui degradare il dettaglio")
    ap.add_argument("--up", type=int, default=224, help="risoluzione finale (ingresso ViT)")
    ap.add_argument("--out", required=True, help="file .pt di output")
    args = ap.parse_args()

    im = load_any(args.src, args.up)
    print("input  {}  (range [{:.3f}, {:.3f}])".format(tuple(im.shape), im.min().item(), im.max().item()))

    lo = F.interpolate(im, size=(args.down, args.down), mode="bilinear", align_corners=False)
    hi = F.interpolate(lo, size=(args.up, args.up), mode="bilinear", align_corners=False).clamp(0, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(hi, args.out)
    print("salvato {} -> {}".format(tuple(hi.shape), args.out))
    print("  dettaglio a {}px, ingresso a {}px".format(args.down, args.up))


if __name__ == "__main__":
    main()
