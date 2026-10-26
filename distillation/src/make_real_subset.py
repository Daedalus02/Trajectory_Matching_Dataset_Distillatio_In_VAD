#!/usr/bin/env python3
"""Save n random real train/good images as a tensor [n,3,res,res] in [0,1] (starting sets, Real-10).
--like_distill picks the same images as distill.py without --init_path and with --seed <seed>
(the "initial" set of the thesis).
Usage: python make_real_subset.py --src ../data/mvtec_anomaly_detection/cable/train/good --n 10 --res 64 --seed 1001 --out r1.pt
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.feature_matching import load_images_tensor

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="train/good folder of the category")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--res", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--like_distill", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if a.like_distill:
    real = load_images_tensor(a.src, img_size=a.res, verbose=False)
    np.random.seed(a.seed)
    x = real[np.random.permutation(len(real))[:a.n]].clone()
else:
    files = sorted(f for f in glob.glob(os.path.join(a.src, "*")) if f.lower().endswith(EXTS))
    pick = random.Random(a.seed).sample(files, min(a.n, len(files)))
    tf = transforms.Compose([transforms.Resize((a.res, a.res), antialias=True), transforms.ToTensor()])
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in pick])
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save(x, a.out)
print("Saved {} -> {} (seed {})".format(tuple(x.shape), a.out, a.seed))
