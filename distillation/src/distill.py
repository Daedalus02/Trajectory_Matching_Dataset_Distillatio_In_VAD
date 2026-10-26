#!/usr/bin/env python3
"""Trajectory-matching distillation for STFPM (feature matching).

A set of ipc synthetic images is optimised so that a student trained on it for syn_steps steps,
starting from an expert snapshot, ends close to the expert snapshot expert_epochs later
(normalised parameter distance, Eq. 2.3 of the thesis). The inner loss is the feature MSE between
the frozen teacher and the student: no labels.

The defaults reproduce the main sweep (Ch. 4: SGD with clipping on the pixels, exact meta-gradient).
The final runs (Ch. 5) add: --zca (optimise ZCA coefficients), --opt_img adam --img_cos,
--fo (first-order meta-gradient), --n_pairs, --threshold 1000 (no skipped updates),
--eval_pairs/--fixed_eval_every (loss on fixed pairs), --init_path (given starting set).

Outputs in outputs/distilled/<category>/<ipc>/real/<tag>/:
images_<it>.pt (pixels in [0,1]), images_best.pt, history.csv, fixed_eval.csv.
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.reparam_module import ReparamModule
from lib.feature_matching import (
    DEFAULT_LAYERS,
    build_student,
    build_teacher,
    check_trajectory_layout,
    evaluate_syn_auroc,
    feature_mse_loss,
    flatten_snapshot,
    get_expert_dir,
    load_images_tensor,
)


class ZCA:
    """ZCA whitening fitted on the real images (eps-regularised, as kornia.ZCAWhitening).
    whiten: pixels -> ZCA space; color: ZCA space -> pixels (exact inverse)."""

    def __init__(self, x, eps=0.1):
        f = x.flatten(1)
        self.shape = x.shape[1:]
        self.mu = f.mean(0, keepdim=True)
        _, sv, vh = torch.linalg.svd(f - self.mu, full_matrices=False)
        lam = sv ** 2 / f.shape[0]
        self.v = vh.T
        self.a = 1.0 / torch.sqrt(lam + eps)   # scale on the directions spanned by the data
        self.b = torch.sqrt(lam + eps)
        self.se = eps ** 0.5                   # scale on the null space

    def whiten(self, x):
        f = x.flatten(1) - self.mu
        p = f @ self.v
        return ((p * self.a) @ self.v.T + (f - p @ self.v.T) / self.se).view(-1, *self.shape)

    def color(self, z):
        f = z.flatten(1)
        p = f @ self.v
        return ((p * self.b) @ self.v.T + (f - p @ self.v.T) * self.se + self.mu).view(-1, *self.shape)


def manual_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_buffer_files(expert_dir):
    files = []
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(len(files)))):
        files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(len(files))))
    if not files:
        raise FileNotFoundError("No buffers in {}".format(expert_dir))
    return files


def main(args):
    manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = os.path.join(args.data_path, args.category, "train", "good")
    test_dir = os.path.join(args.data_path, args.category, "test")
    images_all = load_images_tensor(train_dir, img_size=args.res)
    if args.quadrant is not None:  # patch-split: one 112x112 quadrant of the 224x224 images
        h, w = images_all.shape[-2] // 2, images_all.shape[-1] // 2
        images_all = [images_all[:, :, :h, :w], images_all[:, :, :h, w:],
                      images_all[:, :, h:, :w], images_all[:, :, h:, w:]][args.quadrant]
    eff_res = images_all.shape[-1]

    teacher = build_teacher(backbone=args.backbone, layers=args.layers, device=device)

    # starting set: a given .pt, or ipc random real images
    if args.init_path:
        image_syn = torch.load(args.init_path, map_location="cpu").float()
        assert tuple(image_syn.shape) == (args.ipc, images_all.shape[1], eff_res, eff_res), image_syn.shape
        image_syn = image_syn.to(device)
    else:
        idx = np.random.permutation(len(images_all))[:args.ipc]
        image_syn = images_all[idx].clone().to(device)
    print("Synthetic set: {} (init: {})".format(tuple(image_syn.shape), args.init_path or "random real"))

    # the optimised variable lives in ZCA space if --zca; the networks always see pixels
    zca = ZCA(images_all.to(device), eps=args.zca_eps) if args.zca else None
    if zca is not None:
        image_syn = zca.whiten(image_syn)
    image_syn = image_syn.detach().requires_grad_(True)
    to_pix = zca.color if zca is not None else (lambda t: t)

    syn_lr = torch.tensor(args.lr_teacher, device=device)  # inner (student) learning rate, fixed
    if args.opt_img == "adam":
        optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    else:
        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    sched_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration) if args.img_cos else None

    expert_files = load_buffer_files(get_expert_dir(args.buffer_path, args.dataset, args.model))
    student_net = ReparamModule(build_student(backbone=args.backbone, layers=args.layers, device=device,
                                              bn_track_stats=False, verbose=True))
    file_idx, buffer_idx = 0, 0
    buffer = torch.load(expert_files[file_idx], map_location="cpu")
    check_trajectory_layout(student_net, buffer[0][0])

    # fixed (start, target) pairs: every trajectory x evenly spaced start snapshots, drawn once
    fixed_pairs = []
    if args.eval_pairs > 0:
        trajs = [t for f in expert_files for t in torch.load(f, map_location="cpu")]
        k = max(1, args.eval_pairs // len(trajs))
        starts = sorted({int(round(v)) for v in np.linspace(args.min_start_epoch, args.max_start_epoch - 1, k)})
        fixed_pairs = [(flatten_snapshot(t[e], "cpu"), flatten_snapshot(t[e + args.expert_epochs], "cpu"))
                       for t in trajs for e in starts]
        del trajs
        print("Fixed evaluation: {} pairs (start snapshots {})".format(len(fixed_pairs), starts))

    def fixed_losses(img):
        out = []
        for s0, s1 in fixed_pairs:
            start, target = s0.to(device), s1.to(device)
            w = start.clone().requires_grad_(True)
            for _ in range(args.syn_steps):
                with torch.no_grad():
                    tf = teacher(img)
                g = torch.autograd.grad(feature_mse_loss(tf, student_net(img, flat_param=w)), w)[0]
                w = (w - syn_lr * g).detach().requires_grad_(True)
            out.append((F.mse_loss(w, target, reduction="sum") / F.mse_loss(start, target, reduction="sum")).item())
        return np.array(out)

    tag = ("full{}".format(args.res) if args.quadrant is None else "q{}_{}".format(args.quadrant, args.res)) + args.save_tag
    save_dir = os.path.join("..", "outputs", "distilled", args.category, str(args.ipc), "real", tag)
    os.makedirs(save_dir, exist_ok=True)

    best_auroc, fixed0, history = 0.0, None, []
    for it in range(args.Iteration + 1):
        if fixed_pairs and it % args.fixed_eval_every == 0:
            fl = fixed_losses(to_pix(image_syn.detach()))
            fixed0 = fl if fixed0 is None else fixed0
            rel = float(np.mean(fl / fixed0))
            print("  FIXED-EVAL iter {} | loss {:.4f} | relative {:.4f}".format(it, fl.mean(), rel))
            with open(os.path.join(save_dir, "fixed_eval.csv"), "w" if it == 0 else "a") as fh:
                if it == 0:
                    fh.write("iter,fixed_loss,fixed_rel\n")
                fh.write("{},{:.5f},{:.5f}\n".format(it, fl.mean(), rel))

        auroc = float("nan")
        if it % args.eval_it == 0:  # checkpoint; the test AUROC is only a log (selection is done on the validation half)
            pix = to_pix(image_syn.detach())
            torch.save(pix.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
            if args.eval_epochs > 0:
                auroc = evaluate_syn_auroc(pix, teacher, test_dir, device, img_size=eff_res, epochs=args.eval_epochs,
                                           lr=args.eval_lr, batch_size=args.batch_syn, backbone=args.backbone,
                                           layers=args.layers, verbose=True)["auroc"]
                if auroc >= best_auroc:
                    best_auroc = auroc
                    torch.save(pix.cpu(), os.path.join(save_dir, "images_best.pt"))

        optimizer_img.zero_grad()
        losses = []
        for pair in range(args.n_pairs):
            trajectory = buffer[buffer_idx]
            buffer_idx += 1
            if buffer_idx >= len(buffer):
                buffer_idx, file_idx = 0, (file_idx + 1) % len(expert_files)
                buffer = torch.load(expert_files[file_idx], map_location="cpu")

            lo, hi = args.min_start_epoch, max(args.max_start_epoch, args.min_start_epoch + 1)
            if args.stratified:  # one start snapshot per sub-range of [lo, hi)
                a = lo + (hi - lo) * pair / args.n_pairs
                b = lo + (hi - lo) * (pair + 1) / args.n_pairs
                start_epoch = min(np.random.randint(int(np.floor(a)), max(int(np.floor(a)) + 1, int(np.ceil(b)))), hi - 1)
            else:
                start_epoch = np.random.randint(lo, hi)

            starting_params = flatten_snapshot(trajectory[start_epoch], device)
            target_params = flatten_snapshot(trajectory[start_epoch + args.expert_epochs], device)
            student_params = [starting_params.clone().requires_grad_(True)]

            # inner loop: train the student on the synthetic images (feature MSE), kept differentiable
            for _ in range(args.syn_steps):
                x = to_pix(image_syn[torch.randperm(len(image_syn))[:args.batch_syn]])
                w = student_params[-1].detach().requires_grad_(True) if args.fo else student_params[-1]
                with torch.no_grad():
                    teacher_features = teacher(x)
                loss = feature_mse_loss(teacher_features, student_net(x, flat_param=w))
                grad = torch.autograd.grad(loss, w, create_graph=True)[0]
                student_params.append(student_params[-1] - syn_lr * grad)

            grand_loss = (F.mse_loss(student_params[-1], target_params, reduction="sum")
                          / F.mse_loss(starting_params, target_params, reduction="sum"))
            (grand_loss / args.n_pairs).backward()
            losses.append(grand_loss.item())
            del student_params

        grand_loss = float(np.mean(losses))
        # updates with a loss above --threshold are skipped (as in DATM); the final runs disable this
        if grand_loss <= args.threshold and not np.isnan(grand_loss):
            if args.opt_img == "sgd":
                torch.nn.utils.clip_grad_norm_([image_syn], max_norm=1.0)
            optimizer_img.step()
        if sched_img is not None:
            sched_img.step()

        if it % 10 == 0:
            print("iter {:04d} | grand_loss {:.4f} | start_epoch {}".format(it, grand_loss, start_epoch))
        history.append({"iter": it, "grand_loss": grand_loss, "start_epoch": start_epoch, "auroc": auroc})

    with open(os.path.join(save_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "grand_loss", "start_epoch", "auroc"])
        writer.writeheader()
        writer.writerows(history)
    print("Saved to {}".format(save_dir))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data and experts
    p.add_argument("--category", type=str, default="bottle")
    p.add_argument("--data_path", type=str, default="../data/mvtec_anomaly_detection")
    p.add_argument("--buffer_path", type=str, required=True, help="folder written by generate_buffer.py")
    p.add_argument("--dataset", type=str, default="MVTec_feature", help="buffer sub-folder")
    p.add_argument("--model", type=str, default="ResNet18", help="buffer sub-folder")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--quadrant", type=int, default=None, choices=[0, 1, 2, 3], help="patch-split quadrant (TL, TR, BL, BR)")
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    # synthetic set
    p.add_argument("--ipc", type=int, default=10)
    p.add_argument("--init_path", type=str, default="", help="starting set (.pt, [ipc,3,res,res] in [0,1]); default random real")
    p.add_argument("--Iteration", type=int, default=1000)
    p.add_argument("--lr_img", type=float, default=0.01)
    p.add_argument("--opt_img", type=str, default="sgd", choices=["sgd", "adam"], help="sgd = momentum 0.5 + clipping")
    p.add_argument("--img_cos", action="store_true", help="cosine decay of the image learning rate")
    p.add_argument("--zca", action="store_true", help="optimise ZCA coefficients instead of pixels")
    p.add_argument("--zca_eps", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=1.5, help="skip the update if the loss is above this")
    # matching
    p.add_argument("--syn_steps", type=int, default=20, help="inner steps")
    p.add_argument("--lr_teacher", type=float, default=0.01, help="inner learning rate")
    p.add_argument("--batch_syn", type=int, default=5, help="inner batch size")
    p.add_argument("--expert_epochs", type=int, default=2, help="snapshots between start and target")
    p.add_argument("--min_start_epoch", type=int, default=0)
    p.add_argument("--max_start_epoch", type=int, default=10)
    p.add_argument("--n_pairs", type=int, default=1, help="(trajectory, start) pairs averaged per iteration")
    p.add_argument("--stratified", action="store_true", help="spread the start snapshots of one iteration")
    p.add_argument("--fo", action="store_true", help="first-order meta-gradient")
    # checkpoints and checks
    p.add_argument("--eval_it", type=int, default=100, help="checkpoint every eval_it iterations")
    p.add_argument("--eval_epochs", type=int, default=50, help="epochs of the logged test AUROC; 0 = no AUROC")
    p.add_argument("--eval_lr", type=float, default=0.01)
    p.add_argument("--eval_pairs", type=int, default=0, help="number of fixed pairs for the loss check; 0 = off")
    p.add_argument("--fixed_eval_every", type=int, default=100)
    p.add_argument("--save_tag", type=str, default="", help="suffix of the output folder")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
