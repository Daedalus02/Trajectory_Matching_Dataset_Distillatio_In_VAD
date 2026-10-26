#!/usr/bin/env python3
"""Trajectory-matching distillation for RD4AD.

Same loop as distill.py; trainable half = OCBE bn + decoder, inner loss = RD4AD cosine loss.
The encoder is not under no_grad (it feeds the decoder); only the targets are detached.
Output: outputs/distilled_rd4ad/<cat>/<ipc>/real/full<res>/.
"""


"""

                 ┌─ E1 ───────────── D1 ─┐
x → encoder ─────┼─ E2 ─→ OCBE → decoder ┼─ D2   → confronto per livello
   (congelato)   └─ E3 ───────────── D3 ─┘
   
   
   Quali feature. Da una parte ci sono le feature dell’encoder congelato ai tre stadi — 
   chiamiamole $E_1, E_2, E_3$ (per ResNet18: 64 canali @ H/4, 128 @ H/8, 256 @ H/16). 
   Dall’altra il decoder inverso ricostruisce tre mappe $D_1, D_2, D_3$ agli stessi tre 
   livelli (in ordine inverso, ma allineate per canali e dimensione spaziale). 
"""


import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.reparam_module import ReparamModule
from lib.rd4ad_matching import (
    DEFAULT_BACKBONE,
    DEFAULT_BLUR_SIGMA,
    build_encoder,
    build_rd4ad_student,
    check_trajectory_layout,
    evaluate_syn_auroc,
    flatten_snapshot,
    get_expert_dir,
    load_images_tensor,
    rd4ad_cosine_loss,
)


def manual_seed(seed=0):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_buffer_files(expert_dir):
    files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    return files


def main(args):
    manual_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Real images of the category (to initialize the synthetic ones) ──
    train_dir = os.path.join(args.data_path, args.category, "train", "good")
    test_dir = os.path.join(args.data_path, args.category, "test")
    images_all = load_images_tensor(train_dir, img_size=args.res, normalize=args.normalize)

    if args.quadrant is not None:
        h_mid, w_mid = images_all.shape[-2] // 2, images_all.shape[-1] // 2
        quads = [
            images_all[:, :, :h_mid, :w_mid],   # 0 TL
            images_all[:, :, :h_mid, w_mid:],   # 1 TR
            images_all[:, :, h_mid:, :w_mid],   # 2 BL
            images_all[:, :, h_mid:, w_mid:],   # 3 BR
        ]
        images_all = quads[args.quadrant]
        print(f"Quadrant {args.quadrant}: tensor shape {images_all.shape}")

    eff_res = images_all.shape[-1]
    channel = images_all.shape[1]

    # ── Frozen encoder (feature target AND decoder input) ──
    encoder = build_encoder(
        backbone=args.backbone, device=args.device, pretrained=True, verbose=True
    )

    # ── Synthetic image initialization (random real or noise) ──
    num_syn = args.ipc
    if args.pix_init == "noise":
        image_syn = torch.randn(
            num_syn, channel, eff_res, eff_res
        ).to(args.device).requires_grad_(True)
        print("Synthetic set: {} images {} (init: NOISE)".format(num_syn, tuple(image_syn.shape)))
    else:
        idx = np.random.permutation(len(images_all))[:num_syn]
        image_syn = images_all[idx].clone().detach().to(args.device).requires_grad_(True)
        print("Synthetic set: {} images {} (init: REAL)".format(num_syn, tuple(image_syn.shape)))

    # syn_lr frozen (learnable caused divergence in the feature-matching version)
    syn_lr = torch.tensor(args.lr_teacher, device=args.device)

    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)

    # ── Expert buffer ──
    expert_dir = get_expert_dir(args.buffer_path, args.dataset, args.model,
                                args.category, args.per_category_buffer)
    print("Expert dir: {}".format(expert_dir))
    expert_files = load_buffer_files(expert_dir)
    if args.max_files is not None:
        expert_files = expert_files[:args.max_files]

    # ── Surrogate bn+decoder (ReparamModule) ──
    student_template = build_rd4ad_student(
        backbone=args.backbone, device=args.device, bn_track_stats=False, verbose=True,
    )
    student_net = ReparamModule(student_template)
    detach_target = not args.target_grad
    print("Inner target features: {}".format("differentiable" if args.target_grad else "detached"))

    # ── wandb ──
    wandb.init(project=args.project, config=vars(args),
               mode="offline" if args.offline else "online")

    # load first buffer
    file_idx = 0
    buffer = torch.load(expert_files[file_idx], map_location="cpu")
    buffer_idx = 0
    check_trajectory_layout(student_net, buffer[0][0])

    # ── output directory (separated by pix_init) ──
    tag = f"full{eff_res}" if args.quadrant is None else f"q{args.quadrant}_{eff_res}"
    save_dir = os.path.join("..", "outputs", "distilled_rd4ad", args.category,
                            str(args.ipc), args.pix_init, tag)
    os.makedirs(save_dir, exist_ok=True)

    print("Distillation begins")
    best_auroc = 0.0
    history = []
    last_auroc = float("nan")

    for it in range(0, args.Iteration + 1):

        # ── Periodic evaluation ──
        if it % args.eval_it == 0:
            print("\n--- Eval @ iter {} ---".format(it))
            result = evaluate_syn_auroc(
                image_syn.detach(), encoder, test_dir, args.device,
                img_size=eff_res, epochs=args.eval_epochs, lr=args.eval_lr,
                batch_size=args.batch_syn, optimizer=args.eval_optimizer,
                backbone=args.backbone, detach_target=detach_target,
                normalize=args.normalize, blur_sigma=args.blur_sigma, verbose=True,
            )
            auroc = result["auroc"]
            best_auroc = max(best_auroc, auroc)
            last_auroc = auroc
            wandb.log({"AUROC": auroc, "Best_AUROC": best_auroc}, step=it)

            torch.save(image_syn.detach().cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
            if auroc >= best_auroc:
                torch.save(image_syn.detach().cpu(), os.path.join(save_dir, "images_best.pt"))

        # ── Sample expert trajectory ──
        trajectory = buffer[buffer_idx]
        buffer_idx += 1
        if buffer_idx >= len(buffer):
            buffer_idx = 0
            file_idx = (file_idx + 1) % len(expert_files)
            del buffer
            buffer = torch.load(expert_files[file_idx], map_location="cpu")

        # ── Difficulty alignment (Sequential Generation) ──
        if args.Sequential_Generation:
            upper = args.current_max_start_epoch + int(
                (args.max_start_epoch - args.current_max_start_epoch)
                * it / max(args.expansion_end_epoch, 1)
            )
            upper = min(upper, args.max_start_epoch)
        else:
            upper = args.max_start_epoch
        upper = max(upper, args.min_start_epoch + 1)

        start_epoch = np.random.randint(args.min_start_epoch, upper)

        # ── Start / target parameters ──
        starting_params = flatten_snapshot(trajectory[start_epoch], args.device)
        target_params = flatten_snapshot(
            trajectory[start_epoch + args.expert_epochs], args.device
        )
        student_params = [starting_params.clone().requires_grad_(True)]

        # ── Inner optimization: train bn+decoder on the synthetic images ──
        for step in range(args.syn_steps):
            these = torch.randperm(len(image_syn))[:args.batch_syn]
            x = image_syn[these]

            if args.distributed:
                fwd = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                fwd = student_params[-1]

            # NOT under no_grad: the decoder is fed bn(encoder(x)), so cutting the
            # graph here would also cut the meta-gradient towards image_syn
            enc_features = encoder(x)
            dec_features = student_net(enc_features, flat_param=fwd)

            loss = rd4ad_cosine_loss(enc_features, dec_features, detach_target=detach_target)
            grad = torch.autograd.grad(loss, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - syn_lr * grad)

        # ── Trajectory matching loss (Eq. 1) ──
        param_loss = F.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist = F.mse_loss(starting_params, target_params, reduction="sum")

        if it % 10 == 0:
            print("  DEBUG: nan={} param_loss={:.4f} param_dist={:.4f} syn_lr={:.6f} "
                  "cos_loss={:.4f}".format(
                      torch.isnan(student_params[-1]).any().item(),
                      param_loss.item(), param_dist.item(), syn_lr.item(),
                      loss.item()))

        grand_loss = param_loss / param_dist

        # ── Image update (with clipping, skipped if loss is high or NaN) ──
        optimizer_img.zero_grad()
        grand_loss.backward()
        if grand_loss.item() <= args.threshold and not torch.isnan(grand_loss):
            torch.nn.utils.clip_grad_norm_([image_syn], max_norm=1.0)
            optimizer_img.step()

        for p in student_params:
            del p

        if it % 10 == 0:
            print("iter {:04d} | grand_loss {:.4f} | start_epoch {}".format(
                it, grand_loss.item(), start_epoch))
        wandb.log({"Grand_Loss": grand_loss.item(), "Start_Epoch": start_epoch}, step=it)

        # ── history row (auroc set only at eval iterations) ──
        auroc_val = last_auroc if (it % args.eval_it == 0) else float("nan")
        history.append({
            "iter": it,
            "grand_loss": grand_loss.item(),
            "start_epoch": start_epoch,
            "auroc": auroc_val,
        })
        last_auroc = float("nan")

    print("\nBest AUROC: {:.4f}".format(best_auroc))

    # ── save history to CSV (same folder as the images) ──
    csv_path = os.path.join(save_dir, "history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "grand_loss", "start_epoch", "auroc"])
        writer.writeheader()
        writer.writerows(history)
    print("History saved to {}".format(csv_path))

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="")
    # data
    parser.add_argument("--dataset", type=str, default="MVTec_rd4ad")
    parser.add_argument("--model", type=str, default="RD4AD")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--data_path", type=str, default="../data/mvtec_anomaly_detection")
    parser.add_argument("--buffer_path", type=str, default="../outputs/buffer_rd4ad")
    parser.add_argument("--per_category_buffer", action="store_true")
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--normalize", action="store_true")
    # RD4AD architecture (must match the buffer)
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        choices=["resnet18", "resnet34", "resnet50", "wide_resnet50_2"])
    parser.add_argument("--target_grad", action="store_true",
                        help="do NOT detach the encoder target features in the inner loss "
                             "(default: detached, as distill.py does with the teacher)")
    parser.add_argument("--blur_sigma", type=float, default=DEFAULT_BLUR_SIGMA,
                        help="gaussian smoothing of the anomaly map at evaluation (0 = off)")
    # distillation
    parser.add_argument("--ipc", type=int, default=5)
    parser.add_argument("--Iteration", type=int, default=1000)
    parser.add_argument("--syn_steps", type=int, default=20)
    parser.add_argument("--expert_epochs", type=int, default=2)
    parser.add_argument("--min_start_epoch", type=int, default=0)
    parser.add_argument("--max_start_epoch", type=int, default=10)
    parser.add_argument("--lr_img", type=float, default=0.01)
    parser.add_argument("--lr_teacher", type=float, default=0.01)
    parser.add_argument("--batch_syn", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--distributed", action="store_true")
    # sequential generation
    parser.add_argument("--Sequential_Generation", action="store_true")
    parser.add_argument("--current_max_start_epoch", type=int, default=2)
    parser.add_argument("--expansion_end_epoch", type=int, default=500)
    # evaluation
    parser.add_argument("--eval_it", type=int, default=100)
    parser.add_argument("--eval_epochs", type=int, default=50)
    parser.add_argument("--eval_lr", type=float, default=0.01)
    parser.add_argument("--eval_optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    # logging
    parser.add_argument("--project", type=str, default="DATM_rd4ad_MVTec")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--pix_init", type=str, default="real", choices=["real", "noise"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quadrant", type=int, default=None, choices=[0, 1, 2, 3],
                        help="Estrae UN solo quadrante (0=TL,1=TR,2=BL,3=BR) invece del pool di tutti")
    args = parser.parse_args()

    if args.cfg:
        import yaml
        with open(args.cfg) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    main(args)
