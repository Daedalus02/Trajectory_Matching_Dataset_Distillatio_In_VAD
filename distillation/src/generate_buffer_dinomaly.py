"""Expert trajectories for Dinomaly distillation.

Trains bottleneck + decoder with plain SGD and the hook-free hard-mining loss; one snapshot
per epoch. --half_buffer stores fp16 snapshots (about 6 GB per category instead of 13).
"""

import argparse
import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.dinomaly_matching import (
    BACKBONE_CHOICES,
    DEFAULT_BACKBONE,
    DEFAULT_HM_FACTOR,
    DEFAULT_HM_P,
    DEFAULT_RES,
    build_dinomaly_student,
    build_encoder,
    check_res,
    count_parameters,
    dinomaly_loss,
    get_expert_dir,
    gpu_peak_report,
    load_images_tensor,
    parameter_snapshot,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="xFormers")


def train_epoch(args, loader, encoder, student, optimizer):
    """One pass over the real images with the Dinomaly hard-mining cosine loss."""
    student.train()
    encoder.eval()
    loss_avg, num_exp = 0.0, 0

    for images in loader:
        images = images.float().to(args.device)
        n_b = images.shape[0]

        with torch.no_grad():
            en_list = encoder(images)
        loss = dinomaly_loss(student, en_list, student(en_list), p=args.hm_p, factor=args.hm_factor)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_avg += loss.item() * n_b
        num_exp += n_b

    return loss_avg / max(num_exp, 1)


def save_buffer(save_dir, trajectories, note=""):
    n = 0
    while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
        n += 1
    path = os.path.join(save_dir, "replay_buffer_{}.pt".format(n))
    print("Saving {}{}".format(path, note))
    torch.save(trajectories, path)


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    check_res(args.res, args.backbone)
    print("Hyper-parameters: \n", args.__dict__)
    t_start = time.time()

    train_dir = os.path.join(args.data_path, args.category, "train", "good")
    if not os.path.isdir(train_dir):
        raise RuntimeError("Training folder not found: {}".format(train_dir))

    save_dir = get_expert_dir(
        args.buffer_path, args.dataset, args.model, args.category, args.per_category_buffer
    )
    os.makedirs(save_dir, exist_ok=True)
    print("Saving trajectories to {}".format(save_dir))

    """ organize the real dataset (single category, all images are 'normal') """
    print("BUILDING DATASET")
    images_all = load_images_tensor(train_dir, img_size=args.res, verbose=True)

    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images_all),
        batch_size=min(args.batch_train, len(images_all)),
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: torch.stack([b[0] for b in batch]),
    )

    # frozen target AND decoder input
    encoder = build_encoder(backbone=args.backbone, device=args.device, verbose=True)

    trajectories = []

    for it in range(0, args.num_experts):

        """ Train an expert Dinomaly (bottleneck + decoder) on the real data """
        student = build_dinomaly_student(backbone=args.backbone, device=args.device, verbose=(it == 0))
        if it == 0:
            n_params = count_parameters(student)
            print("Trajectory snapshot: {} tensors, {} parameters (~{:.1f} MB each)".format(
                len(list(student.parameters())), n_params,
                n_params * (2 if args.half_buffer else 4) / 1e6))
            print("Expert training scheme: sgd (lr={}, momentum={})".format(args.lr_teacher, args.mom))

        optimizer = torch.optim.SGD(student.parameters(), lr=args.lr_teacher,
                                    momentum=args.mom, weight_decay=args.l2)

        timestamps = []
        timestamps.append(parameter_snapshot(student, half=args.half_buffer))

        for e in range(args.train_epochs):
            loss = train_epoch(args, trainloader, encoder, student, optimizer)
            print("Itr: {}\tEpoch: {}\tCosine Loss: {:.6f}".format(it, e, loss))
            timestamps.append(parameter_snapshot(student, half=args.half_buffer))

        trajectories.append(timestamps)
        del student, optimizer
        if args.device == "cuda":
            torch.cuda.empty_cache()

        if len(trajectories) == args.save_interval:
            save_buffer(save_dir, trajectories)
            trajectories = []

    # flush whatever is left over when num_experts is not a multiple of save_interval
    if len(trajectories) > 0:
        save_buffer(save_dir, trajectories, note=" (final flush)")

    print(gpu_peak_report(args.device))
    print("Buffer generation time: {:.1f} s".format(time.time() - t_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dinomaly expert trajectories")
    # --- what gets distilled ---------------------------------------------------
    parser.add_argument("--dataset", type=str, default="MVTec_dinomaly", help="buffer folder name")
    parser.add_argument("--model", type=str, default="DINOMALY", help="buffer sub-folder name")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category")
    parser.add_argument("--data_path", type=str, default="../data/mvtec_anomaly_detection")
    parser.add_argument("--buffer_path", type=str, default="../outputs/buffer_dinomaly")
    parser.add_argument(
        "--per_category_buffer",
        action="store_true",
        help="save to <buffer>/<dataset>/<category>/<model> instead of <buffer>/<dataset>/<model>",
    )
    # --- Dinomaly architecture -------------------------------------------------
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, choices=BACKBONE_CHOICES,
                        help="frozen DINOv2 encoder (sets the bottleneck/decoder width)")
    parser.add_argument("--res", type=int, default=DEFAULT_RES, help="training resolution (multiple of 14)")
    parser.add_argument("--hm_p", type=float, default=DEFAULT_HM_P, help="hard-mining percent of the loss")
    parser.add_argument("--hm_factor", type=float, default=DEFAULT_HM_FACTOR,
                        help="gradient factor on the easy points")
    # --- trajectory generation (same knobs as generate_buffer_rd4ad.py) --------
    parser.add_argument("--num_experts", type=int, default=10, help="number of expert trajectories")
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1, help="trajectories per buffer file")
    parser.add_argument("--lr_teacher", type=float, default=0.01)
    parser.add_argument("--batch_train", type=int, default=16)
    parser.add_argument("--mom", type=float, default=0.0, help="momentum")
    parser.add_argument("--l2", type=float, default=0.0, help="l2 regularization")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd"],
                        help="only sgd: the step the distillation inner loop reproduces")
    parser.add_argument("--half_buffer", action="store_true", help="store the snapshots in fp16 (half the disk)")
    args = parser.parse_args()
    main(args)
