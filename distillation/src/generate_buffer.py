"""Expert trajectories for STFPM distillation.

Trains --num_experts students (ResNet18 layer1-3) against the frozen teacher with feature MSE
and saves one parameter snapshot per epoch (replay_buffer_<i>.pt in --buffer_path).
"""

import argparse
import os
import sys
import warnings
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.feature_matching import (
    DEFAULT_LAYERS,
    build_student,
    build_teacher,
    count_parameters,
    feature_mse_loss,
    get_expert_dir,
    load_images_tensor,
    parameter_snapshot,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_augmentation(args):
    """DSA augmentation, imported lazily so the script also runs without kornia."""
    if not args.dsa:
        return None
    from utils.utils_baseline import DiffAugment, ParamDiffAug

    dsa_param = ParamDiffAug()
    return lambda images: DiffAugment(images, args.dsa_strategy, param=dsa_param)


def build_optimizer(args, student, steps_per_epoch):
    """Either the FTD/GSAM optimizer of the original buffer script, or plain SGD."""
    base_optimizer = torch.optim.SGD(
        student.parameters(), lr=args.lr_teacher, momentum=args.mom, weight_decay=args.l2
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        base_optimizer, step_size=args.train_epochs * max(steps_per_epoch, 1), gamma=1
    )

    if args.optimizer == "sgd":
        return base_optimizer, scheduler, None

    from gsam import GSAM, ProportionScheduler

    rho_scheduler = ProportionScheduler(
        pytorch_lr_scheduler=scheduler,
        max_lr=args.lr_teacher,
        min_lr=args.lr_teacher,
        max_value=args.rho_max,
        min_value=args.rho_min,
    )
    optimizer = GSAM(
        params=student.parameters(),
        base_optimizer=base_optimizer,
        model=student,
        gsam_alpha=args.alpha,
        rho_scheduler=rho_scheduler,
        adaptive=args.adaptive,
    )
    return optimizer, scheduler, base_optimizer


def train_epoch(args, loader, student, teacher, optimizer, scheduler, augment):
    """One pass over the real images with the student-teacher feature MSE loss."""
    student.train()
    loss_avg, num_exp = 0.0, 0

    for images in loader:
        images = images.float().to(args.device)
        if augment is not None:
            images = augment(images)

        # The teacher is frozen: its features are the (constant) regression target.
        with torch.no_grad():
            teacher_features = teacher(images)

        n_b = images.shape[0]

        if args.optimizer == "gsam":
            # GSAM calls model(inputs) itself and then loss_fn(outputs, targets).
            optimizer.set_closure(
                lambda student_features, target_features: feature_mse_loss(
                    target_features, student_features
                ),
                images,
                teacher_features,
            )
            _, loss = optimizer.step()
            scheduler.step()
            optimizer.update_rho_t()
            loss_value = loss.item()
        else:
            student_features = student(images)
            loss = feature_mse_loss(teacher_features, student_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_value = loss.item()

        loss_avg += loss_value * n_b
        num_exp += n_b

    return loss_avg / max(num_exp, 1)


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dsa = args.dsa == "True"
    print("Hyper-parameters: \n", args.__dict__)

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
    images_all = load_images_tensor(
        train_dir, img_size=args.res, normalize=args.normalize, verbose=True
    )

    #if args.patch_split:
    #    # Se res=224, estrae 4 patch da 112x112
    #    h_mid, w_mid = images_all.shape[-2] // 2, images_all.shape[-1] // 2
    #    p1 = images_all[:, :, :h_mid, :w_mid]
    #    p2 = images_all[:, :, :h_mid, w_mid:]
    #    p3 = images_all[:, :, h_mid:, :w_mid]
    #    p4 = images_all[:, :, h_mid:, w_mid:]
    #    images_all = torch.cat([p1, p2, p3, p4], dim=0)
    #    print(f"Divided into 4 patches: new tensor shape {images_all.shape} ({h_mid}x{w_mid})")

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

    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images_all),
        batch_size=min(args.batch_train, len(images_all)),
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: torch.stack([b[0] for b in batch]),
    )

    teacher = build_teacher(
        backbone=args.backbone,
        layers=args.layers,
        device=args.device,
        prefer_moviad=not args.no_moviad,
    )
    augment = build_augmentation(args)
    if args.dsa:
        print("DSA augmentation strategy: {}".format(args.dsa_strategy))

    trajectories = []

    for it in range(0, args.num_experts):

        """ Train an expert student on the real data """
        student = build_student(
            backbone=args.backbone,
            layers=args.layers,
            device=args.device,
            bn_track_stats=args.bn_track_stats,
            prefer_moviad=not args.no_moviad,
            verbose=(it == 0),
        )
        if it == 0:
            print(
                "Trajectory snapshot: {} tensors, {} parameters (~{:.1f} MB each)".format(
                    len(list(student.parameters())),
                    count_parameters(student),
                    count_parameters(student) * 4 / 1e6,
                )
            )

        optimizer, scheduler, _ = build_optimizer(args, student, len(trainloader))

        timestamps = []
        timestamps.append(parameter_snapshot(student))

        for e in range(args.train_epochs):
            train_loss = train_epoch(
                args, trainloader, student, teacher, optimizer, scheduler, augment
            )
            print("Itr: {}\tEpoch: {}\tTrain Feature Loss: {:.6f}".format(it, e, train_loss))
            if (e + 1) % args.snap_every == 0:  # snapshot k = epoch k * snap_every
                timestamps.append(parameter_snapshot(student))

        trajectories.append(timestamps)
        del student, optimizer, scheduler
        if args.device == "cuda":
            torch.cuda.empty_cache()

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []

    # flush whatever is left over when num_experts is not a multiple of save_interval
    if len(trajectories) > 0:
        n = 0
        while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
            n += 1
        print("Saving {} (final flush)".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
        torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature-matching expert trajectories")
    # --- what gets distilled ---------------------------------------------------
    parser.add_argument("--dataset", type=str, default="MVTec_feature", help="buffer folder name")
    parser.add_argument("--model", type=str, default="ResNet18", help="buffer sub-folder name")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category")
    parser.add_argument(
        "--data_path", type=str, default="../data/mvtec_anomaly_detection", help="MVTec root"
    )
    parser.add_argument("--buffer_path", type=str, default="../buffer", help="buffer path")
    parser.add_argument(
        "--per_category_buffer",
        action="store_true",
        help="save to <buffer>/<dataset>/<category>/<model> instead of <buffer>/<dataset>/<model>",
    )
    # --- student / teacher -----------------------------------------------------
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--res", type=int, default=64, help="training resolution")
    parser.add_argument(
        "--normalize", action="store_true", help="apply ImageNet mean/std normalisation"
    )
    parser.add_argument(
        "--bn_track_stats",
        action="store_true",
        help="keep BatchNorm running stats in the student (NOT stored in the trajectory)",
    )
    parser.add_argument(
        "--no_moviad", action="store_true", help="always build the backbone with torchvision"
    )
    # --- trajectory generation (same knobs as buffer_FTD.py) -------------------
    parser.add_argument("--num_experts", type=int, default=10, help="number of expert trajectories")
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--snap_every", type=int, default=1, help="epochs between saved snapshots (strong experts: 5)")
    parser.add_argument("--save_interval", type=int, default=1, help="trajectories per buffer file")
    parser.add_argument("--lr_teacher", type=float, default=0.01)
    parser.add_argument("--batch_train", type=int, default=32)
    parser.add_argument("--mom", type=float, default=0.0, help="momentum")
    parser.add_argument("--l2", type=float, default=0.0, help="l2 regularization")
    parser.add_argument("--optimizer", type=str, default="gsam", choices=["gsam", "sgd"])
    parser.add_argument("--rho_max", default=0.01, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--rho_min", default=0.01, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.3, type=float, help="GSAM alpha.")
    parser.add_argument("--adaptive", default=True, type=bool, help="use Adaptive SAM.")
    # --- augmentation ----------------------------------------------------------
    parser.add_argument("--dsa", type=str, default="False", choices=["True", "False"])
    parser.add_argument(
        "--dsa_strategy", type=str, default="color_crop_cutout_flip_scale_rotate"
    )
    parser.add_argument("--patch_split", action="store_true", help="Split 224x224 images into four 112x112 patches")
    parser.add_argument("--quadrant", type=int, default=None, choices=[0,1,2,3],
        help="Estrae UN solo quadrante (0=TL,1=TR,2=BL,3=BR) invece del pool di tutti")
    args = parser.parse_args()
    main(args)
