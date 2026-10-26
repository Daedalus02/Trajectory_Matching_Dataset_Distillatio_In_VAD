"""Expert trajectories for RD4AD distillation.

Trains OCBE bn + decoder against the frozen encoder with plain SGD (the step the inner loop
reproduces); one snapshot per epoch. --half_buffer stores fp16 snapshots.
"""

# NOTE: bn is used in order to avoide the decoder to just copy the encoder behavior, it has to learns only to copy the normal data not the anomalies
# it uses multiscale feature fusion (encoder layer 1-3) and one classs embedding

import argparse
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.rd4ad_matching import (
    DEFAULT_BACKBONE,
    build_encoder,
    build_rd4ad_student,
    build_trainer_optimizer,
    count_parameters,
    get_expert_dir,
    load_images_tensor,
    parameter_snapshot,
    rd4ad_cosine_loss,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_augmentation(args):
    """DSA augmentation, imported lazily so the script also runs without kornia."""
    if not args.dsa:
        return None
    from lib.utils_baseline import DiffAugment, ParamDiffAug

    dsa_param = ParamDiffAug()
    return lambda images: DiffAugment(images, args.dsa_strategy, param=dsa_param)


def train_epoch(args, loader, encoder, student, optimizer, augment):
    """One pass over the real images with the RD4AD cosine loss."""
    student.train()
    encoder.eval()
    loss_avg, num_exp = 0.0, 0

    for images in loader:
        images = images.float().to(args.device)
        if augment is not None:
            images = augment(images)
        n_b = images.shape[0]

        with torch.no_grad():
            enc_features = encoder(images)
        dec_features = student(enc_features)
        loss = rd4ad_cosine_loss(enc_features, dec_features)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_avg += loss.item() * n_b
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

    # frozen target: the RD4AD analogue of the STFPM teacher
    encoder = build_encoder(
        backbone=args.backbone, device=args.device, pretrained=True, verbose=True
    )
    augment = build_augmentation(args)
    if args.dsa:
        print("DSA augmentation strategy: {}".format(args.dsa_strategy))

    trajectories = []

    for it in range(0, args.num_experts):

        """ Train an expert RD4AD (bn + decoder) on the real data """
        student = build_rd4ad_student(
            backbone=args.backbone,
            device=args.device,
            bn_track_stats=args.bn_track_stats,
            verbose=(it == 0),
        )
        if it == 0:
            n_params = count_parameters(student)
            print(
                "Trajectory snapshot: {} tensors, {} parameters (~{:.1f} MB each)".format(
                    len(list(student.parameters())),
                    n_params,
                    n_params * (2 if args.half_buffer else 4) / 1e6,
                )
            )

        optimizer = build_trainer_optimizer(
            student, args.optimizer, args.lr_teacher, momentum=args.mom, weight_decay=args.l2
        )
        if it == 0:
            print("Expert training scheme: {}".format(args.optimizer))

        timestamps = []
        timestamps.append(parameter_snapshot(student, half=args.half_buffer))

        for e in range(args.train_epochs):
            loss = train_epoch(args, trainloader, encoder, student, optimizer, augment)
            print("Itr: {}\tEpoch: {}\tCosine Loss: {:.6f}".format(it, e, loss))
            timestamps.append(parameter_snapshot(student, half=args.half_buffer))

        trajectories.append(timestamps)
        del student, optimizer
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
    parser = argparse.ArgumentParser(description="RD4AD expert trajectories")
    # --- what gets distilled ---------------------------------------------------
    parser.add_argument("--dataset", type=str, default="MVTec_rd4ad", help="buffer folder name")
    parser.add_argument("--model", type=str, default="RD4AD", help="buffer sub-folder name")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category")
    parser.add_argument(
        "--data_path", type=str, default="../data/mvtec_anomaly_detection", help="MVTec root"
    )
    parser.add_argument("--buffer_path", type=str, default="../outputs/buffer_rd4ad")
    parser.add_argument(
        "--per_category_buffer",
        action="store_true",
        help="save to <buffer>/<dataset>/<category>/<model> instead of <buffer>/<dataset>/<model>",
    )
    # --- RD4AD architecture ----------------------------------------------------
    parser.add_argument(
        "--backbone",
        type=str,
        default=DEFAULT_BACKBONE,
        choices=["resnet18", "resnet34", "resnet50", "wide_resnet50_2"],
        help="encoder/decoder pair (the encoder half is frozen and pretrained)",
    )
    parser.add_argument("--res", type=int, default=64, help="training resolution")
    parser.add_argument(
        "--normalize", action="store_true", help="apply ImageNet mean/std normalisation"
    )
    parser.add_argument(
        "--bn_track_stats",
        action="store_true",
        help="keep BatchNorm running stats (NOT stored in the trajectory)",
    )
    # --- trajectory generation (same knobs as generate_buffer.py) --------------
    parser.add_argument("--num_experts", type=int, default=10, help="number of expert trajectories")
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1, help="trajectories per buffer file")
    parser.add_argument("--lr_teacher", type=float, default=0.01)
    parser.add_argument("--batch_train", type=int, default=32)
    parser.add_argument("--mom", type=float, default=0.0, help="momentum (sgd only)")
    parser.add_argument("--l2", type=float, default=0.0, help="l2 regularization (sgd only)")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="sgd = the step the distillation inner loop reproduces, "
             "adam = reference RD4AD trainer (lr ~5e-3, betas=(0.5, 0.999))",
    )
    parser.add_argument(
        "--half_buffer", action="store_true", help="store the snapshots in fp16 (half the disk)"
    )
    # --- augmentation ----------------------------------------------------------
    parser.add_argument("--dsa", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--dsa_strategy", type=str, default="color_crop_cutout_flip_scale_rotate")
    parser.add_argument(
        "--quadrant",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Estrae UN solo quadrante (0=TL,1=TR,2=BL,3=BR) invece del pool di tutti",
    )
    args = parser.parse_args()
    main(args)
