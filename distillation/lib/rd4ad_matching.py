"""Shared code for RD4AD distillation: encoder, trainable half (OCBE bn + decoder), loss, scoring.

Encoder and decoder are in series, so the encoder runs with the graph alive (parameters frozen)
and only the target features are detached. Student BatchNorms have no running statistics.
Training loss: cosine on flattened maps; anomaly map: per-pixel cosine distance, smoothed.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import gaussian_blur

from lib.components.rd4ad.deresnet import (
    de_resnet18,
    de_resnet34,
    de_resnet50,
    de_wide_resnet50_2,
)
from lib.components.rd4ad.resnet import (
    resnet18,
    resnet34,
    resnet50,
    wide_resnet50_2,
)

# re-exported so the RD4AD scripts import their data helpers from one place
from lib.feature_matching import (  # noqa: F401
    IMG_EXTENSIONS,
    FolderImageDataset,
    build_transform,
    check_trajectory_layout,
    count_buffers,
    count_parameters,
    disable_bn_running_stats,
    get_expert_dir,
    load_images_tensor,
)

DEFAULT_BACKBONE = "resnet18"
DEFAULT_BLUR_SIGMA = 4.0

# backbone -> (encoder + OCBE bottleneck factory, decoder factory)
BACKBONES = {
    "resnet18": (resnet18, de_resnet18),
    "resnet34": (resnet34, de_resnet34),
    "resnet50": (resnet50, de_resnet50),
    "wide_resnet50_2": (wide_resnet50_2, de_wide_resnet50_2),
}


def _factories(backbone):
    if backbone not in BACKBONES:
        raise ValueError(
            "Unknown backbone '{}' (available: {})".format(backbone, sorted(BACKBONES))
        )
    return BACKBONES[backbone]


# ---------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------
class RD4ADEncoder(nn.Module):
    """Frozen ImageNet encoder -- the RD4AD analogue of the STFPM teacher.

    ``forward`` returns the ``[layer1, layer2, layer3]`` feature maps.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)


class RD4ADStudent(nn.Module):
    """The trainable half of RD4AD: OCBE bottleneck + reversed decoder.

    ``forward`` takes the encoder feature list and returns the decoder feature
    list in the *same* layer order, so the two can simply be zipped together.
    """

    def __init__(self, bn, decoder):
        super().__init__()
        self.bn = bn
        self.decoder = decoder

    def forward(self, enc_features):
        return self.decoder(self.bn(enc_features))


def build_encoder(backbone=DEFAULT_BACKBONE, device="cuda", pretrained=True, verbose=False):
    """Pretrained, frozen encoder in eval mode (keeps its BN running statistics)."""
    encoder, _ = _factories(backbone)[0](pretrained=pretrained)
    net = RD4ADEncoder(encoder).to(device)
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    if verbose:
        print(
            "Encoder: {} (frozen, pretrained={}, params={})".format(
                backbone, pretrained, count_parameters(net)
            )
        )
    return net


def build_rd4ad_student(
    backbone=DEFAULT_BACKBONE, device="cuda", bn_track_stats=False, verbose=False
):
    """Randomly initialised, trainable ``bn`` + ``decoder`` half of RD4AD."""
    encoder_fn, decoder_fn = _factories(backbone)
    # the encoder half is thrown away here: only the (randomly initialised) OCBE
    # bottleneck shipped alongside it is kept, so pretrained=False is enough
    _, bottleneck = encoder_fn(pretrained=False)
    decoder = decoder_fn(pretrained=False)
    student = RD4ADStudent(bottleneck, decoder).to(device)
    for param in student.parameters():
        param.requires_grad_(True)
    if not bn_track_stats:
        disable_bn_running_stats(student)
    if verbose:
        n_bn = int(sum(p.numel() for p in student.bn.parameters()))
        n_dec = int(sum(p.numel() for p in student.decoder.parameters()))
        print(
            "RD4AD student: {} (params bn={} / decoder={} / tot={}, buffers={})".format(
                backbone, n_bn, n_dec, count_parameters(student), count_buffers(student)
            )
        )
    return student


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def rd4ad_cosine_loss(enc_features, dec_features, detach_target=True):
    """RD4AD training loss: ``sum_l mean(1 - cos(flat(f_dec), flat(f_enc)))``.

    ``detach_target=True`` mirrors ``distill.py``, where the teacher features are
    computed under ``torch.no_grad()``: the encoder is frozen either way, but
    during distillation the flag decides whether the meta-gradient also flows
    through the *target* branch (it always flows through the decoder's input).
    """
    loss = 0.0
    for f_enc, f_dec in zip(enc_features, dec_features):
        target = f_enc.detach() if detach_target else f_enc
        loss = loss + torch.mean(
            1
            - F.cosine_similarity(
                target.reshape(target.shape[0], -1), f_dec.reshape(f_dec.shape[0], -1)
            )
        )
    return loss


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------
def parameter_snapshot(net, half=False):
    """Trajectory timestamp, same format as the feature-matching buffers.

    ``copy=True`` matters for the same reason as in :mod:`lib.feature_matching`:
    on CPU ``.cpu()`` would alias the live parameters. ``half=True`` halves the
    (large) RD4AD buffers on disk; :func:`flatten_snapshot` casts back.
    """
    out = []
    for p in net.parameters():
        tensor = p.detach().to("cpu", copy=True)
        out.append(tensor.half() if half else tensor)
    return out


def flatten_snapshot(snapshot, device):
    """Flatten a trajectory timestamp into the vector consumed by ReparamModule."""
    return torch.cat(
        [p.data.to(device=device, dtype=torch.float32).reshape(-1) for p in snapshot], 0
    )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def build_trainer_optimizer(
    net, optimizer, lr, momentum=0.0, weight_decay=0.0, beta1=0.5, beta2=0.999
):
    """``sgd`` reproduces the distillation inner step, ``adam`` the RD4AD paper's."""
    if optimizer == "sgd":
        return torch.optim.SGD(
            net.parameters(), lr=float(lr), momentum=momentum, weight_decay=weight_decay
        )
    return torch.optim.Adam(net.parameters(), lr=float(lr), betas=(beta1, beta2))


def train_rd4ad(
    images,
    encoder,
    device,
    epochs=100,
    lr=0.01,
    batch_size=32,
    momentum=0.0,
    weight_decay=0.0,
    optimizer="sgd",
    backbone=DEFAULT_BACKBONE,
    bn_track_stats=False,
    detach_target=True,
    augment=None,
    student=None,
    lr_decay=False,
    verbose=True,
):
    """Train a (fresh, by default) RD4AD ``bn`` + ``decoder`` on ``images``.

    Feature-matching analogue: :func:`lib.feature_matching.train_student_features`.
    """
    images = images.detach().cpu()
    if student is None:
        student = build_rd4ad_student(
            backbone=backbone, device=device, bn_track_stats=bn_track_stats
        )
    student.train()
    encoder.eval()
    opt = build_trainer_optimizer(
        student, optimizer, lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[epochs // 2 + 1], gamma=0.1)
        if lr_decay
        else None
    )
    loader = DataLoader(
        TensorDataset(images),
        batch_size=max(1, min(batch_size, len(images))),
        shuffle=True,
        num_workers=0,
    )

    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            if augment is not None:
                batch = augment(batch)
            with torch.no_grad():
                enc_features = encoder(batch)
            dec_features = student(enc_features)
            loss = rd4ad_cosine_loss(enc_features, dec_features, detach_target=detach_target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
            print(
                "    eval-train epoch {:4d}/{} | loss {:.6f}".format(
                    epoch + 1, epochs, epoch_loss / max(n_batches, 1)
                )
            )
    return student


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def blur_kernel_size(sigma, max_side):
    """Odd kernel wide enough for ``sigma``, never larger than the map itself."""
    size = int(2 * math.ceil(2.0 * float(sigma)) + 1)
    side = int(max_side)
    limit = max(1, side if side % 2 == 1 else side - 1)
    return max(1, min(size, limit))


@torch.no_grad()
def anomaly_scores(encoder, student, batch, out_shape=None, blur_sigma=DEFAULT_BLUR_SIGMA):
    """RD4AD post-processing: summed per-pixel cosine distances, then smoothed.

    Counterpart of :func:`lib.feature_matching.anomaly_scores`; returns
    ``(anomaly_maps, scores)`` with ``scores`` the peak of each map.
    """
    out_shape = out_shape or batch.shape[2:]
    enc_features = encoder(batch)
    dec_features = student(enc_features)
    anomaly_map = None
    for f_enc, f_dec in zip(enc_features, dec_features):
        a_map = (1 - F.cosine_similarity(f_dec, f_enc)).unsqueeze(1)
        a_map = F.interpolate(a_map, size=out_shape, mode="bilinear", align_corners=True)
        anomaly_map = a_map if anomaly_map is None else anomaly_map + a_map
    if blur_sigma and blur_sigma > 0:
        kernel = blur_kernel_size(blur_sigma, min(anomaly_map.shape[-2:]))
        if kernel > 1:
            anomaly_map = gaussian_blur(anomaly_map, [kernel, kernel], [blur_sigma, blur_sigma])
    scores = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1)[0]
    return anomaly_map, scores


@torch.no_grad()
def evaluate_auroc(
    student,
    encoder,
    test_root,
    img_size,
    device,
    normalize=False,
    batch_size=16,
    blur_sigma=DEFAULT_BLUR_SIGMA,
    verbose=True,
):
    """Image-level AUROC on an MVTec ``test/`` folder (``good`` + anomalous)."""
    from sklearn import metrics

    dataset = FolderImageDataset(
        test_root, img_size=img_size, normalize=normalize, return_label=True
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: (
            torch.stack([i[0] for i in items]),
            torch.tensor([i[1] for i in items]),
        ),
    )
    student.eval()
    encoder.eval()
    labels, scores = [], []
    for images, batch_labels in loader:
        _, batch_scores = anomaly_scores(
            encoder, student, images.to(device), blur_sigma=blur_sigma
        )
        scores.extend(batch_scores.cpu().tolist())
        labels.extend(batch_labels.tolist())

    auroc = float(metrics.roc_auc_score(labels, scores))
    good = [s for s, l in zip(scores, labels) if l == 0]
    bad = [s for s, l in zip(scores, labels) if l == 1]
    if verbose:
        print("  test images: {} ({} good / {} anomalous)".format(len(labels), len(good), len(bad)))
        if good:
            print(
                "  good  - mean %.6f min %.6f max %.6f"
                % (np.mean(good), np.min(good), np.max(good))
            )
        if bad:
            print(
                "  anom  - mean %.6f min %.6f max %.6f" % (np.mean(bad), np.min(bad), np.max(bad))
            )
        print("  AUROC: %.4f" % auroc)
    return {"auroc": auroc, "scores": scores, "labels": labels}


def evaluate_syn_auroc(
    image_syn,
    encoder,
    test_root,
    device,
    img_size,
    epochs=100,
    lr=0.01,
    batch_size=32,
    optimizer="sgd",
    backbone=DEFAULT_BACKBONE,
    bn_track_stats=False,
    detach_target=True,
    normalize=False,
    augment=None,
    eval_batch_size=16,
    blur_sigma=DEFAULT_BLUR_SIGMA,
    verbose=True,
):
    """Train a fresh RD4AD on the synthetic images, then report MVTec AUROC.

    RD4AD replacement for :func:`lib.feature_matching.evaluate_syn_auroc`.
    """
    student = train_rd4ad(
        image_syn.detach().cpu(),
        encoder,
        device,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        optimizer=optimizer,
        backbone=backbone,
        bn_track_stats=bn_track_stats,
        detach_target=detach_target,
        augment=augment,
        verbose=verbose,
    )
    result = evaluate_auroc(
        student,
        encoder,
        test_root,
        img_size=img_size,
        device=device,
        normalize=normalize,
        batch_size=eval_batch_size,
        blur_sigma=blur_sigma,
        verbose=verbose,
    )
    del student
    return result
