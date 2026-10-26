"""Shared code for Dinomaly distillation: encoder, trainable student, loss, training and scoring.

Every script builds the student with build_dinomaly_student, so its flat parameters match the
expert trajectories. Plain SGD everywhere (experts, inner loop, evaluation). Images are in [0, 1];
the encoder applies the ImageNet normalisation. Anomaly map: per-pixel cosine distance between
fused encoder and decoder features, smoothed (sigma 4); image score = max of the map.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import gaussian_blur

from lib.components.dinomaly.build import (
    DinomalyStudent,
    build_bottleneck_decoder,
    encode,
    init_trainable,
    load_frozen_encoder,
    target_layers_for,
)
from lib.components.dinomaly.loss import global_cosine_hm_percent_meta

# re-exported so the Dinomaly scripts import their helpers from one place
from lib.rd4ad_matching import (  # noqa: F401
    IMG_EXTENSIONS,
    FolderImageDataset,
    blur_kernel_size,
    check_trajectory_layout,
    count_parameters,
    flatten_snapshot,
    get_expert_dir,
    load_images_tensor,
    parameter_snapshot,
)
from lib.feature_matching import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_BACKBONE = "dinov2reg_vit_small_14"
BACKBONE_CHOICES = ["dinov2reg_vit_small_14", "dinov2reg_vit_base_14", "dinov2reg_vit_large_14"]
DEFAULT_RES = 224
DEFAULT_BLUR_SIGMA = 4.0
DEFAULT_HM_P = 0.9
DEFAULT_HM_FACTOR = 0.1


# ---------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------
class DinomalyEncoder(nn.Module):
    """Frozen DINOv2 encoder. ``forward`` takes ``[0, 1]`` images and returns the
    token list at the target layers (graph to the input kept, see module notes)."""

    def __init__(self, vit, target_layers, checkpoint=False):
        super().__init__()
        self.vit = vit
        self.target_layers = target_layers
        self.checkpoint = checkpoint
        self.num_register_tokens = vit.num_register_tokens
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return encode(self.vit, x, self.target_layers, checkpoint=self.checkpoint)


def check_res(res, backbone=DEFAULT_BACKBONE):
    patch = int(backbone.split("_")[-1])
    if res % patch != 0:
        raise ValueError("--res {} is not a multiple of the ViT patch size {}".format(res, patch))


def build_encoder(backbone=DEFAULT_BACKBONE, device="cuda", checkpoint=False, verbose=False):
    """Pretrained, frozen DINOv2 encoder in eval mode."""
    vit = load_frozen_encoder(backbone, device)
    net = DinomalyEncoder(vit, target_layers_for(backbone), checkpoint=checkpoint).to(device)
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    if verbose:
        print("Encoder: {} (frozen, params={}, checkpoint={})".format(
            backbone, count_parameters(net), checkpoint))
    return net


def build_dinomaly_student(backbone=DEFAULT_BACKBONE, device="cuda", verbose=False):
    """Freshly initialised (Dinomaly init) trainable bottleneck + decoder."""
    bottleneck, decoder = build_bottleneck_decoder(backbone)
    trainable = nn.ModuleList([bottleneck, decoder])
    init_trainable(trainable)
    num_register_tokens = 4 if "reg" in backbone else 0
    student = DinomalyStudent(bottleneck, decoder, num_register_tokens).to(device)
    for param in student.parameters():
        param.requires_grad_(True)
    if verbose:
        n_bn = count_parameters(student.bottleneck)
        n_dec = count_parameters(student.decoder)
        print("Dinomaly student: {} (params bottleneck={} / decoder={} / tot={})".format(
            backbone, n_bn, n_dec, count_parameters(student)))
    return student


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def dinomaly_loss(student_module, en_list, de_maps, p=DEFAULT_HM_P, factor=DEFAULT_HM_FACTOR):
    """Dinomaly hard-mining cosine loss between fused encoder maps and decoder maps.

    ``student_module`` is the plain :class:`DinomalyStudent` (``reparam.module``
    when wrapped), used only for its parameter-free ``fuse_encoder``. The target
    (encoder) side is always detached, as in Dinomaly.
    """
    en_maps = student_module.fuse_encoder(en_list)
    return global_cosine_hm_percent_meta(en_maps, de_maps, p=p, factor=factor)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_dinomaly(
    images,
    encoder,
    device,
    epochs=200,
    lr=0.01,
    batch_size=16,
    momentum=0.0,
    weight_decay=0.0,
    backbone=DEFAULT_BACKBONE,
    p=DEFAULT_HM_P,
    factor=DEFAULT_HM_FACTOR,
    student=None,
    verbose=True,
):
    """Train a (fresh, by default) Dinomaly bottleneck + decoder on ``images`` with SGD."""
    images = images.detach().cpu()
    if student is None:
        student = build_dinomaly_student(backbone=backbone, device=device)
    student.train()
    encoder.eval()
    opt = torch.optim.SGD(student.parameters(), lr=float(lr), momentum=momentum,
                          weight_decay=weight_decay)
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
            with torch.no_grad():
                en_list = encoder(batch)
            loss = dinomaly_loss(student, en_list, student(en_list), p=p, factor=factor)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
            print("    eval-train epoch {:4d}/{} | loss {:.6f}".format(
                epoch + 1, epochs, epoch_loss / max(n_batches, 1)))
    return student


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def anomaly_scores(encoder, student, batch, out_shape=None, blur_sigma=DEFAULT_BLUR_SIGMA):
    """Dinomaly anomaly map (mean per-pixel cosine distance over the fused groups),
    upsampled to ``out_shape`` and smoothed. Returns ``(anomaly_maps, scores)``."""
    out_shape = out_shape or batch.shape[2:]
    en_list = encoder(batch)
    en_maps = student.fuse_encoder(en_list)
    de_maps = student(en_list)
    maps = []
    for f_en, f_de in zip(en_maps, de_maps):
        a_map = (1 - F.cosine_similarity(f_en, f_de)).unsqueeze(1)
        maps.append(F.interpolate(a_map, size=out_shape, mode="bilinear", align_corners=True))
    anomaly_map = torch.cat(maps, dim=1).mean(dim=1, keepdim=True)
    if blur_sigma and blur_sigma > 0:
        kernel = blur_kernel_size(blur_sigma, min(anomaly_map.shape[-2:]))
        if kernel > 1:
            anomaly_map = gaussian_blur(anomaly_map, [kernel, kernel], [blur_sigma, blur_sigma])
    scores = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1)[0]
    return anomaly_map, scores


@torch.no_grad()
def evaluate_auroc(student, encoder, test_root, img_size, device, batch_size=16,
                   blur_sigma=DEFAULT_BLUR_SIGMA, verbose=True):
    """Image-level AUROC on an MVTec ``test/`` folder (``good`` + anomalous)."""
    from sklearn import metrics

    dataset = FolderImageDataset(test_root, img_size=img_size, return_label=True)
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
        _, batch_scores = anomaly_scores(encoder, student, images.to(device), blur_sigma=blur_sigma)
        scores.extend(batch_scores.cpu().tolist())
        labels.extend(batch_labels.tolist())

    auroc = float(metrics.roc_auc_score(labels, scores))
    good = [s for s, l in zip(scores, labels) if l == 0]
    bad = [s for s, l in zip(scores, labels) if l == 1]
    if verbose:
        print("  test images: {} ({} good / {} anomalous)".format(len(labels), len(good), len(bad)))
        if good:
            print("  good  - mean %.6f min %.6f max %.6f" % (np.mean(good), np.min(good), np.max(good)))
        if bad:
            print("  anom  - mean %.6f min %.6f max %.6f" % (np.mean(bad), np.min(bad), np.max(bad)))
        print("  AUROC: %.4f" % auroc)
    return {"auroc": auroc, "scores": scores, "labels": labels}


def evaluate_syn_auroc(image_syn, encoder, test_root, device, img_size, epochs=50, lr=0.01,
                       batch_size=16, backbone=DEFAULT_BACKBONE, p=DEFAULT_HM_P,
                       factor=DEFAULT_HM_FACTOR, eval_batch_size=16,
                       blur_sigma=DEFAULT_BLUR_SIGMA, verbose=True):
    """Train a fresh Dinomaly on the synthetic images, then report MVTec AUROC."""
    student = train_dinomaly(
        image_syn.detach().cpu().clamp(0, 1), encoder, device, epochs=epochs, lr=lr,
        batch_size=batch_size, backbone=backbone, p=p, factor=factor, verbose=verbose,
    )
    result = evaluate_auroc(student, encoder, test_root, img_size=img_size, device=device,
                            batch_size=eval_batch_size, blur_sigma=blur_sigma, verbose=verbose)
    del student
    return result


def gpu_peak_report(device):
    """One-line peak GPU memory summary (empty on CPU)."""
    if device != "cuda":
        return "Peak GPU memory: n/a (cpu)"
    return "Peak GPU memory: {:.2f} GB allocated / {:.2f} GB reserved".format(
        torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9)
