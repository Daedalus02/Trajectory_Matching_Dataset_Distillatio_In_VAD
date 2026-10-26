"""Shared code for STFPM distillation: data loading, teacher/student, loss, training and scoring.

Every script builds the student with build_student, so its flat parameters match the expert
trajectories. Student BatchNorms have no running statistics: trajectories store parameters only.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_LAYERS = ["layer1", "layer2", "layer3"]
# Since data used to train backbones (ResNet18 is ImageNet).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_transform(img_size, normalize=False):
    """Resize + ToTensor (+ optional ImageNet normalisation).

    ``normalize=False`` is the default because ``eval_stfpm_auroc.py`` keeps the
    normalisation commented out; keeping the same convention everywhere means the
    buffers, the distillation and the evaluation all see the same pixel range.
    """
    ops = [transforms.Resize((img_size, img_size), antialias=True), transforms.ToTensor()]
    if normalize:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


class FolderImageDataset(Dataset):
    """Every image under ``root`` (recursively).

    Labels follow the MVTec convention used by ``eval_stfpm_auroc.py``:
    ``0`` when the image lives in a ``good/`` folder, ``1`` otherwise.
    """

    def __init__(self, root, img_size=64, normalize=False, return_label=False):
        self.root = str(root)
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError("No images found in {}".format(root))
        self.transform = build_transform(img_size, normalize)
        self.return_label = return_label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = self.transform(Image.open(path).convert("RGB"))
        if not self.return_label:
            return image
        return image, 0 if path.parent.name == "good" else 1, str(path)


def load_images_tensor(root, img_size=64, normalize=False, verbose=True):
    """Load a whole folder of images into a single ``(N, 3, H, W)`` CPU tensor."""
    dataset = FolderImageDataset(root, img_size=img_size, normalize=normalize)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    images = torch.cat([batch for batch in loader], dim=0)
    if verbose:
        print("Loaded {} images from {} at {}x{}".format(len(images), root, img_size, img_size))
        for ch in range(images.shape[1]):
            print(
                "  channel %d: mean = %.4f, std = %.4f"
                % (ch, images[:, ch].mean(), images[:, ch].std())
            )
    return images


# ---------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------
def _build_with_moviad(backbone, layers, device, frozen):
    from moviad.moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor

    return CustomFeatureExtractor(backbone, list(layers), device, frozen=frozen).model


def _build_with_torchvision(backbone, layers, device, pretrained):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = getattr(torchvision.models, backbone)(weights=weights)
    return create_feature_extractor(model, {layer: layer for layer in layers}).to(device)


def build_backbone(backbone, layers, device, pretrained, prefer_moviad=True):
    """Return ``(module, source)`` -- a trimmed backbone returning a dict of features.

    Tries MoViAD's ``CustomFeatureExtractor`` first (as requested by the task) and
    falls back to an equivalent torchvision construction when MoViAD is not
    importable (e.g. one of its optional backbone dependencies is missing).
    Both paths build the same module, so parameter layouts stay compatible.
    """
    if prefer_moviad:
        try:
            return _build_with_moviad(backbone, layers, device, pretrained), "moviad"
        except Exception as exc:  # pragma: no cover - environment dependent
            warnings.warn(
                "Falling back to torchvision feature extractor "
                "(MoViAD CustomFeatureExtractor unavailable: {})".format(exc)
            )
    return _build_with_torchvision(backbone, layers, device, pretrained), "torchvision"


class FeatureNet(nn.Module):
    """Wrap a feature extractor so ``forward`` returns a *list* of feature maps."""

    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = list(layers)

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return [out[layer] for layer in self.layers]
        if isinstance(out, (list, tuple)):
            return list(out)
        return [out]


def disable_bn_running_stats(model):
    """Make every BatchNorm use batch statistics and drop its buffers."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            module.num_batches_tracked = None
    return model


def build_teacher(backbone="resnet18", layers=DEFAULT_LAYERS, device="cuda", prefer_moviad=True):
    """Pretrained, frozen teacher in eval mode (keeps its BN running statistics)."""
    model, source = build_backbone(
        backbone, layers, device, pretrained=True, prefer_moviad=prefer_moviad
    )
    teacher = FeatureNet(model, layers).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    print("Teacher: {} {} (frozen, source={})".format(backbone, list(layers), source))
    return teacher


def build_student(
    backbone="resnet18",
    layers=DEFAULT_LAYERS,
    device="cuda",
    bn_track_stats=False,
    prefer_moviad=True,
    verbose=False,
):
    """Randomly initialised, trainable student."""
    model, source = build_backbone(
        backbone, layers, device, pretrained=False, prefer_moviad=prefer_moviad
    )
    student = FeatureNet(model, layers).to(device)
    for param in student.parameters():
        param.requires_grad_(True)
    if not bn_track_stats:
        disable_bn_running_stats(student)
    if verbose:
        print(
            "Student: {} {} (source={}, params={}, buffers={})".format(
                backbone, list(layers), source, count_parameters(student), count_buffers(student)
            )
        )
    return student


def count_parameters(net):
    return int(sum(np.prod(p.size()) for p in net.parameters()))


def count_buffers(net):
    return int(sum(np.prod(b.size()) for b in net.buffers()))


# ---------------------------------------------------------------------------
# losses / parameters
# ---------------------------------------------------------------------------
def feature_mse_loss(teacher_features, student_features):
    """Raw (un-normalised) MSE between matching teacher/student feature maps."""
    return sum(
        F.mse_loss(s_feat, t_feat)
        for t_feat, s_feat in zip(teacher_features, student_features)
    )


def parameter_snapshot(net):
    """Trajectory timestamp, in the same format the original DATM buffers use.

    ``copy=True`` matters: ``.cpu()`` is a no-op that returns the *same* tensor
    when the model already lives on the CPU, which would make every timestamp of
    a trajectory alias the final parameters (and hence ``param_dist == 0`` during
    distillation). The original ``buffer_FTD.py`` only ever runs on CUDA, where
    ``.cpu()`` implicitly copies.
    """
    return [p.detach().to("cpu", copy=True) for p in net.parameters()]


def flatten_snapshot(snapshot, device):
    """Flatten a trajectory timestamp into the vector consumed by ReparamModule."""
    return torch.cat([p.data.to(device).reshape(-1) for p in snapshot], 0)


def check_trajectory_layout(reparam_net, snapshot):
    """Fail loudly if a buffer was generated with a different student topology."""
    expected = [tuple(s) for s in reparam_net._param_shapes]
    got = [tuple(p.shape) for p in snapshot]
    if expected != got:
        raise AssertionError(
            "Expert trajectory does not match the student parameter layout "
            "({} tensors / {} params in the buffer vs {} tensors / {} params in the student). "
            "Regenerate the buffers with the same --backbone/--layers/--bn_track_stats.".format(
                len(got),
                sum(int(np.prod(g)) for g in got),
                len(expected),
                reparam_net.param_numel,
            )
        )


def get_expert_dir(buffer_path, dataset, model, category=None, per_category=False):
    """``<buffer_path>/<dataset>[/<category>]/<model>`` -- shared by both scripts."""
    path = os.path.join(buffer_path, dataset)
    if per_category and category:
        path = os.path.join(path, category)
    return os.path.join(path, model)


# ---------------------------------------------------------------------------
# STFPM anomaly scoring / evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def anomaly_scores(teacher, student, batch, out_shape=None):
    """STFPM post-processing: product of the per-layer normalised feature distances."""
    out_shape = out_shape or batch.shape[2:]
    teacher_features = teacher(batch)
    student_features = student(batch)
    score_maps = torch.tensor([1.0], device=batch.device)
    for t_feat, s_feat in zip(teacher_features, student_features):
        t_feat = F.normalize(t_feat, dim=1)
        s_feat = F.normalize(s_feat, dim=1)
        score_map = torch.sum((t_feat - s_feat) ** 2, 1, keepdim=True)
        score_map = F.interpolate(score_map, size=out_shape, mode="bilinear", align_corners=False)
        score_maps = score_maps * score_map
    scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
    return score_maps, scores


def train_student_features(
    images,
    teacher,
    device,
    epochs=100,
    lr=0.01,
    batch_size=32,
    momentum=0.9,
    weight_decay=5e-4,
    backbone="resnet18",
    layers=DEFAULT_LAYERS,
    bn_track_stats=False,
    prefer_moviad=True,
    augment=None,
    student=None,
    lr_decay=True,
    verbose=True,
):
    """Train a (fresh, by default) student on ``images`` with the feature MSE loss."""
    if student is None:
        student = build_student(
            backbone=backbone,
            layers=layers,
            device=device,
            bn_track_stats=bn_track_stats,
            prefer_moviad=prefer_moviad,
        )
    student.train()
    optimizer = torch.optim.SGD(
        student.parameters(), lr=float(lr), momentum=momentum, weight_decay=weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2 + 1], gamma=0.1)
        if lr_decay
        else None
    )
    loader = DataLoader(
        TensorDataset(images.detach().cpu()),
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
                teacher_features = teacher(batch)
            student_features = student(batch)
            loss = feature_mse_loss(teacher_features, student_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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


@torch.no_grad()
def evaluate_auroc(
    student, teacher, test_root, img_size, device, normalize=False, batch_size=1, verbose=True
):
    """Image-level AUROC on an MVTec ``test/`` folder (``good`` + anomalous subfolders)."""
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
    teacher.eval()
    labels, scores = [], []
    for images, batch_labels in loader:
        _, batch_scores = anomaly_scores(teacher, student, images.to(device))
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
    teacher,
    test_root,
    device,
    img_size,
    epochs=100,
    lr=0.01,
    batch_size=32,
    backbone="resnet18",
    layers=DEFAULT_LAYERS,
    bn_track_stats=False,
    prefer_moviad=True,
    normalize=False,
    augment=None,
    eval_batch_size=1,
    verbose=True,
):
    """Train a fresh student on the synthetic images, then report MVTec AUROC.

    This is the feature-matching replacement for DATM's ``evaluate_synset``; it
    replicates the logic of ``eval_stfpm_auroc.py`` on in-memory tensors.
    """
    student = train_student_features(
        image_syn.detach().cpu(),
        teacher,
        device,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        backbone=backbone,
        layers=layers,
        bn_track_stats=bn_track_stats,
        prefer_moviad=prefer_moviad,
        augment=augment,
        verbose=verbose,
    )
    result = evaluate_auroc(
        student,
        teacher,
        test_root,
        img_size=img_size,
        device=device,
        normalize=normalize,
        batch_size=eval_batch_size,
        verbose=verbose,
    )
    del student
    return result
