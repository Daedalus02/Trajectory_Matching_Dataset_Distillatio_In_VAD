"""
DINOv2 encoder loader, trimmed from Dinomaly's ``models/vit_encoder.py``.

Only the ``dinov2[reg]_vit_{small,base,large}_14`` branch is kept (the DINOv1 /
BEiT / MAE / ... branches pulled in extra dependencies we do not need).

Checkpoints are fetched from dl.fbaipublicfiles.com into the weights directory,
which defaults to ``<VAD_distill>/backbones/weights`` and can be overridden with
the ``DINOMALY_WEIGHTS_DIR`` environment variable. If the machine cannot reach
the internet, download the ``.pth`` file elsewhere and drop it in that folder:
``load`` uses the cached file without touching the network.
"""

import logging
import os

import torch
from torch.hub import HASH_REGEX, download_url_to_file, urlparse

from .dinov2.models import vision_transformer as vision_transformer_dinov2

_logger = logging.getLogger(__name__)

_WEIGHTS_DIR = os.environ.get(
    "DINOMALY_WEIGHTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "backbones", "weights"),
)

_ARCH_SHORT = {"small": "s", "base": "b", "large": "l"}


def checkpoint_url(name):
    """URL of the official DINOv2 checkpoint for ``name`` (e.g. 'dinov2reg_vit_small_14')."""
    arch, patchsize = name.split("_")[-2], name.split("_")[-1]
    if "dinov2" not in name:
        raise ValueError("Only DINOv2 encoders are supported, got {}".format(name))
    if arch not in _ARCH_SHORT:
        raise ValueError("Invalid architecture {}: must be 'small', 'base' or 'large'.".format(arch))
    a = _ARCH_SHORT[arch]
    suffix = "_reg4_pretrain.pth" if "reg" in name else "_pretrain.pth"
    return "https://dl.fbaipublicfiles.com/dinov2/dinov2_vit{a}{p}/dinov2_vit{a}{p}{s}".format(
        a=a, p=patchsize, s=suffix)


def load(name):
    arch, patchsize = name.split("_")[-2], name.split("_")[-1]
    kwargs = dict(patch_size=int(patchsize), img_size=518, block_chunks=0, init_values=1e-8,
                  interpolate_antialias=False, interpolate_offset=0.1)
    if "reg" in name:
        kwargs["num_register_tokens"] = 4
    model = vision_transformer_dinov2.__dict__[f'vit_{arch}'](**kwargs)

    ckpt_pth = download_cached_file(checkpoint_url(name))
    state_dict = torch.load(ckpt_pth, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model


def download_cached_file(url, check_hash=True, progress=True):
    """
    Mostly copy-paste from timm library.
    (https://github.com/rwightman/pytorch-image-models/blob/29fda20e6d428bf636090ab207bbcf60617570ca/timm/models/_hub.py#L54)
    """
    os.makedirs(_WEIGHTS_DIR, exist_ok=True)
    filename = os.path.basename(urlparse(url).path)
    cached_file = os.path.join(_WEIGHTS_DIR, filename)
    if not os.path.exists(cached_file):
        _logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file
