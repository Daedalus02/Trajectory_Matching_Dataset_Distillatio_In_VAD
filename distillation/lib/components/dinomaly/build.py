"""Dinomaly model, split into a frozen encoder pass and a trainable half (bottleneck + decoder).

The encoder is run with the graph alive (parameters frozen) so that the meta-gradient
reaches the synthetic images; ViTill.forward uses no_grad and cannot be used for distillation.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint

from . import vit_encoder
from .uad import ViTill
from .vision_transformer import Block as VitBlock, bMlp, LinearAttention2

TARGET_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]
FUSE_LAYER_ENCODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
FUSE_LAYER_DECODER = [[0, 1, 2, 3], [4, 5, 6, 7]]


def arch_dims(encoder_name):
    if 'small' in encoder_name:
        return 384, 6
    if 'base' in encoder_name:
        return 768, 12
    if 'large' in encoder_name:
        return 1024, 16
    raise ValueError("Architecture not in small, base, large.")


def target_layers_for(encoder_name):
    if 'large' in encoder_name:
        return [4, 6, 8, 10, 12, 14, 16, 18]
    return list(TARGET_LAYERS)


def init_trainable(trainable):
    """Dinomaly's init of bottleneck+decoder (trunc_normal std 0.01, zero bias, LN = id)."""
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def build_bottleneck_decoder(encoder_name, num_decoder_blocks=8, bottleneck_drop=0.2):
    embed_dim, num_heads = arch_dims(encoder_name)
    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=bottleneck_drop)])
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                 attn=LinearAttention2)
        for _ in range(num_decoder_blocks)
    ])
    return bottleneck, decoder


def load_frozen_encoder(encoder_name, device):
    encoder = vit_encoder.load(encoder_name).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def build_vitill(encoder_name='dinov2reg_vit_small_14', device='cuda', encoder=None):
    """Full ViTill as in dinomaly_mvtec_sep.py. Returns (model, encoder, trainable)."""
    if encoder is None:
        encoder = load_frozen_encoder(encoder_name, device)
    bottleneck, decoder = build_bottleneck_decoder(encoder_name)
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=target_layers_for(encoder_name), mask_neighbor_size=0,
                   fuse_layer_encoder=FUSE_LAYER_ENCODER, fuse_layer_decoder=FUSE_LAYER_DECODER)
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])
    init_trainable(trainable)
    return model, encoder, trainable


def encode(encoder, x, target_layers, checkpoint=False):
    """Encoder tokens at ``target_layers`` (class + register tokens included), graph to x kept."""
    x = encoder.prepare_tokens(x)
    en_list = []
    for i, blk in enumerate(encoder.blocks):
        if i > target_layers[-1]:
            break
        if checkpoint:
            x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        else:
            x = blk(x)
        if i in target_layers:
            en_list.append(x)
    return en_list


class DinomalyStudent(nn.Module):
    """Trainable half of ViTill (bottleneck + decoder + fusion), fed with encoder tokens."""

    def __init__(self, bottleneck, decoder, num_register_tokens,
                 fuse_layer_encoder=FUSE_LAYER_ENCODER, fuse_layer_decoder=FUSE_LAYER_DECODER):
        super().__init__()
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.num_register_tokens = num_register_tokens
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder

    @staticmethod
    def fuse_feature(feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def _to_maps(self, feats):
        skip = 1 + self.num_register_tokens
        feats = [f[:, skip:, :] for f in feats]
        side = int(math.sqrt(feats[0].shape[1]))
        return [f.permute(0, 2, 1).reshape([f.shape[0], -1, side, side]).contiguous() for f in feats]

    def fuse_encoder(self, en_list):
        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        return self._to_maps(en)

    def forward(self, en_list):
        x = self.fuse_feature(en_list)
        for blk in self.bottleneck:
            x = blk(x)
        de_list = []
        for blk in self.decoder:
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]
        return self._to_maps(de)
