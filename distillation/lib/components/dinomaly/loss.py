"""Dinomaly hard-mining cosine loss.

global_cosine_hm_percent: original version, scales the gradient of easy points with a backward hook.
global_cosine_hm_percent_meta: same value and same gradient without hooks
(b_eff = w*b + detach((1-w)*b)), so it can be differentiated twice in the inner loop.
"""

from functools import partial

import torch


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def global_cosine_hm_percent_meta(a, b, p=0.9, factor=0., detach_target=True):
    """Hook-free, double-backward-safe version of ``global_cosine_hm_percent``."""
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach() if detach_target else a[item]
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
            k = int(point_dist.numel() * (1 - p))
            if k > 0:
                thresh = torch.topk(point_dist.reshape(-1), k=k)[0][-1]
                w = torch.where(point_dist < thresh,
                                torch.full_like(point_dist, factor),
                                torch.ones_like(point_dist))
            else:  # p = 1: the original would fail on topk(k=0); no point is "hard"
                w = torch.full_like(point_dist, factor)
        b_eff = w * b_ + ((1 - w) * b_).detach()

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_eff.reshape(b_eff.shape[0], -1)))

    loss = loss / len(a)
    return loss
