"""Inference-time Conv+BN folding.

Folds the batch-norm that follows a convolution into the convolution's own
weight and bias, then replaces the norm with ``nn.Identity``. With frozen (or
eval-mode) statistics this is the standard exact reparameterisation

    scale = gamma / sqrt(var + eps)
    W'    = W * scale          (per output channel)
    b'    = beta - mean * scale (+ scale * b if the conv already had a bias)

so the fused module computes the same function up to floating-point rounding.
It removes two full activation-map read/write passes per norm (~95 Conv+BN
pairs in the R50 deployment), which is pure memory traffic at inference time
and is relatively costlier on bandwidth-limited devices such as Jetson.

Folding is deliberately restricted to the repo's ``ConvNormLayer`` wrappers,
whose forward is exactly ``act(norm(conv(x)))``; duck-typing arbitrary
``.conv``/``.norm`` attribute pairs could fold modules whose forward wires
them differently.

Call only on a model in eval/deploy mode, and fold BEFORE any cast to fp16 so
the arithmetic happens in fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone.common import ConvNormLayer as _BackboneConvNorm
from .backbone.common import FrozenBatchNorm2d

__all__ = ["fuse_conv_norm"]


def _norm_stats(norm: nn.Module):
    """Return (weight, bias, mean, var, eps) for a foldable norm, else None."""
    if isinstance(norm, (nn.BatchNorm2d, FrozenBatchNorm2d)):
        weight = norm.weight if norm.weight is not None else torch.ones_like(norm.running_mean)
        bias = norm.bias if norm.bias is not None else torch.zeros_like(norm.running_mean)
        return weight, bias, norm.running_mean, norm.running_var, norm.eps
    return None


def fuse_conv_norm(model: nn.Module) -> int:
    """Fold BN into conv for every ConvNormLayer in ``model``; return the count.

    Handles both the backbone wrapper (``src.nn.backbone.common.ConvNormLayer``,
    used with ``FrozenBatchNorm2d`` when ``freeze_norm`` is set) and the
    encoder's own ``ConvNormLayer`` (plain ``BatchNorm2d``). Modules already
    fused, or whose norm is not a (frozen) batch norm, are left untouched.
    """
    # The encoder defines an identically-shaped ConvNormLayer of its own;
    # import lazily to keep src.nn free of a hard zoo dependency.
    from ..zoo.rtdetr.hybrid_encoder import ConvNormLayer as _EncoderConvNorm

    n_fused = 0
    for module in model.modules():
        if not isinstance(module, (_BackboneConvNorm, _EncoderConvNorm)):
            continue
        stats = _norm_stats(module.norm)
        if stats is None:
            continue
        weight, bias, mean, var, eps = stats
        conv = module.conv
        scale = weight.float() * (var.float() + eps).rsqrt()
        fused_w = conv.weight.data.float() * scale.view(-1, 1, 1, 1)
        fused_b = bias.float() - mean.float() * scale
        if conv.bias is not None:
            fused_b = fused_b + conv.bias.data.float() * scale
            conv.bias.data = fused_b.to(conv.weight.dtype)
        else:
            conv.bias = nn.Parameter(fused_b.to(conv.weight.dtype))
        conv.weight.data = fused_w.to(conv.weight.dtype)
        module.norm = nn.Identity()
        n_fused += 1
    return n_fused
