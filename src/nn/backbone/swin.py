"""Swin Transformer backbone (torchvision) for RT-DETR / AffKernel.

Wraps ``torchvision.models.swin_t`` as a feature-pyramid backbone that matches
the :class:`PResNet` contract used by the rest of the pipeline:

* exposes ``out_channels`` / ``out_strides`` for the requested stages, and
* ``forward(x)`` returns a list of NCHW feature maps (finest-to-coarsest).

This varies the backbone architecture class rather than its depth: in our
experiments, going from ResNet-50vd to ResNet-101vd did not close the residual
gap to transformer-backbone results, so the backbone family is the variable
this module exists to test. The detector wiring is
backbone-agnostic — ``RTDETR.forward`` routes the finest level (C2, stride-4)
to the affordance branch and feeds the last three levels to the encoder — so
only the channel dims differ vs ResNet (C2/C3/C4/C5 = 96/192/384/768 for
Swin-T, vs 256/512/1024/2048 for ResNet-50d).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core import register

__all__ = ["SwinTransformerBackbone"]


# torchvision ``swin_t().features`` is a length-8 Sequential emitting NHWC
# tensors: [PatchEmbed, stage1, PatchMerging, stage2, PatchMerging, stage3,
# PatchMerging, stage4]. The stage-k output is ready after module _CAPTURE[k].
_CAPTURE = [1, 3, 5, 7]              # module idx after which stage k is complete
_SWIN_T_CHANNELS = [96, 192, 384, 768]
_SWIN_T_STRIDES = [4, 8, 16, 32]


@register
class SwinTransformerBackbone(nn.Module):
    """Swin-T feature pyramid with the PResNet output contract.

    Args:
        variant: only ``'tiny'`` is supported (matches the SOTA backbone).
        return_idx: which stages to return, 0=C2(stride4) .. 3=C5(stride32).
        pretrained: load torchvision ImageNet-1k weights.
        freeze_at: freeze the patch embed and stages ``0..freeze_at`` (-1 = none).
    """

    def __init__(
        self,
        variant: str = "tiny",
        return_idx: tuple[int, ...] = (0, 1, 2, 3),
        pretrained: bool = True,
        freeze_at: int = -1,
    ):
        super().__init__()
        if variant != "tiny":
            raise ValueError(f"only 'tiny' is supported, got {variant!r}")

        from torchvision.models import Swin_T_Weights, swin_t

        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
        self.features = model.features  # 8-module Sequential, NHWC outputs

        self.return_idx = list(return_idx)
        self.out_channels = [_SWIN_T_CHANNELS[i] for i in self.return_idx]
        self.out_strides = [_SWIN_T_STRIDES[i] for i in self.return_idx]
        self._capture = {_CAPTURE[stage]: stage for stage in range(len(_CAPTURE))}

        if freeze_at >= 0:
            self._freeze_stages(freeze_at)

        if pretrained:
            print("Load Swin-T (torchvision IMAGENET1K_V1) backbone")

    def _freeze_stages(self, freeze_at: int) -> None:
        """Freeze the patch embed and every stage up to ``freeze_at``."""
        max_module = _CAPTURE[min(freeze_at, len(_CAPTURE) - 1)]
        for idx, module in enumerate(self.features):
            if idx <= max_module:
                for p in module.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs: list[torch.Tensor] = []
        feat = x
        for idx, module in enumerate(self.features):
            feat = module(feat)
            stage = self._capture.get(idx)
            if stage is not None and stage in self.return_idx:
                # NHWC (torchvision Swin) -> NCHW (RT-DETR encoder / affordance).
                outs.append(feat.permute(0, 3, 1, 2).contiguous())
        return outs
