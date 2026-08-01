"""Tests for inference-time Conv+BN folding (src/nn/fuse.py)."""

import copy
import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nn.backbone.common import ConvNormLayer as BackboneConvNorm  # noqa: E402
from src.nn.backbone.common import FrozenBatchNorm2d  # noqa: E402
from src.nn.fuse import fuse_conv_norm  # noqa: E402
from src.zoo.rtdetr.hybrid_encoder import ConvNormLayer as EncoderConvNorm  # noqa: E402


def _randomize_bn(bn: nn.Module) -> None:
    """Give a norm layer non-trivial affine parameters and statistics."""
    with torch.no_grad():
        bn.weight.copy_(torch.rand_like(bn.weight) * 1.5 + 0.25)
        bn.bias.copy_(torch.randn_like(bn.bias))
        bn.running_mean.copy_(torch.randn_like(bn.running_mean))
        bn.running_var.copy_(torch.rand_like(bn.running_var) * 2.0 + 0.1)


class TestFuseConvNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _assert_fused_matches(self, layer: nn.Module, in_ch: int):
        x = torch.randn(2, in_ch, 16, 16)
        layer = layer.eval()
        with torch.no_grad():
            ref = layer(x)
        fused = copy.deepcopy(layer)
        n = fuse_conv_norm(fused)
        self.assertEqual(n, 1)
        self.assertIsInstance(fused.norm, nn.Identity)
        self.assertIsNotNone(fused.conv.bias)
        with torch.no_grad():
            out = fused(x)
        self.assertTrue(
            torch.allclose(ref, out, atol=1e-5),
            f"max diff {(ref - out).abs().max().item():.2e}",
        )

    def test_backbone_convnorm_batchnorm_matches(self):
        layer = BackboneConvNorm(3, 8, 3, 1, act="relu")
        _randomize_bn(layer.norm)
        self._assert_fused_matches(layer, 3)

    def test_backbone_convnorm_frozen_bn_matches(self):
        layer = BackboneConvNorm(4, 6, 3, 2, act="relu")
        frozen = FrozenBatchNorm2d(6)
        _randomize_bn(frozen)
        layer.norm = frozen
        self._assert_fused_matches(layer, 4)

    def test_encoder_convnorm_matches(self):
        layer = EncoderConvNorm(5, 7, 1, 1, act="silu")
        _randomize_bn(layer.norm)
        self._assert_fused_matches(layer, 5)

    def test_conv_with_existing_bias_matches(self):
        layer = BackboneConvNorm(3, 8, 3, 1, bias=True, act=None)
        _randomize_bn(layer.norm)
        with torch.no_grad():
            layer.conv.bias.copy_(torch.randn_like(layer.conv.bias))
        self._assert_fused_matches(layer, 3)

    def test_already_fused_layer_is_skipped(self):
        layer = BackboneConvNorm(3, 8, 3, 1, act="relu").eval()
        self.assertEqual(fuse_conv_norm(layer), 1)
        self.assertEqual(fuse_conv_norm(layer), 0)  # norm is Identity now

    def test_non_convnorm_modules_untouched(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8)).eval()
        self.assertEqual(fuse_conv_norm(model), 0)
        self.assertIsInstance(model[1], nn.BatchNorm2d)


if __name__ == "__main__":
    unittest.main()
