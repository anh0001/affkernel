"""Tests for the Swin-T feature-pyramid backbone.

Locks in the PResNet output contract (out_channels / out_strides / NCHW
list-returning forward) and the channel/stride map (C2/C3/C4/C5 =
96/192/384/768 at strides 4/8/16/32). Uses ``pretrained=False`` to avoid the
ImageNet weight download.
"""

import unittest

import torch
from src.nn.backbone.swin import SwinTransformerBackbone


class TestSwinTransformerBackbone(unittest.TestCase):
    def test_full_pyramid_contract(self):
        bb = SwinTransformerBackbone(return_idx=[0, 1, 2, 3], pretrained=False)
        self.assertEqual(bb.out_channels, [96, 192, 384, 768])
        self.assertEqual(bb.out_strides, [4, 8, 16, 32])

    def test_forward_shapes_and_layout(self):
        bb = SwinTransformerBackbone(return_idx=[0, 1, 2, 3], pretrained=False).eval()
        with torch.no_grad():
            outs = bb(torch.randn(2, 3, 256, 256))
        self.assertEqual(len(outs), 4)
        # NCHW with the documented channels and spatial strides.
        channels = [96, 192, 384, 768]
        strides = [4, 8, 16, 32]
        for i, feat in enumerate(outs):
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], channels[i])
            self.assertEqual(feat.shape[2], 256 // strides[i])
            self.assertEqual(feat.shape[3], 256 // strides[i])

    def test_return_idx_subset_drops_c2(self):
        # The detector-only pyramid (no C2 lateral) returns just C3/C4/C5.
        bb = SwinTransformerBackbone(return_idx=[1, 2, 3], pretrained=False).eval()
        self.assertEqual(bb.out_channels, [192, 384, 768])
        self.assertEqual(bb.out_strides, [8, 16, 32])
        with torch.no_grad():
            outs = bb(torch.randn(1, 3, 256, 256))
        self.assertEqual([f.shape[1] for f in outs], [192, 384, 768])

    def test_rejects_unsupported_variant(self):
        with self.assertRaises(ValueError):
            SwinTransformerBackbone(variant="small", pretrained=False)


if __name__ == "__main__":
    unittest.main()
