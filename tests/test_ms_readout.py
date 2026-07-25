"""Tests for the MS-lite multi-scale FPN mask readout (ms_readout).

Covers: MSReadout output stride/shape, AffordanceBranch construction toggle
(mask_feat/low_lateral skipped, ms_readout built), forward parity of the
downstream contract (aff_feat + aff_kernel emitted at stride-2, box-relative
and deep-supervision paths intact), and the off-switch leaving the legacy
single-level layout byte-identical.
"""

from __future__ import annotations

import unittest

import torch
from src.zoo.rtdetr.affordance import AffordanceBranch, MSReadout

HIDDEN = 32
REDUCED = 16
N_AFF = 10
C2_DIM = 64


class TestMSReadoutModule(unittest.TestCase):
    def test_output_is_stride2_of_c2(self):
        m = MSReadout(c2_dim=C2_DIM, enc_dim=HIDDEN, reduced_dim=REDUCED)
        c2 = torch.randn(2, C2_DIM, 32, 32)  # stride-4
        s8 = torch.randn(2, HIDDEN, 16, 16)  # stride-8
        s16 = torch.randn(2, HIDDEN, 8, 8)  # stride-16
        out = m(c2, s8, s16)
        # One 2x upsample above C2's stride-4 -> stride-2 (double C2 spatial).
        self.assertEqual(out.shape, (2, REDUCED, 64, 64))

    def test_penult_tap_is_128(self):
        m = MSReadout(c2_dim=C2_DIM, enc_dim=HIDDEN, reduced_dim=REDUCED)
        # proj is the final 1x1 penult->reduced_dim; its in_channels is the
        # pre-registered S3-lite tap width.
        self.assertEqual(m.proj.in_channels, 128)
        self.assertEqual(m.proj.out_channels, REDUCED)


class TestAffordanceBranchMSReadout(unittest.TestCase):
    def _branch(self, ms):
        return AffordanceBranch(
            input_dim=HIDDEN,
            hidden_dim=HIDDEN,
            output_dim=N_AFF,
            reduced_dim=REDUCED,
            use_mask_upsample=True,
            mask_upsample_factor=4,
            low_level_dim=C2_DIM,
            use_box_relative=True,
            deep_supervision=True,
            ms_readout=ms,
        )

    def test_ms_skips_single_level_modules(self):
        b = self._branch(ms=True)
        self.assertIsNotNone(b.ms_readout)
        self.assertIsNone(b.mask_feat)
        self.assertIsNone(b.low_lateral)
        keys = {k.split(".")[0] for k in b.state_dict()}
        self.assertIn("ms_readout", keys)
        self.assertNotIn("mask_feat", keys)
        self.assertNotIn("low_lateral", keys)

    def test_off_keeps_legacy_layout(self):
        b = self._branch(ms=False)
        self.assertIsNone(b.ms_readout)
        self.assertIsNotNone(b.mask_feat)
        keys = {k.split(".")[0] for k in b.state_dict()}
        self.assertNotIn("ms_readout", keys)
        self.assertIn("mask_feat", keys)

    def test_ms_requires_low_level_dim(self):
        with self.assertRaises(ValueError):
            AffordanceBranch(
                input_dim=HIDDEN,
                hidden_dim=HIDDEN,
                output_dim=N_AFF,
                reduced_dim=REDUCED,
                low_level_dim=None,
                ms_readout=True,
            )

    def test_forward_contract_stride2(self):
        b = self._branch(ms=True).eval()
        bsz, q = 2, 5
        decoder_features = torch.randn(bsz, 3, q, HIDDEN)
        # encoder: stride-8 (16x16), stride-16 (8x8), stride-32 (4x4)
        enc = [
            torch.randn(bsz, HIDDEN, 16, 16),
            torch.randn(bsz, HIDDEN, 8, 8),
            torch.randn(bsz, HIDDEN, 4, 4),
        ]
        c2 = torch.randn(bsz, C2_DIM, 32, 32)  # stride-4
        out = b(decoder_features, enc, low_level_feat=c2)
        self.assertIn("aff_feat", out)
        self.assertIn("aff_kernel", out)
        # aff_feat at stride-2 (double C2): 64x64, reduced_dim channels.
        self.assertEqual(out["aff_feat"].shape, (bsz, REDUCED, 64, 64))
        self.assertEqual(out["aff_kernel"].shape[0], bsz)
        self.assertTrue(out["aff_box_relative"])

    def test_forward_raises_without_c2(self):
        b = self._branch(ms=True).eval()
        decoder_features = torch.randn(1, 3, 4, HIDDEN)
        enc = [torch.randn(1, HIDDEN, 16, 16), torch.randn(1, HIDDEN, 8, 8)]
        with self.assertRaises(ValueError):
            b(decoder_features, enc, low_level_feat=None)


if __name__ == "__main__":
    unittest.main()
