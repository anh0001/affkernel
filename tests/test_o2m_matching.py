"""Tests for the one-to-many auxiliary matching (H-DETR style) A1 lever.

Covers the two pieces added without needing the full model / GPU:
  1. ``RTDETRTransformer._expand_attn_mask_for_o2m`` — the appended o2m block is
     fully isolated (cannot see / be seen by primary or DN queries) yet attends
     within itself, for both the DN-on and DN-off cases.
  2. ``SetCriterion``'s one-to-many loss block — GT is repeated K times, matched
     against the o2m group, and emits weighted ``*_o2m_i`` detection losses;
     absent an o2m group the loss dict is unchanged (byte-identical toggle-off).
"""

import unittest

import torch
from src.zoo.rtdetr.matcher import HungarianMatcher
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer


class _DecoderStub:
    """Exposes only what ``_expand_attn_mask_for_o2m`` reads (self.num_queries)."""

    num_queries = 300


class TestExpandAttnMaskForO2M(unittest.TestCase):
    def test_dn_off_isolates_o2m_and_keeps_primary_full_attention(self):
        stub = _DecoderStub()
        nq, no2m = 300, 100
        m = RTDETRTransformer._expand_attn_mask_for_o2m(stub, None, no2m, "cpu")
        self.assertEqual(tuple(m.shape), (nq + no2m, nq + no2m))
        self.assertEqual(m.dtype, torch.bool)
        # Primary <-> primary: full attention (no blocking).
        self.assertFalse(m[:nq, :nq].any())
        # Primary cannot see o2m; o2m cannot see primary.
        self.assertTrue(m[:nq, nq:].all())
        self.assertTrue(m[nq:, :nq].all())
        # o2m attends within itself.
        self.assertFalse(m[nq:, nq:].any())

    def test_dn_on_preserves_dn_block_and_appends_isolated_o2m(self):
        stub = _DecoderStub()
        nq, no2m, ndn = 300, 100, 40
        base = ndn + nq
        # A non-trivial DN mask (match cannot see reconstruction, as in denoising.py).
        dn_mask = torch.zeros(base, base, dtype=torch.bool)
        dn_mask[ndn:, :ndn] = True
        m = RTDETRTransformer._expand_attn_mask_for_o2m(stub, dn_mask, no2m, "cpu")
        self.assertEqual(tuple(m.shape), (base + no2m, base + no2m))
        # Original DN block copied verbatim into the top-left.
        self.assertTrue(torch.equal(m[:base, :base], dn_mask))
        # Nobody in [DN|o2o] can see o2m; o2m can see nobody outside itself.
        self.assertTrue(m[:base, base:].all())
        self.assertTrue(m[base:, :base].all())
        self.assertFalse(m[base:, base:].any())


def _detection_criterion(num_classes=10):
    matcher = HungarianMatcher(
        {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
    )
    return SetCriterion(
        matcher,
        weight_dict={"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2},
        losses=["vfl", "boxes"],
        num_classes=num_classes,
        use_affordance=False,
        o2m_loss_weight=1.0,
    )


def _synthetic(bs=2, q=10, qm=20, layers=2, n_gt=2, num_classes=10, with_o2m=True):
    torch.manual_seed(0)

    def boxes(k):
        cxcy = torch.rand(k, 2) * 0.6 + 0.2
        wh = torch.rand(k, 2) * 0.2 + 0.1
        return torch.cat([cxcy, wh], dim=1)

    outputs = {
        "pred_logits": torch.randn(bs, q, num_classes),
        "pred_boxes": boxes(bs * q).view(bs, q, 4),
    }
    if with_o2m:
        outputs["o2m_aux_outputs"] = [
            {
                "pred_logits": torch.randn(bs, qm, num_classes),
                "pred_boxes": boxes(bs * qm).view(bs, qm, 4),
            }
            for _ in range(layers)
        ]
        outputs["o2m_meta"] = {"o2m_group_repeat": 4}
    targets = [
        {"labels": torch.randint(0, num_classes, (n_gt,)), "boxes": boxes(n_gt)}
        for _ in range(bs)
    ]
    return outputs, targets


class TestO2MCriterionBlock(unittest.TestCase):
    def test_emits_weighted_o2m_losses_for_every_layer(self):
        crit = _detection_criterion()
        outputs, targets = _synthetic(layers=2, with_o2m=True)
        losses = crit(outputs, targets)
        for i in range(2):
            for base in ("loss_vfl", "loss_bbox", "loss_giou"):
                key = f"{base}_o2m_{i}"
                self.assertIn(key, losses)
                self.assertTrue(torch.isfinite(losses[key]))
        # No affordance/mask leakage into the detection-only o2m group.
        self.assertFalse(any(k.startswith("loss_affordance_o2m") for k in losses))

    def test_toggle_off_has_no_o2m_keys(self):
        crit = _detection_criterion()
        outputs, targets = _synthetic(with_o2m=False)
        losses = crit(outputs, targets)
        self.assertFalse(any("_o2m_" in k for k in losses))

    def test_repeat_scales_positive_matches(self):
        # With K repeats and qm >> K*n_gt, the o2m bbox loss should aggregate
        # over K*n_gt positives per image (normaliser num_boxes_o2o * K keeps it
        # finite and > 0).
        crit = _detection_criterion()
        outputs, targets = _synthetic(qm=20, n_gt=2, with_o2m=True)
        losses = crit(outputs, targets)
        self.assertGreater(losses["loss_bbox_o2m_0"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
