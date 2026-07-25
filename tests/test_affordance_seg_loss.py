"""Tests for the configurable affordance segmentation loss knobs.

Locks in that the new ``focal_gamma`` / ``boundary_w`` / ``dice_w`` parameters
of ``_affordance_seg_loss`` (a) default to the v2/v3 constants byte-identically
and (b) move the loss in the documented way, so the loss-hyperparameter sweep
configs override exactly what they claim to.
"""

import unittest

import torch
from src.zoo.rtdetr.rtdetr_criterion import (
    _AFF_BOUNDARY_W,
    _AFF_DICE_W,
    _AFF_FOCAL_GAMMA,
    SetCriterion,
    _affordance_boundary_weight,
    _affordance_seg_loss,
)


class _NumClassesStub:
    """Minimal stand-in exposing only what ``_validate_class_boundary_w`` reads."""

    num_affordance_classes = SetCriterion.num_affordance_classes


def _fixture():
    """Small deterministic logits + a target with a real class boundary."""
    torch.manual_seed(0)
    n, c, h, w = 2, 4, 8, 8
    src = torch.randn(n, c, h, w)
    target = torch.zeros(n, h, w, dtype=torch.long)
    target[:, :, w // 2:] = 1  # left half class 0, right half class 1 -> a boundary
    return src, target


class TestAffordanceSegLossKnobs(unittest.TestCase):
    def test_default_matches_explicit_constants(self):
        src, target = _fixture()
        default = _affordance_seg_loss(src, target)
        explicit = _affordance_seg_loss(
            src, target,
            focal_gamma=_AFF_FOCAL_GAMMA,
            boundary_w=_AFF_BOUNDARY_W,
            dice_w=_AFF_DICE_W,
        )
        self.assertTrue(torch.equal(default, explicit))

    def test_dice_w_scales_dice_term_linearly(self):
        src, target = _fixture()
        l0 = _affordance_seg_loss(src, target, dice_w=0.0)  # focal_ce only
        l1 = _affordance_seg_loss(src, target, dice_w=1.0)
        l2 = _affordance_seg_loss(src, target, dice_w=2.0)
        dice = (l1 - l0).item()
        self.assertGreater(dice, 0.0)  # dice term is positive here
        # loss(dice_w=k) = focal_ce + k * dice  ->  (l2-l0) == 2*(l1-l0)
        self.assertAlmostEqual((l2 - l0).item(), 2.0 * dice, places=5)

    def test_boundary_w_one_disables_edge_weighting(self):
        src, target = _fixture()
        valid = torch.ones_like(target, dtype=torch.bool)
        w_unit = _affordance_boundary_weight(target, valid, boundary_w=1.0)
        self.assertTrue(torch.equal(w_unit, torch.ones_like(w_unit)))
        # With a boundary present, weighting at 4.0 differs from 1.0.
        l_unit = _affordance_seg_loss(src, target, boundary_w=1.0)
        l_def = _affordance_seg_loss(src, target, boundary_w=4.0)
        self.assertFalse(torch.allclose(l_unit, l_def))

    def test_focal_gamma_zero_reduces_to_weighted_ce(self):
        src, target = _fixture()
        # gamma=0 -> (1-pt)^0 == 1 -> focal == ce; just assert it is finite and
        # differs from the gamma=2 default (hard-pixel focus changes the value).
        l_g0 = _affordance_seg_loss(src, target, focal_gamma=0.0)
        l_g2 = _affordance_seg_loss(src, target, focal_gamma=2.0)
        self.assertTrue(torch.isfinite(l_g0))
        self.assertFalse(torch.allclose(l_g0, l_g2))

    def test_empty_valid_returns_zero(self):
        src, target = _fixture()
        target.fill_(255)  # all ignored
        loss = _affordance_seg_loss(src, target)
        self.assertEqual(loss.item(), 0.0)


class TestPerClassBoundaryWeight(unittest.TestCase):
    """B1: {class_index: weight} override up-weights one class's edges only."""

    def test_none_is_byte_identical_to_global_only(self):
        src, target = _fixture()
        valid = torch.ones_like(target, dtype=torch.bool)
        base = _affordance_boundary_weight(target, valid, boundary_w=4.0)
        with_none = _affordance_boundary_weight(
            target, valid, boundary_w=4.0, class_boundary_w=None
        )
        with_empty = _affordance_boundary_weight(
            target, valid, boundary_w=4.0, class_boundary_w={}
        )
        self.assertTrue(torch.equal(base, with_none))
        self.assertTrue(torch.equal(base, with_empty))
        # And end-to-end the loss is unchanged when the override is absent.
        self.assertTrue(torch.equal(
            _affordance_seg_loss(src, target),
            _affordance_seg_loss(src, target, class_boundary_w=None),
        ))

    def test_override_raises_only_target_class_edges(self):
        _, target = _fixture()  # class 0 (left) | class 1 (right), shared edge
        valid = torch.ones_like(target, dtype=torch.bool)
        w = _affordance_boundary_weight(
            target, valid, boundary_w=4.0, class_boundary_w={1: 8.0}
        )
        is_boundary = w > 1.0
        # Class-1 boundary pixels lifted to 8.0; class-0 boundary pixels stay 4.0.
        cls1_edge = is_boundary & (target == 1)
        cls0_edge = is_boundary & (target == 0)
        self.assertTrue(cls1_edge.any() and cls0_edge.any())
        self.assertTrue(torch.all(w[cls1_edge] == 8.0))
        self.assertTrue(torch.all(w[cls0_edge] == 4.0))
        # Non-boundary interior stays at 1.0 regardless.
        self.assertTrue(torch.all(w[~is_boundary] == 1.0))

    def test_override_changes_loss_value(self):
        src, target = _fixture()
        base = _affordance_seg_loss(src, target, boundary_w=4.0)
        boosted = _affordance_seg_loss(
            src, target, boundary_w=4.0, class_boundary_w={1: 8.0}
        )
        self.assertFalse(torch.allclose(base, boosted))

    def test_validate_coerces_str_keys_and_rejects_out_of_range(self):
        stub = _NumClassesStub()
        self.assertIsNone(SetCriterion._validate_class_boundary_w(stub, None))
        self.assertIsNone(SetCriterion._validate_class_boundary_w(stub, {}))
        # YAML may hand back str keys; they coerce to int, values to float.
        out = SetCriterion._validate_class_boundary_w(stub, {"5": 8})
        self.assertEqual(out, {5: 8.0})
        self.assertIsInstance(next(iter(out)), int)
        with self.assertRaises(ValueError):
            SetCriterion._validate_class_boundary_w(stub, {99: 8.0})
        with self.assertRaises(ValueError):
            SetCriterion._validate_class_boundary_w(stub, [5, 8])


if __name__ == "__main__":
    unittest.main()
