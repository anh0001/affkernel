"""Tests for the dense auxiliary affordance head (F recipe: dense_aux).

Covers: head construction toggle, forward emission in train AND eval modes
(eval needs the logits for the slot-gated dense fallback), the dense semantic
target builder (merge, conflict-ignore, out-of-range, empty), and the
criterion loss (guard on absent key, finite value, weight_dict wiring).
"""

from __future__ import annotations

import unittest

import torch
from src.zoo.rtdetr.affordance import AffordanceBranch
from src.zoo.rtdetr.rtdetr_criterion import (
    _AFF_IGNORE_INDEX,
    SetCriterion,
    _dense_semantic_target,
)

HIDDEN = 32
REDUCED = 8
N_AFF = 10


def _tiny_branch(dense_aux):
    return AffordanceBranch(
        input_dim=HIDDEN,
        hidden_dim=HIDDEN,
        output_dim=N_AFF,
        reduced_dim=REDUCED,
        use_mask_upsample=True,
        mask_upsample_factor=2,
        dense_aux=dense_aux,
    )


def _tiny_criterion(use_affordance=True):
    return SetCriterion(
        matcher=None,
        weight_dict={"loss_affordance": 3, "loss_affordance_dense": 1.0},
        losses=["affordances", "affordances_dense"],
        num_classes=10,
        use_affordance=use_affordance,
    )


class TestDenseHeadConstruction(unittest.TestCase):
    def test_head_present_only_when_enabled(self):
        on = _tiny_branch(dense_aux=True)
        off = _tiny_branch(dense_aux=False)
        self.assertIsNotNone(on.dense_head)
        self.assertIsNone(off.dense_head)
        on_keys = {k for k in on.state_dict() if k.startswith("dense_head")}
        off_keys = {k for k in off.state_dict() if k.startswith("dense_head")}
        self.assertEqual(len(on_keys), 2)  # 1x1 conv weight + bias
        self.assertEqual(off_keys, set())

    def test_off_is_key_identical_to_legacy(self):
        # dense_aux=False must not change the checkpoint layout at all.
        off = _tiny_branch(dense_aux=False)
        legacy = AffordanceBranch(
            input_dim=HIDDEN,
            hidden_dim=HIDDEN,
            output_dim=N_AFF,
            reduced_dim=REDUCED,
            use_mask_upsample=True,
            mask_upsample_factor=2,
        )
        self.assertEqual(set(off.state_dict()), set(legacy.state_dict()))


class TestDenseForward(unittest.TestCase):
    def _forward(self, branch):
        b, q, hf = 2, 5, 8
        decoder_features = torch.randn(b, 3, q, HIDDEN)
        encoder_feats = [torch.randn(b, HIDDEN, hf, hf)]
        return branch(decoder_features, encoder_feats)

    def test_emits_dense_logits_in_train_and_eval(self):
        branch = _tiny_branch(dense_aux=True)
        for mode in (branch.train, branch.eval):
            mode()
            out = self._forward(branch)
            self.assertIn("aff_dense_logits", out)
            # mask_upsample_factor=2 -> one 2x upsample of the 8x8 encoder map.
            self.assertEqual(out["aff_dense_logits"].shape, (2, N_AFF, 16, 16))

    def test_no_dense_key_when_disabled(self):
        branch = _tiny_branch(dense_aux=False)
        out = self._forward(branch)
        self.assertNotIn("aff_dense_logits", out)


class TestDenseSemanticTarget(unittest.TestCase):
    def test_merge_and_conflict_ignore(self):
        h = w = 4
        a = torch.zeros(h, w, dtype=torch.long)
        b = torch.zeros(h, w, dtype=torch.long)
        a[0, :2] = 5  # grasp strip
        b[1, :2] = 2  # cut strip (disjoint)
        a[2, 2] = 5
        b[2, 2] = 2  # conflicting pixel
        tgt = _dense_semantic_target(torch.stack([a, b]), N_AFF)
        self.assertEqual(tgt[0, 0].item(), 5)
        self.assertEqual(tgt[1, 0].item(), 2)
        self.assertEqual(tgt[2, 2].item(), _AFF_IGNORE_INDEX)
        self.assertEqual(tgt[3, 3].item(), 0)  # untouched background

    def test_out_of_range_labels_ignored(self):
        m = torch.zeros(1, 3, 3, dtype=torch.long)
        m[0, 0, 0] = 255  # transform padding
        m[0, 1, 1] = N_AFF  # out of range
        tgt = _dense_semantic_target(m, N_AFF)
        self.assertEqual(tgt[0, 0].item(), _AFF_IGNORE_INDEX)
        self.assertEqual(tgt[1, 1].item(), _AFF_IGNORE_INDEX)

    def test_empty_instances_all_background(self):
        m = torch.zeros(0, 5, 6, dtype=torch.long)
        tgt = _dense_semantic_target(m, N_AFF)
        self.assertEqual(tgt.shape, (5, 6))
        self.assertTrue((tgt == 0).all())


class TestDenseLoss(unittest.TestCase):
    def test_guard_when_key_absent(self):
        crit = _tiny_criterion()
        # Aux/dn loop entries never carry aff_dense_logits -> must no-op.
        out = crit.loss_affordances_dense({"pred_logits": torch.randn(1, 5, 10)}, [], [], 1)
        self.assertEqual(out, {})

    def test_finite_loss_and_upsampling(self):
        torch.manual_seed(0)
        crit = _tiny_criterion()
        b, hf, th = 2, 8, 32  # logits at 8x8, GT at 32x32 -> upsample path
        outputs = {"aff_dense_logits": torch.randn(b, N_AFF, hf, hf, requires_grad=True)}
        targets = []
        for _ in range(b):
            inst = torch.zeros(2, th, th, dtype=torch.long)
            inst[0, :8, :8] = 1
            inst[1, 20:, 20:] = 5
            targets.append({"masks": inst})
        out = crit.loss_affordances_dense(outputs, targets, [], 1)
        self.assertIn("loss_affordance_dense", out)
        loss = out["loss_affordance_dense"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()  # gradient flows to the dense logits
        self.assertIsNotNone(outputs["aff_dense_logits"].grad)

    def test_losses_pruned_without_affordance(self):
        crit = _tiny_criterion(use_affordance=False)
        self.assertNotIn("affordances", crit.losses)
        self.assertNotIn("affordances_dense", crit.losses)
        self.assertNotIn("loss_affordance_dense", crit.weight_dict)


if __name__ == "__main__":
    unittest.main()
