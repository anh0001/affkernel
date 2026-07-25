"""Tests for UMDEvaluator and the config-driven affordance class count.

Covers:
  * UMDEvaluator scores a perfect synthetic prediction as F_beta^w = 1.0 per
    class for BOTH beta^2 conventions (1 primary, 0.3 aux).
  * A missed class (GT present, no prediction) contributes ~0 and is NOT
    skipped (matches IIT/AffordanceNet semantics), while a class with no GT on
    any image is skipped entirely.
  * The parameterized affordance class count preserves the IIT default (10):
    matcher / criterion / postprocessor built WITHOUT the new kwarg behave as
    before, and honour an explicit non-IIT count (8) when given.

Synthetic ground truth is written as real ``*_label.mat`` files into a temp
dir, so the actual loadmat GT path is exercised without the dataset on disk.
"""

import os
import tempfile
import unittest

import numpy as np
import torch
from scipy.io import savemat
from src.data.umd.umd_dataset import UMDDetection
from src.data.umd.umd_eval import UMDEvaluator
from src.zoo.rtdetr.matcher import HungarianMatcher
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor

_KNIFE_LABEL = UMDDetection.object_classes.index("knife")
_GRASP = UMDDetection.affordance_classes.index("grasp")  # 1
_CUT = UMDDetection.affordance_classes.index("cut")      # 2


class _StubUMD:
    """Minimal UMD-dataset stand-in exposing only what UMDEvaluator reads."""

    object_classes = UMDDetection.object_classes
    affordance_classes = UMDDetection.affordance_classes

    def __init__(self, tmp_dir, ids):
        self.tmp_dir = tmp_dir
        self.ids = ids
        self.affordance_dict = {n: i for i, n in enumerate(self.affordance_classes)}

    @staticmethod
    def _tool_category(tool):
        return tool.rsplit("_", 1)[0]

    def _frame_path(self, index, kind):
        tool, frame = self.ids[index]
        return os.path.join(self.tmp_dir, f"{tool}_{frame}_{kind}")


def _write_label(stub, index, label):
    savemat(stub._frame_path(index, "label.mat"), {"gt_label": label})


def _make_gt(h=16, w=16):
    """Two 16x16 label maps: grasp+cut on image 0, grasp only on image 1."""
    lab0 = np.zeros((h, w), dtype=np.uint8)
    lab0[:, : w // 2] = _GRASP
    lab0[:, w // 2:] = _CUT
    lab1 = np.zeros((h, w), dtype=np.uint8)
    lab1[: h // 2, : w // 2] = _GRASP
    return [lab0, lab1]


def _prediction(label_map):
    """A single high-score query whose per-pixel affordance == label_map."""
    return {
        "scores": torch.tensor([0.99]),
        "labels": torch.tensor([_KNIFE_LABEL]),
        "boxes": torch.tensor([[0.0, 0.0, float(label_map.shape[1]), float(label_map.shape[0])]]),
        "affordances": torch.as_tensor(label_map[None].astype(np.int64)),
    }


class TestUMDEvaluatorPerfect(unittest.TestCase):
    def test_perfect_prediction_scores_one_both_betas(self):
        labels = _make_gt()
        with tempfile.TemporaryDirectory() as tmp:
            ids = [("knife_01", "00000000"), ("knife_01", "00000001")]
            stub = _StubUMD(tmp, ids)
            for i, lab in enumerate(labels):
                _write_label(stub, i, lab)

            ev = UMDEvaluator(stub, use_affordance=True)
            ev.update({0: _prediction(labels[0]), 1: _prediction(labels[1])})
            ev.summarize()

            for bucket, _beta in UMDEvaluator.FBW_BETAS:
                fbw = list(ev.stats[bucket]["Fbw"])
                # grasp (both images) and cut (image 0) are present in GT.
                self.assertEqual(len(fbw), 2)
                for q in fbw:
                    self.assertAlmostEqual(q, 1.0, places=6)

    def test_iou_types_and_valid_affordances_are_umd(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubUMD(tmp, [("knife_01", "00000000")])
            ev = UMDEvaluator(stub, use_affordance=True)
            self.assertEqual(ev.iou_types, ["bbox", "affordance"])
            self.assertEqual(ev.valid_affordances, {})


class TestUMDEvaluatorMiss(unittest.TestCase):
    def test_missed_class_contributes_zero_not_skipped(self):
        labels = _make_gt()
        with tempfile.TemporaryDirectory() as tmp:
            ids = [("knife_01", "00000000"), ("knife_01", "00000001")]
            stub = _StubUMD(tmp, ids)
            for i, lab in enumerate(labels):
                _write_label(stub, i, lab)

            ev = UMDEvaluator(stub, use_affordance=True)
            # Model predicts an all-background affordance map -> no class mask.
            blank = np.zeros_like(labels[0])
            ev.update({0: _prediction(blank), 1: _prediction(blank)})
            ev.summarize()

            fbw = list(ev.stats["affordance_fbw_b1"]["Fbw"])
            # grasp + cut are present in GT -> both scored (not skipped),
            # each ~0 because nothing was predicted.
            self.assertEqual(len(fbw), 2)
            for q in fbw:
                self.assertLess(q, 0.05)

    def test_class_with_no_gt_is_skipped(self):
        # GT uses only grasp+cut; scoop/contain/etc. never appear -> not scored.
        labels = _make_gt()
        with tempfile.TemporaryDirectory() as tmp:
            ids = [("knife_01", "00000000"), ("knife_01", "00000001")]
            stub = _StubUMD(tmp, ids)
            for i, lab in enumerate(labels):
                _write_label(stub, i, lab)

            ev = UMDEvaluator(stub, use_affordance=True)
            ev.update({0: _prediction(labels[0]), 1: _prediction(labels[1])})
            ev.summarize()

            # Exactly the two GT-present classes are scored.
            self.assertEqual(len(list(ev.stats["affordance_fbw_b03"]["Fbw"])), 2)


class TestAffordanceClassCountParameterization(unittest.TestCase):
    """The new kwarg defaults to the IIT count (10) and honours overrides (8)."""

    def _matcher(self, **kw):
        return HungarianMatcher(
            weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2, "cost_affordance": 4},
            **kw,
        )

    def test_matcher_default_is_iit_ten(self):
        m = self._matcher()
        self.assertEqual(m.num_affordance_classes, 10)
        # Legacy IIT object->affordance table preserved at the IIT count.
        self.assertEqual(m.valid_affordances[10], [1, 9])

    def test_matcher_override_eight_has_empty_table(self):
        m = self._matcher(num_affordance_classes=8)
        self.assertEqual(m.num_affordance_classes, 8)
        self.assertEqual(m.valid_affordances, {})

    def test_postprocessor_default_and_override(self):
        self.assertEqual(RTDETRPostProcessor(use_affordance=True).num_affordance_classes, 10)
        self.assertEqual(
            RTDETRPostProcessor(use_affordance=True, num_affordance_classes=8).num_affordance_classes,
            8,
        )

    def test_criterion_default_and_override(self):
        wd = {"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2}
        default = SetCriterion(matcher=self._matcher(), weight_dict=wd, losses=["boxes"])
        self.assertEqual(default.num_affordance_classes, 10)
        # Class-level attribute still readable (used by other tests / configs).
        self.assertEqual(SetCriterion.num_affordance_classes, 10)

        override = SetCriterion(
            matcher=self._matcher(num_affordance_classes=8),
            weight_dict=wd,
            losses=["boxes"],
            num_affordance_classes=8,
        )
        self.assertEqual(override.num_affordance_classes, 8)


if __name__ == "__main__":
    unittest.main()
