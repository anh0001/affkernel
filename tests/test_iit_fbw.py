import unittest

import numpy as np
from src.data.iit.iit_eval import weighted_fbeta_measure


class TestWeightedFbetaMeasure(unittest.TestCase):
    """F_beta^w (Margolin et al., 2014) — the IIT-AFF/AffordanceNet metric."""

    def setUp(self):
        # A simple solid square as ground truth.
        self.gt = np.zeros((40, 40), dtype=np.uint8)
        self.gt[10:30, 10:30] = 1

    def test_perfect_prediction_scores_near_one(self):
        f = weighted_fbeta_measure(self.gt.astype(float), self.gt)
        self.assertIsNotNone(f)
        self.assertGreater(f, 0.95)
        self.assertLessEqual(f, 1.0 + 1e-9)

    def test_empty_prediction_with_nonempty_gt_scores_zero(self):
        f = weighted_fbeta_measure(np.zeros_like(self.gt, dtype=float), self.gt)
        self.assertIsNotNone(f)
        self.assertAlmostEqual(f, 0.0, places=6)

    def test_empty_gt_returns_none(self):
        # No GT for this class in this image -> pair must not be scored.
        f = weighted_fbeta_measure(np.zeros((40, 40)), np.zeros((40, 40), dtype=np.uint8))
        self.assertIsNone(f)

    def test_partial_overlap_between_zero_and_one(self):
        pred = np.zeros_like(self.gt, dtype=float)
        pred[10:30, 10:20] = 1.0  # half of the GT square
        f = weighted_fbeta_measure(pred, self.gt)
        self.assertIsNotNone(f)
        self.assertGreater(f, 0.0)
        self.assertLess(f, 1.0)

    def test_better_overlap_scores_higher(self):
        half = np.zeros_like(self.gt, dtype=float)
        half[10:30, 10:20] = 1.0
        most = np.zeros_like(self.gt, dtype=float)
        most[10:30, 10:28] = 1.0
        self.assertGreater(
            weighted_fbeta_measure(most, self.gt),
            weighted_fbeta_measure(half, self.gt),
        )

    def test_beta2_is_configurable(self):
        pred = np.zeros_like(self.gt, dtype=float)
        pred[10:30, 10:20] = 1.0
        f_default = weighted_fbeta_measure(pred, self.gt, beta2=0.3)
        f_recall = weighted_fbeta_measure(pred, self.gt, beta2=1.0)
        self.assertNotAlmostEqual(f_default, f_recall, places=4)


if __name__ == '__main__':
    unittest.main()
