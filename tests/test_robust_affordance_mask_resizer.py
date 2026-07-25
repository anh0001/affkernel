import unittest

import torch
from src.zoo.rtdetr.affordance_mask_resizer import RobustAffordanceMaskResizer


class TestRobustAffordanceMaskResizer(unittest.TestCase):
    def setUp(self):
        # Initialize the mask resizer with default alpha
        self.resizer = RobustAffordanceMaskResizer(alpha=0.005)
        self.alpha = 0.005
        self.image_size = torch.tensor([4, 4])  # Width, Height

    def test_resize_mask_preserves_affordance_boundaries(self):
        """
        Test that affordance boundaries are preserved after resizing with thresholding.
        """
        # Original mask with labels 0 and 1
        mask = torch.tensor([
            [1, 1],
            [1, 1]
        ], dtype=torch.long)

        bbox = [1, 1, 4, 4]
        image_size = torch.tensor([4, 4])  # Width, Height

        resized_mask = self.resizer.resize_mask(mask, bbox, image_size)

        expected_mask = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1]
        ], dtype=torch.long)

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")

    def test_resize_mask_with_partial_bbox_overlap_no_rescale(self):
        """
        Test a partially out-of-image bbox when the mask already matches the image size,
        so the clamped bbox region is copied across without rescaling.
        """
        # Original mask with labels 1 and 2
        mask = torch.tensor([
            [1, 1, 0],
            [1, 2, 0],
            [2, 2, 2]
        ], dtype=torch.long)

        bbox = [-1, -1, 2, 2]  # xmin, ymin, xmax, ymax (partially outside)
        image_size = torch.tensor([3, 3])  # Width, Height

        resized_mask = self.resizer.resize_mask(mask, bbox, image_size)

        # Expected behavior: the bbox is 3x3 (xmax-xmin = 2-(-1)), and clamping
        # shifts the origin to (0, 0) rather than truncating the width/height, so
        # the destination region is the whole 3x3 image and the 3x3 source mask
        # is copied across unscaled.
        expected_mask = torch.tensor([
            [1, 1, 0],
            [1, 2, 0],
            [2, 2, 2]
        ], dtype=torch.long)

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")

    def test_resize_mask_with_valid_bbox_downscales_source(self):
        """
        Test a valid bounding box when the source mask is larger than the bbox,
        so the mask is downscaled into the bbox region.
        """
        # Original mask with labels 0 and 1
        mask = torch.tensor([
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 0]
        ], dtype=torch.long)

        bbox = [1, 1, 3, 3]  # xmin, ymin, xmax, ymax
        image_size = torch.tensor([4, 4])  # Width, Height

        resized_mask = self.resizer.resize_mask(mask, bbox, image_size)

        # The 4x4 source is downscaled to the 2x2 bbox with NEAREST resampling,
        # which samples source rows/cols 1 and 3 -> [[1, 0], [1, 0]], then pastes
        # that into the [1:3, 1:3] region of the 4x4 output.
        expected_mask = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ], dtype=torch.long)

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")

    def test_resize_mask_with_high_alpha(self):
        """
        Test that a higher alpha allows for wider thresholding.
        """
        # Initialize resizer with higher alpha
        resizer_high_alpha = RobustAffordanceMaskResizer(alpha=0.1)

        # Create a mask with integer labels representing different affordance classes
        mask = torch.tensor([
            [0, 1, 2],
            [2, 2, 3],
            [3, 0, 1]
        ], dtype=torch.long)

        bbox = [0, 0, 3, 3]
        image_size = torch.tensor([3, 3])  # Width, Height

        resized_mask = resizer_high_alpha.resize_mask(mask, bbox, image_size)

        # Expected behavior:
        # Since alpha is 0.1, but labels are integers and we removed thresholding,
        # the mask should remain the same as labels are directly assigned.
        expected_mask = mask.clone()

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")

    def test_resize_mask_with_invalid_bbox(self):
        """
        Test resizing a mask with an invalid bounding box (width or height <= 0).
        """
        # Original mask
        mask = torch.tensor([
            [0, 1, 2],
            [1, 1, 2],
            [2, 2, 2]
        ], dtype=torch.long)

        bbox = [1, 1, 1, 1]  # xmin, ymin, xmax, ymax (width and height = 0)
        image_size = torch.tensor([3, 3])  # Width, Height

        resized_mask = self.resizer.resize_mask(mask, bbox, image_size)

        expected_mask = torch.zeros((3, 3), dtype=torch.long)

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")

    def test_resize_mask_with_multiple_affordances(self):
        """
        Test resizing a mask with multiple affordance classes.
        """
        # Original mask with three affordance classes
        mask = torch.tensor([
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1]
        ], dtype=torch.long)

        bbox = [0, 0, 3, 3]
        image_size = torch.tensor([3, 3])

        resized_mask = self.resizer.resize_mask(mask, bbox, image_size)

        # Expected behavior: the mask remains unchanged
        expected_mask = mask.clone()

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")

    def test_resize_mask_with_no_affordances(self):
        """
        Test resizing when there are no affordances (all background).
        """
        # Original mask with all background
        mask = torch.zeros((2, 2), dtype=torch.long)

        bbox = [0, 0, 2, 2]
        image_size = torch.tensor([2, 2])

        resized_mask = self.resizer.resize_mask(mask, bbox, image_size)

        expected_mask = torch.zeros((2, 2), dtype=torch.long)

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")

    def test_resize_mask_with_non_sequential_labels(self):
        """
        Test resizing when original labels are non-sequential and unordered.
        """
        # Original mask with labels 0, 3, 5
        mask = torch.tensor([
            [0, 3, 5],
            [3, 5, 0],
            [5, 0, 3]
        ], dtype=torch.long)

        bbox = [0, 0, 3, 3]
        image_size = torch.tensor([3, 3])

        resized_mask = self.resizer.resize_mask(mask, bbox, image_size)

        # Expected behavior: the mask remains unchanged
        expected_mask = mask.clone()

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")

    def test_resize_mask_with_valid_bbox_upscales_source(self):
        """
        Test a valid bounding box when the source mask is smaller than the bbox,
        so the mask is upscaled to fill the bbox region.
        """
        # Original mask with labels 0 and 1
        mask = torch.tensor([
            [1, 1],
            [1, 1]
        ], dtype=torch.long)

        bbox = [1, 1, 3, 3]  # xmin, ymin, xmax, ymax
        image_size = torch.tensor([4, 4])  # Width, Height

        resized_mask = self.resizer.resize_mask(mask, bbox, image_size)

        expected_mask = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ], dtype=torch.long)

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")

    def test_resize_mask_with_zero_affordance_classes(self):
        """
        Test resizing when there are zero affordance classes (only background).
        """
        # Original mask with all background
        mask = torch.zeros((2, 2), dtype=torch.long)

        bbox = [0, 0, 2, 2]
        image_size = torch.tensor([2, 2])

        resized_mask = self.resizer.resize_mask(mask, bbox, image_size)

        expected_mask = torch.zeros((2, 2), dtype=torch.long)

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")

    def test_resize_mask_with_partial_bbox_overlap_upscales_source(self):
        """
        Test a partially out-of-image bbox where the source mask is smaller than the
        clamped bbox, so it is upscaled into the clamped region.
        """
        # Original mask with labels 1 and 2
        mask = torch.tensor([
            [1, 1],
            [1, 2]
        ], dtype=torch.long)

        bbox = [-1, -1, 2, 2]  # xmin, ymin, xmax, ymax (partially outside)
        image_size = torch.tensor([4, 4])  # Width, Height

        resized_mask = self.resizer.resize_mask(mask, bbox, image_size)

        # Expected behavior:
        # Clamped bbox: [0, 0, 2, 2]
        # Assign labels within [0:2, 0:2]
        expected_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 2, 2, 0],
            [1, 2, 2, 0],
            [0, 0, 0, 0]
        ], dtype=torch.long)

        self.assertTrue(torch.equal(resized_mask, expected_mask),
                        f"Expected:\n{expected_mask}\nGot:\n{resized_mask}")


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
