import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from src.data.iit.iit_eval import IITEvaluator


class MockDataset:
    """
    A mock dataset to provide necessary attributes for IITEvaluator.
    """
    def __init__(self, object_classes, affordance_classes, aff_dict, annos_path, mask_cache_path):
        self.object_classes = object_classes
        self.affordance_classes = affordance_classes
        self.affordance_dict = aff_dict
        self.annos_path = annos_path
        self.mask_cache_path = mask_cache_path

    def side_effect_load_ground_truth(self, cls):
        """
        Custom side effect function for load_ground_truth mock.
        Returns ground truth boxes based on the class name.
        """
        class_gt = {
            'bowl': [{'bbox': [50, 50, 150, 150]}],
            'pan': [{'bbox': [30, 30, 100, 100]}],
            'hammer': [{'bbox': [120, 120, 180, 180]}],
            # Add more classes as needed
            'cup': [{'bbox': [60, 60, 160, 160]}],  # For test_no_predictions
        }
        return class_gt.get(cls, [])

    def side_effect_load_ground_truth_masks(self, aff_cls):
        """
        Custom side effect function for load_ground_truth_masks mock.
        Returns ground truth mask dictionaries keyed by affordance class.
        """
        class_masks = {
            'contain': [{'mask': np.random.randint(0, 2, (200, 200), dtype=np.uint8)}],
            'grasp': [{'mask': np.random.randint(0, 2, (200, 200), dtype=np.uint8)}],
            'pound': [{'mask': np.random.randint(0, 2, (200, 200), dtype=np.uint8)}],
            'cut': [{'mask': np.random.randint(0, 2, (200, 200), dtype=np.uint8)}],
            # Add more classes as needed
            'cup': [{'mask': np.random.randint(0, 2, (200, 200), dtype=np.uint8)}],
        }
        return class_masks.get(aff_cls, [])

class TestIITEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up a mock dataset and IITEvaluator instance for testing.
        """
        # Define mock object classes and affordance classes
        cls.object_classes = ['__background__', 'bowl', 'tvm', 'pan', 'hammer', 'knife',
                              'cup', 'drill', 'racket', 'spatula', 'bottle']
        cls.affordance_classes = ['__background__', 'contain', 'cut', 'display', 'engine',
                                   'grasp', 'hit', 'pound', 'support', 'w-grasp']

        cls.affordance_dict = {name: idx for idx, name in enumerate(cls.affordance_classes)}

        # Create temporary directories for annotations and masks
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.annos_path = cls.temp_dir.name
        cls.mask_cache_path = os.path.join(cls.temp_dir.name, 'masks')
        os.makedirs(cls.mask_cache_path, exist_ok=True)

        # Initialize mock dataset
        cls.dataset = MockDataset(
            object_classes=cls.object_classes,
            affordance_classes=cls.affordance_classes,
            aff_dict=cls.affordance_dict,
            annos_path=cls.annos_path,
            mask_cache_path=cls.mask_cache_path
        )

        # Initialize IITEvaluator
        cls.evaluator = IITEvaluator(
            dataset=cls.dataset,
            iou_thresh=0.5,
            use_07_metric=True,
            use_affordance=True
        )

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the temporary directory after tests.
        """
        cls.temp_dir.cleanup()

    def setUp(self):
        """
        Reset the evaluator state before each test.
        """
        self.evaluator.reset()

    def create_mock_annotation(self, img_id, objects):
        """
        Create a mock annotation XML file.

        Args:
            img_id (str): Image identifier.
            objects (list): List of object dictionaries with keys 'name' and 'bbox'.
        """
        anno_path = os.path.join(self.annos_path, f"{img_id}.xml")
        with open(anno_path, 'w') as f:
            f.write('<annotation>\n')
            size_written = False
            for obj in objects:
                f.write('  <object>\n')
                f.write(f"    <name>{obj['name']}</name>\n")
                f.write("    <pose>Unspecified</pose>\n")
                f.write("    <truncated>0</truncated>\n")
                f.write("    <difficult>0</difficult>\n")
                f.write("    <bndbox>\n")
                f.write(f"      <xmin>{obj['bbox'][0]}</xmin>\n")
                f.write(f"      <ymin>{obj['bbox'][1]}</ymin>\n")
                f.write(f"      <xmax>{obj['bbox'][2]}</xmax>\n")
                f.write(f"      <ymax>{obj['bbox'][3]}</ymax>\n")
                f.write("    </bndbox>\n")
                f.write("  </object>\n")
                if not size_written:
                    # Assuming all objects in the same image have the same size
                    # For testing, we can set a fixed image size
                    f.write("  <size>\n")
                    f.write("    <width>200</width>\n")
                    f.write("    <height>200</height>\n")
                    f.write("    <depth>3</depth>\n")
                    f.write("  </size>\n")
                    size_written = True
            f.write('</annotation>\n')

    def create_mock_mask(self, img_id, mask_id, class_name, mask_shape=(200, 200)):
        """
        Create a mock affordance mask file.

        Args:
            img_id (str): Image identifier.
            mask_id (int): Mask identifier.
            class_name (str): Affordance class name.
            mask_shape (tuple): Shape of the mask.
        """
        mask_path = os.path.join(self.mask_cache_path, f"{img_id}_{mask_id}_segmask.sm")
        mask = np.zeros(mask_shape, dtype=np.uint8)
        # Assign the affordance class index to a specific region for testing
        # For simplicity, let's set a square region
        mask[50:150, 50:150] = self.dataset.affordance_dict[class_name]
        with open(mask_path, 'wb') as f:
            pickle.dump(mask, f)

    def test_evaluator_initialization(self):
        """
        Test if the evaluator is initialized correctly.
        """
        self.assertEqual(self.evaluator.iou_thresh, 0.5, "IOU threshold should be 0.5")
        self.assertTrue(self.evaluator.use_07_metric, "use_07_metric should be True")
        self.assertTrue(self.evaluator.use_affordance, "use_affordance should be True")
        self.assertListEqual(self.evaluator.object_classes, self.object_classes,
                             "Object classes should match the dataset's object classes")
        self.assertListEqual(self.evaluator.affordance_classes, self.affordance_classes,
                             "Affordance classes should match the dataset's affordance classes")
        self.assertIn('bbox', self.evaluator.iou_types, "IoU types should include 'bbox'")
        self.assertIn('affordance', self.evaluator.iou_types, "IoU types should include 'affordance'")

    def test_update_and_summarize_single_image(self):
        """
        Test updating the evaluator with predictions from a single image and summarizing results.
        """
        img_id = 'image_1'
        objects = [{
            'name': 'bowl',
            'bbox': [50, 50, 150, 150]
        }]
        self.create_mock_annotation(img_id, objects)
        self.create_mock_mask(img_id, mask_id=1, class_name='contain')

        # Mock predictions
        predictions = {
            img_id: {
                'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),  # 'bowl'
                'scores': torch.tensor([0.9], dtype=torch.float32),
                'affordances': torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.float32)  # 'contain'
            }
        }

        # Mock ground truth parsing by overriding load_ground_truth and load_ground_truth_masks
        self.evaluator.load_ground_truth = MagicMock(return_value=[{
            'bbox': [50, 50, 150, 150]
        }])
        self.evaluator.load_ground_truth_masks = MagicMock(return_value=[
            {"mask": np.random.randint(0, 2, (200, 200), dtype=np.uint8)}
        ])

        # Update evaluator with predictions
        self.evaluator.update(predictions)

        # Since we're not in a distributed setting, skip synchronization

        # Accumulate results
        self.evaluator.accumulate()

        # Capture the summarize output
        with tempfile.TemporaryFile(mode='w+') as tmp_stdout:
            original_stdout = os.dup(1)
            os.dup2(tmp_stdout.fileno(), 1)
            try:
                self.evaluator.summarize()
                sys.stdout.flush()
                os.fsync(1)
                tmp_stdout.seek(0)
                summarize_output = tmp_stdout.read()
            finally:
                os.dup2(original_stdout, 1)
                os.close(original_stdout)

        # Check if AP is computed and printed
        self.assertIn("AP for bowl", summarize_output, "Summarize should include 'bowl' class AP")
        self.assertIn("mAP for Bounding Boxes", summarize_output, "Summarize should include mAP for bounding boxes")
        if self.evaluator.use_affordance:
            self.assertIn("AP for Affordance contain", summarize_output, "Summarize should include affordance AP for 'contain'")
            self.assertIn("mAP for Affordance Masks", summarize_output, "Summarize should include mAP for affordance masks")

    def test_update_with_multiple_images(self):
        """
        Test updating the evaluator with predictions from multiple images and summarizing results.
        """
        objects_img1 = [{
            'name': 'bowl',
            'bbox': [50, 50, 150, 150]
        }]
        objects_img2 = [
            {
                'name': 'pan',
                'bbox': [30, 30, 100, 100]
            },
            {
                'name': 'hammer',
                'bbox': [120, 120, 180, 180]
            }
        ]
        self.create_mock_annotation('image_1', objects_img1)
        self.create_mock_mask('image_1', mask_id=1, class_name='contain')

        self.create_mock_annotation('image_2', objects_img2)
        self.create_mock_mask('image_2', mask_id=1, class_name='grasp')
        self.create_mock_mask('image_2', mask_id=2, class_name='pound')

        # Mock predictions
        predictions = {
            'image_1': {
                'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),  # 'bowl'
                'scores': torch.tensor([0.9], dtype=torch.float32),
                'affordances': torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.float32)  # 'contain'
            },
            'image_2': {
                'boxes': torch.tensor([[30, 30, 100, 100], [120, 120, 180, 180]], dtype=torch.float32),
                'labels': torch.tensor([3, 4], dtype=torch.int64),  # 'pan' and 'hammer'
                'scores': torch.tensor([0.85, 0.75], dtype=torch.float32),
                'affordances': torch.tensor([
                    [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],  # 'grasp'
                    [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]   # 'pound'
                ], dtype=torch.float32)
            }
        }

        # Mock ground truth parsing by overriding load_ground_truth and load_ground_truth_masks
        self.evaluator.load_ground_truth = MagicMock(side_effect=self.dataset.side_effect_load_ground_truth)
        self.evaluator.load_ground_truth_masks = MagicMock(side_effect=self.dataset.side_effect_load_ground_truth_masks)

        # Update evaluator with predictions
        self.evaluator.update(predictions)

        # Since we're not in a distributed setting, skip synchronization

        # Accumulate results
        self.evaluator.accumulate()

        # Capture the summarize output
        with tempfile.TemporaryFile(mode='w+') as tmp_stdout:
            original_stdout = os.dup(1)
            os.dup2(tmp_stdout.fileno(), 1)
            try:
                self.evaluator.summarize()
                sys.stdout.flush()
                os.fsync(1)
                tmp_stdout.seek(0)
                summarize_output = tmp_stdout.read()
            finally:
                os.dup2(original_stdout, 1)
                os.close(original_stdout)

        # Check if AP is computed and printed for all classes
        for cls in ['bowl', 'pan', 'hammer']:
            self.assertIn(f"AP for {cls}", summarize_output, f"Summarize should include '{cls}' class AP")
        if self.evaluator.use_affordance:
            self.assertIn("AP for Affordance contain", summarize_output, "Summarize should include affordance AP for 'contain'")
            self.assertIn("AP for Affordance grasp", summarize_output, "Summarize should include affordance AP for 'grasp'")
            self.assertIn("AP for Affordance pound", summarize_output, "Summarize should include affordance AP for 'pound'")
            self.assertIn("mAP for Bounding Boxes", summarize_output, "Summarize should include mAP for bounding boxes")
            self.assertIn("mAP for Affordance Masks", summarize_output, "Summarize should include mAP for affordance masks")

    def test_invalid_bboxes_handling(self):
        """
        Test how the evaluator handles invalid bounding boxes.
        """
        img_id = 'image_invalid'
        objects = [{
            'name': 'bowl',
            'bbox': [150, 150, 100, 100]  # Invalid bbox: xmax < xmin and ymax < ymin
        }]
        self.create_mock_annotation(img_id, objects)
        self.create_mock_mask(img_id, mask_id=1, class_name='contain')

        # Mock predictions with invalid bbox
        predictions = {
            img_id: {
                'boxes': torch.tensor([[150, 150, 100, 100]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),  # 'bowl'
                'scores': torch.tensor([0.6], dtype=torch.float32),
                'affordances': torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.float32)  # 'contain'
            }
        }

        # Mock ground truth parsing
        self.evaluator.load_ground_truth = MagicMock(return_value=[{
            'bbox': [150, 150, 100, 100]
        }])
        self.evaluator.load_ground_truth_masks = MagicMock(return_value=[{'mask': np.random.randint(0, 2, (200, 200), dtype=np.uint8)}])

        # Update evaluator with predictions
        self.evaluator.update(predictions)

        # Since we're not in a distributed setting, skip synchronization

        # Accumulate results
        self.evaluator.accumulate()

        # Capture the summarize output
        with tempfile.TemporaryFile(mode='w+') as tmp_stdout:
            original_stdout = os.dup(1)
            os.dup2(tmp_stdout.fileno(), 1)
            try:
                self.evaluator.summarize()
                sys.stdout.flush()
                os.fsync(1)
                tmp_stdout.seek(0)
                summarize_output = tmp_stdout.read()
            finally:
                os.dup2(original_stdout, 1)
                os.close(original_stdout)

        # Check if invalid bbox was handled (AP should be 0 for this class)
        self.assertIn("AP for bowl: 0.0000", summarize_output, "AP for 'bowl' should be 0.0000 due to invalid bbox")
        self.assertIn("mAP for Bounding Boxes: 0.0000", summarize_output, "mAP for Bounding Boxes should be 0.0000")
        if self.evaluator.use_affordance:
            self.assertIn("AP for Affordance contain", summarize_output, "Summarize should include affordance AP for 'contain'")
            self.assertIn("mAP for Affordance Masks", summarize_output, "mAP for Affordance Masks should be 0.0000")

    def test_no_affordances_present(self):
        """
        Test evaluator behavior when no affordances are present in predictions.
        """
        img_id = 'image_no_affordance'
        objects = [{
            'name': 'knife',
            'bbox': [60, 60, 160, 160]
        }]
        self.create_mock_annotation(img_id, objects)
        self.create_mock_mask(img_id, mask_id=1, class_name='cut')

        # Mock predictions without affordances
        predictions = {
            img_id: {
                'boxes': torch.tensor([[60, 60, 160, 160]], dtype=torch.float32),
                'labels': torch.tensor([5], dtype=torch.int64),  # 'knife'
                'scores': torch.tensor([0.8], dtype=torch.float32),
                # 'affordances' key is missing
            }
        }

        # Mock ground truth parsing
        self.evaluator.load_ground_truth = MagicMock(return_value=[{
            'bbox': [60, 60, 160, 160]
        }])
        self.evaluator.load_ground_truth_masks = MagicMock(return_value=[self.dataset.affordance_dict['cut']])

        # Update evaluator with predictions
        self.evaluator.update(predictions)

        # Accumulate results
        self.evaluator.accumulate()

        # Capture the summarize output
        with tempfile.TemporaryFile(mode='w+') as tmp_stdout:
            original_stdout = os.dup(1)
            os.dup2(tmp_stdout.fileno(), 1)
            try:
                self.evaluator.summarize()
                sys.stdout.flush()
                os.fsync(1)
                tmp_stdout.seek(0)
                summarize_output = tmp_stdout.read()
            finally:
                os.dup2(original_stdout, 1)
                os.close(original_stdout)

        # Check if AP is computed for bounding boxes only
        self.assertIn("AP for knife", summarize_output, "Summarize should include 'knife' class AP")
        self.assertIn("mAP for Bounding Boxes", summarize_output, "Summarize should include mAP for bounding boxes")
        if self.evaluator.use_affordance:
            self.assertNotIn("AP for Affordance", summarize_output, "Summarize should not include affordance AP when affordances are missing")
            self.assertNotIn("mAP for Affordance Masks", summarize_output, "Summarize should not include mAP for affordance masks when affordances are missing")

    def test_multiple_affordances_per_object(self):
        """
        Test evaluator handling objects with multiple affordances.
        """
        img_id = 'image_multiple_affordances'
        objects = [{
            'name': 'knife',
            'bbox': [40, 40, 140, 140]
        }]
        self.create_mock_annotation(img_id, objects)
        self.create_mock_mask(img_id, mask_id=1, class_name='cut')
        self.create_mock_mask(img_id, mask_id=2, class_name='grasp')

        # Mock predictions with multiple affordances
        predictions = {
            img_id: {
                'boxes': torch.tensor([[40, 40, 140, 140]], dtype=torch.float32),
                'labels': torch.tensor([5], dtype=torch.int64),  # 'knife'
                'scores': torch.tensor([0.95], dtype=torch.float32),
                'affordances': torch.tensor([
                    [[0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]  # 'cut' and 'grasp'
                ], dtype=torch.float32)  # Affordances for 'knife'
            }
        }

        # Mock ground truth parsing
        self.evaluator.load_ground_truth = MagicMock(return_value=[{
            'bbox': [40, 40, 140, 140]
        }])
        self.evaluator.load_ground_truth_masks = MagicMock(return_value=[
            {"mask": np.random.randint(0, 2, (200, 200), dtype=np.uint8)},
            {"mask": np.random.randint(0, 2, (200, 200), dtype=np.uint8)}
        ])

        # Update evaluator with predictions
        self.evaluator.update(predictions)

        # Accumulate results
        self.evaluator.accumulate()

        # Capture the summarize output
        with tempfile.TemporaryFile(mode='w+') as tmp_stdout:
            original_stdout = os.dup(1)
            os.dup2(tmp_stdout.fileno(), 1)
            try:
                self.evaluator.summarize()
                sys.stdout.flush()
                os.fsync(1)
                tmp_stdout.seek(0)
                summarize_output = tmp_stdout.read()
            finally:
                os.dup2(original_stdout, 1)
                os.close(original_stdout)

        # Check if AP is computed for both affordances
        self.assertIn("AP for knife", summarize_output, "Summarize should include 'knife' class AP")
        self.assertIn("AP for Affordance cut", summarize_output, "Summarize should include affordance AP for 'cut'")
        self.assertIn("AP for Affordance grasp", summarize_output, "Summarize should include affordance AP for 'grasp'")
        self.assertIn("mAP for Bounding Boxes", summarize_output, "Summarize should include mAP for bounding boxes")
        self.assertIn("mAP for Affordance Masks", summarize_output, "Summarize should include mAP for affordance masks")

    def test_no_predictions(self):
        """
        Test evaluator behavior when there are no predictions for certain classes.
        Specifically, ensure that AP is reported as 0.0 for classes with ground truths but no predictions.
        """
        img_id = 'image_no_predictions'
        objects = [{
            'name': 'bowl',  # This object has the 'contain' affordance
            'bbox': [50, 50, 150, 150]
        }]
        self.create_mock_annotation(img_id, objects)
        self.create_mock_mask(img_id, mask_id=1, class_name='contain')

        # Provide no predictions for 'contain' class
        predictions = {
            img_id: {
                'boxes': torch.tensor([[60, 60, 140, 140]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),  # 'bowl'
                'scores': torch.tensor([0.8], dtype=torch.float32),
                'affordances': torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.float32)  # No 'contain' affordance predicted
            }
        }

        # Mock ground truth parsing by overriding load_ground_truth and load_ground_truth_masks
        self.evaluator.load_ground_truth = MagicMock(return_value=[{
            'bbox': [50, 50, 150, 150]
        }])
        self.evaluator.load_ground_truth_masks = MagicMock(return_value=[
            {"mask": np.random.randint(0, 2, (200, 200), dtype=np.uint8)}
        ])

        # Update evaluator with predictions
        self.evaluator.update(predictions)

        # Accumulate results
        self.evaluator.accumulate()

        # Capture the summarize output
        with tempfile.TemporaryFile(mode='w+') as tmp_stdout:
            original_stdout = os.dup(1)
            os.dup2(tmp_stdout.fileno(), 1)
            try:
                self.evaluator.summarize()
                sys.stdout.flush()
                os.fsync(1)
                tmp_stdout.seek(0)
                summarize_output = tmp_stdout.read()
            finally:
                os.dup2(original_stdout, 1)
                os.close(original_stdout)

        # Check that AP for 'contain' is reported as 0.0
        self.assertIn("AP for Affordance contain: 0.0000", summarize_output,
                    "Summarize should include 'AP for Affordance contain: 0.0000' when there are GTs but no predictions.")

        # Additionally, ensure mAP is correctly calculated
        self.assertIn("mAP for Affordance Masks: 0.0000", summarize_output,
                    "mAP for Affordance Masks should be 0.0000 when no predictions are correct.")

if __name__ == '__main__':
    unittest.main()
