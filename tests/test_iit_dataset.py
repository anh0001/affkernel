"""
Test Suite for IITDetection Dataset

Description:
------------
This test suite is designed to verify the integrity and functionality of the IITDetection dataset class.
It ensures that the dataset is correctly initialized, that individual items (images and annotations) are
properly structured, and that data transformations maintain consistency across images, bounding boxes,
and masks. Additionally, the suite includes a visualization test that allows for visual inspection of
sample data with overlaid bounding boxes and masks to facilitate manual verification.

Key Features:
-------------
1. **Dataset Initialization**:
   - Checks if the dataset is not empty.
   - Verifies the correct number of object and affordance classes.

2. **Single Item Structure**:
   - Ensures that each data item contains all required keys.
   - Validates the structure and dimensions of images, bounding boxes, and masks.

3. **Transformation Consistency**:
   - Confirms that geometric transformations are consistently applied to images, bounding boxes, and masks.
   - Checks that bounding box coordinates remain within image dimensions after transformations.
   - Validates mask values against the number of affordance classes.

4. **Horizontal Flip Consistency**:
   - Tests the horizontal flip transformation by comparing manually flipped images with those processed by the transformation.

5. **Batch Collation**:
   - Verifies that data can be correctly batched.
   - Ensures that all images in a batch have the same dimensions.

6. **Visualization**:
   - Provides visual plots of sample images with overlaid bounding boxes and masks.
   - Facilitates manual inspection to ensure data integrity and transformation accuracy.

How to Run the Tests:
---------------------
1. **Enable Visualization (Optional)**:
   - The visualization test (`test_visualization`) is optional and can be enabled by setting the `ENABLE_VISUALIZATION` environment variable to `1`.
   - This prevents visualization from running during automated test runs unless explicitly desired.

   **On Unix/Linux/macOS:**
   ```bash
   export ENABLE_VISUALIZATION=1
   python -m unittest tests/test_iit_dataset.py
    ```
"""

import os
import unittest

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.data import IITDetection
from src.data.transforms import (
    Compose,
    ConvertDtype,
    RandomHorizontalFlip,
    RandomPhotometricDistort,
    RandomZoomOut,
    Resize,
    ToImageTensor,
)
from torchvision import datapoints

torch.manual_seed(0)

IIT_ROOT = './dataset/iit/data'
DATA_AVAILABLE = os.path.isdir(os.path.join(IIT_ROOT, 'VOCdevkit2012'))


@unittest.skipUnless(DATA_AVAILABLE, "IIT-AFF dataset not found under dataset/iit - skipping")
class TestIITDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup dataset path and transforms
        cls.root = IIT_ROOT
        cls.transforms = Compose([
            RandomZoomOut(fill=0),
            RandomHorizontalFlip(p=0.5),
            RandomPhotometricDistort(p=0.5),
            Resize(size=(640, 640)),
            ToImageTensor(),
            ConvertDtype(torch.float32),
        ])

        # Initialize dataset
        cls.dataset = IITDetection(
            root=cls.root,
            year='2012',
            image_set='train',
            transforms=cls.transforms
        )

    def test_dataset_initialization(self):
        """Test basic dataset properties"""
        self.assertTrue(len(self.dataset) > 0, "Dataset should not be empty")
        self.assertEqual(len(self.dataset.object_classes), 11,
                        "Should have 11 object classes (including background)")
        self.assertEqual(len(self.dataset.affordance_classes), 10,
                        "Should have 10 affordance classes (including background)")

    def test_single_item_structure(self):
        """Test structure of a single item from dataset"""
        img, target = self.dataset[0]

        # Test image properties
        self.assertIsInstance(img, torch.Tensor, "Image should be a tensor after transforms")
        self.assertEqual(img.dim(), 3, "Image should have 3 dimensions (C,H,W)")
        self.assertEqual(img.shape[0], 3, "Image should have 3 channels")
        self.assertEqual(img.shape[1:], (640, 640), "Image should be resized to 640x640")

        # Test target properties
        required_keys = ['boxes', 'labels', 'masks', 'area', 'iscrowd',
                        'image_id', 'orig_size', 'size']
        for key in required_keys:
            self.assertIn(key, target, f"Target should contain {key}")

        # Test boxes
        self.assertIsInstance(target['boxes'], datapoints.BoundingBox,
                            "Boxes should be a BoundingBox datapoint")

        # Test masks
        self.assertIsInstance(target['masks'], datapoints.Mask,
                            "Masks should be a Mask datapoint")
        self.assertEqual(target['masks'].dim(), 3,
                        "Masks should have 3 dimensions (N,H,W)")
        self.assertEqual(target['masks'].shape[1:], (640, 640),
                        "Masks should be resized to 640x640")

    def test_transform_consistency(self):
        """Test consistency between image, boxes, and masks after transformations"""
        img, target = self.dataset[0]

        boxes = target['boxes'].data
        self.assertIsNotNone(boxes, "Boxes tensor should not be None")

        # Check the rest of the properties as before
        self.assertEqual(boxes.dim(), 2, "Boxes should have shape [N, 4]")
        self.assertTrue(torch.all(boxes >= 0), "Box coordinates should be non-negative")
        self.assertTrue(torch.all(boxes[:, [0,2]] <= img.shape[2]),
                    "Box x-coordinates should be within image width")
        self.assertTrue(torch.all(boxes[:, [1,3]] <= img.shape[1]),
                    "Box y-coordinates should be within image height")

        # Check mask values
        self.assertTrue(torch.all(target['masks'] >= 0),
                    "Mask values should be non-negative")
        self.assertTrue(torch.all(target['masks'] <= 9),  # 9 affordance classes + background
                    "Mask values should not exceed number of affordance classes")

    def test_horizontal_flip_consistency(self):
        """Test consistency of horizontal flip transformation"""

        # Define a transform without RandomHorizontalFlip for the original dataset
        original_transform = Compose([
            Resize(size=(640, 640), antialias=True),
            ToImageTensor(),
            ConvertDtype(torch.float32),
        ])

        # Initialize a dataset without any horizontal flip
        dataset_original = IITDetection(
            root=self.root,
            year='2012',
            image_set='train',
            transforms=original_transform
        )

        # Initialize a dataset with deterministic horizontal flip (p=1.0)
        flip_transform = Compose([
            Resize(size=(640, 640), antialias=True),
            RandomHorizontalFlip(p=1.0),
            ToImageTensor(),
            ConvertDtype(torch.float32),
        ])

        dataset_with_flip = IITDetection(
            root=self.root,
            year='2012',
            image_set='train',
            transforms=flip_transform
        )

        # Get the original and flipped images
        orig_img, orig_target = dataset_original[0]
        flip_img, flip_target = dataset_with_flip[0]

        # Manually flip the original image
        manually_flipped_img = orig_img.flip(-1)

        # Calculate the difference
        difference = torch.abs(manually_flipped_img - flip_img)
        max_diff = difference.max()
        print(f"Maximum difference between flipped images: {max_diff.item()}")

        # Assert that the images are almost identical
        self.assertTrue(torch.allclose(
            manually_flipped_img, flip_img, atol=1e-3  # Tolerance can be adjusted if needed
        ), "Flipped image should match original image flipped horizontally")

    def test_batch_collation(self):
        """Test if the dataset can be properly collated into batches"""
        from torch.utils.data import DataLoader

        batch_size = 4
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=lambda x: tuple(zip(*x))
        )

        # Get first batch
        images, targets = next(iter(dataloader))

        # Test batch structure
        self.assertEqual(len(images), batch_size,
                        f"Should have {batch_size} images in batch")
        self.assertEqual(len(targets), batch_size,
                        f"Should have {batch_size} targets in batch")

        # Test if all images in batch have same size
        sizes = [img.shape for img in images]
        self.assertEqual(len(set(sizes)), 1,
                        "All images in batch should have same dimensions")

    def test_visualization(self):
        """Visualize a few samples from the dataset with bounding boxes and masks."""

        # Check if visualization is enabled via environment variable
        if os.getenv('ENABLE_VISUALIZATION') != '1':
            self.skipTest("Visualization not enabled. Set ENABLE_VISUALIZATION=1 to run this test.")

        num_samples = 10  # Number of samples to visualize
        for idx in range(num_samples):
            img, target = self.dataset[idx]

            # Convert image tensor to numpy array for plotting
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)  # Assuming ToImageTensor scales to [0,1]

            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(img_np)

            # Plot bounding boxes
            boxes = target['boxes'].data  # BoundingBox datapoint
            for box in boxes:
                bbox = box.numpy()
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),  # (x,y)
                    bbox[2] - bbox[0],    # width
                    bbox[3] - bbox[1],    # height
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)

            # Overlay masks
            masks = target['masks'].data  # Mask datapoint [N, H, W]
            for mask in masks:
                # Create a color mask
                mask_np = mask.cpu().numpy()
                if mask_np.sum() == 0:
                    continue  # Skip empty masks
                ax.imshow(np.ma.masked_where(mask_np == 0, mask_np),
                          cmap='jet', alpha=0.6)

            ax.set_title(f"Sample {idx}")
            plt.axis('off')
            plt.show()

if __name__ == '__main__':
    unittest.main()
