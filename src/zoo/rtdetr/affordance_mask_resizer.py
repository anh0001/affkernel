"""
RobustAffordanceMaskResizer: Specialized mask resizing utility for affordance detection
======================================================================================

This module implements a robust resizing strategy for affordance masks with multi-thresholding.
Unlike binary mask resizing used in Mask-RCNN, this approach handles multiple affordance
classes within each object through a specialized mapping and thresholding process.

Key Features:
------------
1. Linear mapping of original labels to sequential integers
2. Multi-threshold resizing to maintain class boundaries
3. Reverse mapping to restore original label values
4. Support for arbitrary number of affordance classes

Usage example with the AffordanceBranch:
# Initialize mask resizer
mask_resizer = RobustAffordanceMaskResizer(
    target_size=(244, 244),
    alpha=0.005
)

# During training
gt_mask = torch.tensor(...)  # Original ground truth mask
bbox = [xmin, ymin, xmax, ymax]  # Bounding box coordinates
image_size = torch.tensor([width, height])  # Original image size
resized_mask = mask_resizer.resize_mask(gt_mask, bbox, image_size)

# Use resized_mask for loss computation with affordance branch output
affordance_output = affordance_branch(features)
loss = criterion(affordance_output['affordances_mask'], resized_mask)

by Anhar
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


class RobustAffordanceMaskResizer:
    def __init__(self, alpha: float = 0.005):
        """
        Initialize the mask resizer.

        Args:
            alpha: Threshold margin for class boundary determination (default: 0.005)
        """
        self.alpha = alpha
        self._label_maps = {}  # Cache for label mappings

    def _create_label_mapping(self, unique_labels: torch.Tensor) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Create bidirectional mapping between original labels and sequential integers.

        Args:
            unique_labels: Tensor of unique label values in the original mask

        Returns:
            forward_map: Dict mapping original labels to sequential integers
            reverse_map: Dict mapping sequential integers back to original labels
        """
        sorted_labels = torch.sort(unique_labels)[0]

        # Ensure enumeration starts at 0
        forward_map = {int(label): idx for idx, label in enumerate(sorted_labels, start=0)}
        reverse_map = {idx: int(label) for idx, label in enumerate(sorted_labels, start=0)}

        return forward_map, reverse_map

    def _apply_thresholding(self, resized_mask: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Apply multi-thresholding to resized mask values.

        Args:
            resized_mask: Tensor containing interpolated mask values
            num_classes: Number of classes in the mapped label space

        Returns:
            Tensor containing thresholded class assignments
        """
        # Initialize output with background class (0)
        thresholded = torch.zeros_like(resized_mask, dtype=torch.long)

        # Apply thresholding for each class
        for class_idx in range(num_classes):
            # Assign class_idx where the resized_mask is within [class_idx - alpha, class_idx + alpha]
            class_mask = (resized_mask >= (class_idx - self.alpha)) & (resized_mask <= (class_idx + self.alpha))
            thresholded[class_mask] = class_idx

        # Optionally handle values outside the class range
        # For example, set to background or a default class
        # Here, we leave them as background (0)

        return thresholded

    def resize_mask(self, mask: torch.Tensor, bbox: List[float], image_size: torch.Tensor) -> torch.Tensor:
        """
        Resize an affordance mask output from an object detection model.
        The algorithm detects an object in an image outputting a bounding box and a mask that have similar size.
        Using the resize_mask function, the mask is scaled to the desired bbox dimensions.
        As the mask is not a full image, the final output is a full-size image where the region
        corresponding to the bbox contains the scaled mask and all pixels outside the bbox are set to zero.

        Args:
            mask (torch.Tensor): The input affordance mask (of shape [H, W] or [1, H, W]) of a detected object,
                                 containing discrete class labels.
            bbox (List[float]): A list of four values [xmin, ymin, xmax, ymax] defining the region in the full image
                                where the mask should be placed.
            image_size (torch.Tensor): A tensor [width, height] indicating the dimensions of the full image.

        Returns:
            torch.Tensor: A full-size mask (shape [height, width]) where the region defined by the bbox contains the
                          scaled affordance mask (mapped back to original labels) and all pixels outside the bbox
                          are set to background (0).
        """
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        # Predictions arrive on the model device (e.g. CUDA); all the
        # numpy/PIL ops below require a host tensor.
        mask = mask.detach().cpu()

        # Get unique labels and create mapping if not cached
        unique_labels = torch.unique(mask)
        cache_key = tuple(unique_labels.tolist())

        if cache_key not in self._label_maps:
            self._label_maps[cache_key] = self._create_label_mapping(unique_labels)

        forward_map, reverse_map = self._label_maps[cache_key]

        # Map original labels to sequential integers
        mapped_mask = torch.zeros_like(mask)
        for orig_label, mapped_label in forward_map.items():
            mapped_mask[mask == orig_label] = mapped_label

        # Extract raw bounding box coords
        xmin, ymin, xmax, ymax = bbox

        # Compute preliminary bbox width/height
        pre_width = int(round(xmax - xmin))
        pre_height = int(round(ymax - ymin))

        if pre_width <= 0 or pre_height <= 0:
            # print(f"Invalid bbox dimensions (before clamp): width={pre_width}, height={pre_height}")
            return torch.zeros(image_size[1], image_size[0], dtype=torch.long)

        # Clamp coordinates
        x_start = max(int(round(xmin)), 0)
        y_start = max(int(round(ymin)), 0)
        x_end = min(x_start + pre_width, image_size[0])
        y_end = min(y_start + pre_height, image_size[1])

        # Recompute final clamped dimensions
        bbox_width = x_end - x_start
        bbox_height = y_end - y_start

        if bbox_width <= 0 or bbox_height <= 0:
            print(f"Clamped bbox results in non-positive size: width={bbox_width}, height={bbox_height}")
            return torch.zeros(image_size[1], image_size[0], dtype=torch.long)

        # Resize using clamped dimensions
        mask_pil = Image.fromarray(mapped_mask.numpy().astype(np.uint8)).convert('L')
        resized_mask_pil = mask_pil.resize((bbox_width, bbox_height), resample=Image.NEAREST)
        resized_mask = torch.from_numpy(np.array(resized_mask_pil)).long()

        # Apply multi-thresholding
        thresholded = self._apply_thresholding(resized_mask, len(forward_map))

        # Create and fill final mask
        final_mask = torch.zeros(image_size[1], image_size[0], dtype=torch.long)
        mask_region = final_mask[y_start:y_end, x_start:x_end]

        for mapped_label, orig_label in reverse_map.items():
            mask_condition = (thresholded == mapped_label)
            mask_region[mask_condition] = orig_label

        return final_mask

    # New method to crop and resize a full-size mask using nearest neighbor interpolation.
    def crop_and_resize_mask(
        self,
        full_mask: torch.Tensor,
        bbox: List[float],
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Extract a region from the full image mask defined by the bbox and
        resize the cropped mask to the desired target dimensions.

        Args:
            full_mask (torch.Tensor): Full image mask of shape [H, W] containing
                discrete affordance labels.
            bbox (List[float]): Bounding box specified as [xmin, ymin, xmax, ymax].
            target_size (Tuple[int, int]): Desired output size as (width, height).

        Returns:
            torch.Tensor: A resized mask with shape [target_height, target_width].
                If the bbox is invalid, returns a tensor of zeros with the target size.
        """
        # Ensure full_mask is on CPU for PIL conversion
        if full_mask.device.type != "cpu":
            full_mask_cpu = full_mask.detach().cpu()
        else:
            full_mask_cpu = full_mask

        # Full image dimensions
        H_full, W_full = full_mask_cpu.shape

        # Parse and clamp bbox coordinates
        xmin, ymin, xmax, ymax = bbox
        x_start = max(int(round(xmin)), 0)
        y_start = max(int(round(ymin)), 0)
        x_end = min(int(round(xmax)), W_full)
        y_end = min(int(round(ymax)), H_full)

        # If the bounding box is invalid, return a zero mask of the target size.
        if x_end <= x_start or y_end <= y_start:
            return torch.zeros(target_size[1], target_size[0], dtype=full_mask.dtype)

        # Crop the full mask using the clamped bounding box coordinates
        cropped_mask = full_mask_cpu[y_start:y_end, x_start:x_end]

        # Convert the cropped tensor to a PIL Image for resizing.
        cropped_pil = Image.fromarray(cropped_mask.numpy().astype(np.uint8))

        # Resize the cropped mask to the desired target dimensions using nearest neighbor.
        resized_pil = cropped_pil.resize(target_size, resample=Image.NEAREST)

        # Convert the resized PIL image back to a torch.Tensor.
        resized_mask = torch.from_numpy(np.array(resized_pil)).type(full_mask.dtype)

        return resized_mask
