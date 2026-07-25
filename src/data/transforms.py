"""
Transforms for Object Detection and Mask Processing
------------------------------------------------

This module provides custom transforms for processing both images and their corresponding masks
in object detection tasks. The key challenge it addresses is handling multiple instance masks
per image while maintaining proper geometric transformations.

Key Problems Solved:
1. Multiple Masks Per Image:
   - Standard torchvision transforms are designed for single mask segmentation
   - In instance segmentation/detection, each image can have N masks (one per object)
   - These masks are stacked as [N, H, W] tensors

2. Geometric Transformation Consistency:
   - When applying geometric transforms (resize, flip, crop), both image and all masks
     must be transformed identically
   - Bounding boxes must also be transformed consistently with masks

3. Transform Pipeline Management:
   - Some transforms only affect images (e.g., color distortion)
   - Some affect both images and masks (e.g., resize, crop)
   - Need to handle both types seamlessly

Key Components:
1. MultiMaskTransform:
   - Wrapper class that handles multiple instance masks
   - Applies transforms to each mask separately while maintaining consistency
   - Preserves mask stacking structure [N, H, W]

2. Wrapped Geometric Transforms:
   - RandomZoomOut, RandomIoUCrop, RandomHorizontalFlip, Resize
   - Automatically wrapped with MultiMaskTransform for proper mask handling

3. Direct Transforms:
   - RandomPhotometricDistort, ToImageTensor, etc.
   - Applied directly as they don't affect mask geometry

Usage:
    transforms:
      ops:
        - {type: RandomZoomOut, fill: 0}           # Geometric - uses MultiMaskTransform
        - {type: RandomHorizontalFlip}             # Geometric - uses MultiMaskTransform
        - {type: RandomPhotometricDistort, p: 0.5} # Non-geometric - direct application
        - {type: Resize, size: [640, 640]}         # Geometric - uses MultiMaskTransform

Note:
    The transforms maintain the structure of the target dictionary, which includes:
    - boxes: Bounding boxes in various formats (XYXY, CXCYWH)
    - masks: Instance masks tensor [N, H, W]
    - labels: Classification labels
    - other metadata

By Anhar
"""


import torch
import torch.nn as nn
import torchvision

torchvision.disable_beta_transforms_warning()
# E402: the imports below reach torchvision's beta datapoints/v2 API and must stay
# after disable_beta_transforms_warning(); reordering re-enables the beta warning.
from typing import Any, Dict, List, Optional  # noqa: E402

import torchvision.transforms.v2 as T  # noqa: E402
import torchvision.transforms.v2.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import datapoints  # noqa: E402

from src.core import GLOBAL_CONFIG, register  # noqa: E402

__all__ = ['Compose', ]

RandomPhotometricDistort = register(T.RandomPhotometricDistort)
ToImageTensor = register(T.ToImageTensor)
ConvertDtype = register(T.ConvertDtype)
SanitizeBoundingBox = register(T.SanitizeBoundingBox)
Normalize = register(T.Normalize)

@register
class MultiMaskTransform(T.Transform):
    """
    Handle multiple masks and other annotations for each image by applying transforms jointly
    to the image, masks, and all associated annotations.

    This class ensures that all geometric transformations are applied consistently across the image,
    all masks, bounding boxes, and any other spatial annotations, maintaining spatial alignment
    and preventing inconsistencies in spatial dimensions.
    """

    def __init__(self, transform: T.Transform) -> None:
        """
        Initialize the MultiMaskTransform with a given transform.

        Args:
            transform (T.Transform): The geometric transform to be applied jointly to the image, masks,
                                     and other annotations.
        """
        super().__init__()
        self.transform = transform

    def forward(self, *args, **kwargs):
        """
        Apply the transformation jointly to the image and all annotations.

        Args:
            img (Image.Image): The input image.
            target (Dict[str, Any]): The target dictionary containing annotations like masks, boxes, etc.

        Returns:
            Tuple containing:
                - Transformed image.
                - Updated target dictionary with transformed annotations.
        """

        # Handle both positional and keyword arguments
        if len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 2:
            img, target = args[0]
        elif len(args) == 2:
            img, target = args
        else:
            # Try to get from kwargs
            img = kwargs.get('img', None)
            target = kwargs.get('target', None)
            if img is None or target is None:
                raise ValueError("Expected img and target arguments either as tuple/list, separate arguments, or kwargs")

        if not isinstance(target, dict):
            # If target is not a dictionary, apply transform directly
            return self.transform(img, target)

        if 'masks' not in target and 'boxes' not in target:
            # If there are no masks or boxes, apply transform directly
            return self.transform(img, target)

        # Optional Assertions
        assert isinstance(target['boxes'], datapoints.BoundingBox), \
            f"Expected 'boxes' to be datapoints.BoundingBox, got {type(target['boxes'])}"

        assert isinstance(target['masks'], datapoints.Mask), \
            f"Expected 'masks' to be datapoints.Mask, got {type(target['masks'])}"

        # Apply the transformation jointly to the image and target
        transformed_img, transformed_target = self.transform(img, target)

        return transformed_img, transformed_target

# Register the wrapped versions of geometric transforms
@register
class RandomZoomOut(T.Transform):
    def __init__(self, **kwargs):
        super().__init__()
        self.transform = MultiMaskTransform(T.RandomZoomOut(**kwargs))

    def forward(self, *args):
        return self.transform(*args)

@register
class RandomHorizontalFlip(T.Transform):
    def __init__(self, **kwargs):
        super().__init__()
        self.transform = MultiMaskTransform(T.RandomHorizontalFlip(**kwargs))

    def forward(self, *args):
        return self.transform(*args)

@register
class Resize(T.Transform):
    def __init__(self, **kwargs):
        super().__init__()
        self.transform = MultiMaskTransform(T.Resize(**kwargs))

    def forward(self, *args):
        return self.transform(*args)

@register
class RandomCrop(T.Transform):
    def __init__(self, **kwargs):
        super().__init__()
        self.transform = MultiMaskTransform(T.RandomCrop(**kwargs))

    def forward(self, *args):
        return self.transform(*args)

@register
class RandomIoUCrop(T.Transform):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1,
                 min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2,
                 sampler_options: Optional[List[float]] = None, trials: int = 40,
                 p: float = 1.0):
        super().__init__()
        # Create the base IoU crop transform
        base_transform = T.RandomIoUCrop(
            min_scale=min_scale,
            max_scale=max_scale,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            sampler_options=sampler_options,
            trials=trials
        )
        # Wrap it with MultiMaskTransform for mask handling
        self.transform = MultiMaskTransform(base_transform)
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        # Apply probability
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        # Apply transform with mask handling
        return self.transform(*inputs)



@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                    # op['type'] = name
                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]

        super().__init__(transforms=transforms)

@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    _transformed_types = (
        Image.Image,
        datapoints.Image,
        datapoints.Video,
        datapoints.Mask,
        datapoints.BoundingBox,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)

        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs

@register
class ConvertBox(T.Transform):
    _transformed_types = (
        datapoints.BoundingBox,
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if self.out_fmt:
            spatial_size = inpt.spatial_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]

        return inpt
