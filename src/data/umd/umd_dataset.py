"""
UMD Part Affordance Dataset (Myers, Teo, Fermuller & Aloimonos, ICRA 2015).

Dataset Structure (as shipped in part-affordance-dataset-tools.tar.gz):
    <root>/part-affordance-dataset/
        category_split.txt      2-way instance split (1=train, 2=test); tools
                                from EVERY category appear on both sides
                                ("novel instance" protocol).
        novel_split.txt         2-way category split (1=train, 2=test); 9
                                categories are entirely held out for testing
                                ("novel category" protocol).
        tool_categories.txt     tool -> category id (1..17, alphabetical).
        tools/<tool>/<tool>_<frame:08d>_rgb.jpg         480x640 RGB
        tools/<tool>/<tool>_<frame:08d>_depth.png       (unused; RGB-only model)
        tools/<tool>/<tool>_<frame:08d>_label.mat       gt_label 480x640 uint8,
                                                        most-likely affordance
                                                        id per pixel, 0 = bg
        tools/<tool>/<tool>_<frame:08d>_label_rank.mat  gt_label 480x640x7,
                                                        rank per affordance

GT convention (verified on the raw data, 2026-07-11): for every affordance k,
``rank[:, :, k-1] == 1`` is pixel-identical to ``label.mat == k`` (ties aside),
so binarizing the single-label map IS the "ranked GT binarized at rank 1"
convention used by the AffordanceNet lineage. This loader therefore reads only
``*_label.mat``.

Object instances: UMD images contain exactly one tool on a clutter-free
turntable, so each image yields ONE instance whose box is the tight bounding
box of the non-zero affordance pixels (documented design choice - UMD ships no
box annotations). Frames whose label map is empty yield zero instances.

Frame subsampling: human GT exists for every 3rd frame only (per the official
README); the remaining frames carry automatically propagated labels that Myers
et al. excluded from train/test. The shipped files do not distinguish the two,
so ``frame_stride``/``frame_offset`` expose the choice to the config instead of
hard-coding it.

Target dict interface is identical to IITDetection (src/data/iit/iit_dataset.py):
    target['masks']: datapoints.Mask uint8 [N, H, W], pixel value = affordance
    class id (0 = background); boxes XYXY float32; labels = object category id.

Unlike IITDetection, no eager per-image size scan is done at init (nothing in
the pipeline consumes ``dataset.image_sizes``, and UMD is uniformly 640x480;
scanning 28,843 headers per instantiation would only slow startup).
"""

import os

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from scipy.io import loadmat
from torchvision import datapoints

from src.core import register

__all__ = ["UMDDetection"]

# Frames whose *_label.mat is human GT ("every third image", official README).
GT_FRAME_STRIDE = 3

_SPLIT_TRAIN_ID = 1
_SPLIT_TEST_ID = 2
_SPLIT_FILES = {
    "category": "category_split.txt",  # novel-instance protocol (AffordanceNet lineage)
    "novel": "novel_split.txt",  # novel-category protocol
}


@register
class UMDDetection(torch.utils.data.Dataset):
    __inject__ = ["transforms"]

    # Index == category id in tool_categories.txt (1..17, alphabetical).
    object_classes = [
        "__background__",
        "bowl",
        "cup",
        "hammer",
        "knife",
        "ladle",
        "mallet",
        "mug",
        "pot",
        "saw",
        "scissors",
        "scoop",
        "shears",
        "shovel",
        "spoon",
        "tenderizer",
        "trowel",
        "turner",
    ]

    # Index == gt_label pixel value in *_label.mat (official README order).
    affordance_classes = [
        "__background__",
        "grasp",
        "cut",
        "scoop",
        "contain",
        "pound",
        "support",
        "w-grasp",
    ]

    def __init__(
        self,
        root,
        image_set="train",
        split="category",
        frame_stride=1,
        frame_offset=0,
        transforms=None,
    ):
        """
        Args:
            root: dataset root containing ``part-affordance-dataset/``.
            image_set: 'train' or 'test'.
            split: 'category' (novel-instance, AffordanceNet lineage) or
                'novel' (novel-category).
            frame_stride: keep every Nth labeled frame per tool (1 = all
                frames; GT_FRAME_STRIDE = human-GT frames only).
            frame_offset: 0-based offset into each tool's sorted frame list
                before striding.
            transforms: optional transform pipeline (injected from YAML).
        """
        if image_set not in ("train", "test"):
            raise ValueError(f"image_set must be 'train' or 'test', got {image_set!r}")
        if split not in _SPLIT_FILES:
            raise ValueError(f"split must be one of {sorted(_SPLIT_FILES)}, got {split!r}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        if frame_offset < 0:
            raise ValueError(f"frame_offset must be >= 0, got {frame_offset}")

        self.root = root
        self.image_set = image_set
        self.split = split
        self.frame_stride = frame_stride
        self.frame_offset = frame_offset
        self._transforms = transforms

        self.data_root = os.path.join(root, "part-affordance-dataset")
        self.tools_path = os.path.join(self.data_root, "tools")
        if not os.path.isdir(self.tools_path):
            raise FileNotFoundError(
                f"UMD tools directory not found: {self.tools_path} "
                "(expected part-affordance-dataset-tools.tar.gz extracted under root)"
            )

        self.object_dict = {name: i for i, name in enumerate(self.object_classes)}
        self.affordance_dict = {name: i for i, name in enumerate(self.affordance_classes)}

        self.tools = self._load_split_tools()
        self.ids = self._index_frames()
        if not self.ids:
            raise RuntimeError(
                f"No UMD frames indexed for split={split!r}, image_set={image_set!r} "
                f"under {self.tools_path}"
            )

    # ------------------------------------------------------------------ #
    # indexing
    # ------------------------------------------------------------------ #
    def _load_split_tools(self):
        """Read the split file -> sorted list of tool names for this image_set."""
        split_path = os.path.join(self.data_root, _SPLIT_FILES[self.split])
        wanted = _SPLIT_TRAIN_ID if self.image_set == "train" else _SPLIT_TEST_ID
        tools = []
        with open(split_path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 2:
                    raise ValueError(f"Malformed split line in {split_path}: {line!r}")
                split_id, tool = int(parts[0]), parts[1]
                if split_id not in (_SPLIT_TRAIN_ID, _SPLIT_TEST_ID):
                    raise ValueError(f"Unexpected split id {split_id} in {split_path}: {line!r}")
                if self._tool_category(tool) not in self.object_dict:
                    raise ValueError(f"Unknown tool category for {tool!r} in {split_path}")
                if split_id == wanted:
                    tools.append(tool)
        return sorted(tools)

    def _index_frames(self):
        """[(tool, frame_str), ...] over every labeled frame of the split tools."""
        ids = []
        for tool in self.tools:
            tool_dir = os.path.join(self.tools_path, tool)
            suffix = "_label.mat"
            prefix = tool + "_"
            frames = sorted(
                fname[len(prefix) : -len(suffix)]
                for fname in os.listdir(tool_dir)
                if fname.startswith(prefix) and fname.endswith(suffix)
            )
            ids.extend(
                (tool, frame) for frame in frames[self.frame_offset :: self.frame_stride]
            )
        return ids

    @staticmethod
    def _tool_category(tool):
        """'knife_01' -> 'knife'."""
        return tool.rsplit("_", 1)[0]

    def _frame_path(self, index, kind):
        tool, frame = self.ids[index]
        return os.path.join(self.tools_path, tool, f"{tool}_{frame}_{kind}")

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def _load_raw(self, index):
        """Load one sample (PIL image + native-resolution target), no transforms."""
        tool, _ = self.ids[index]
        img = Image.open(self._frame_path(index, "rgb.jpg")).convert("RGB")

        label = loadmat(self._frame_path(index, "label.mat"))["gt_label"]
        label = np.asarray(label, dtype=np.uint8)
        if label.shape != (img.height, img.width):
            raise ValueError(
                f"Label shape {label.shape} != image size {(img.height, img.width)} "
                f"for {self.ids[index]}"
            )
        max_aff = len(self.affordance_classes) - 1
        if int(label.max(initial=0)) > max_aff:
            raise ValueError(
                f"Affordance id {int(label.max())} out of range [0, {max_aff}] "
                f"for {self.ids[index]}"
            )

        target = self._build_target(index, tool, label, img.height, img.width)
        return img, target

    def _build_target(self, index, tool, label, height, width):
        """One whole-tool instance: box = tight bbox of non-zero label pixels."""
        target = {"image_id": torch.tensor([index])}

        ys, xs = np.nonzero(label)
        if len(xs) > 0:
            # Half-open pixel box guarantees xmax > xmin even for 1-px parts.
            box = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ]
            boxes = torch.tensor([box], dtype=torch.float32)
            labels = torch.tensor([self.object_dict[self._tool_category(tool)]])
            masks = torch.as_tensor(label[None], dtype=torch.uint8)
        else:  # empty label map -> zero instances
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)

        target["boxes"] = datapoints.BoundingBox(
            boxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=(height, width),
        )
        target["labels"] = labels
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        target["difficult"] = torch.zeros((len(boxes),), dtype=torch.bool)
        target["masks"] = datapoints.Mask(masks)
        target["orig_size"] = torch.as_tensor([width, height])
        target["size"] = torch.as_tensor([width, height])
        return target

    def __getitem__(self, index):
        img, target = self._load_raw(index)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
            # After transformations, update the 'size' in target
            h, w = img.shape[-2:]
            target["size"] = torch.as_tensor([w, h])

        return img, target

    def __len__(self):
        return len(self.ids)

    def extra_repr(self) -> str:
        return (
            f"Split: {self.image_set} ({self.split}), tools: {len(self.tools)}, "
            f"frame_stride: {self.frame_stride}"
        )


# Mapping dictionaries mirroring src/data/iit/iit_dataset.py
umd_object_category2name = {i: name for i, name in enumerate(UMDDetection.object_classes)}
umd_object_category2label = {i: i for i in range(len(UMDDetection.object_classes))}
umd_object_label2category = {i: i for i in range(len(UMDDetection.object_classes))}

umd_affordance_category2name = {i: name for i, name in enumerate(UMDDetection.affordance_classes)}
umd_affordance_category2label = {i: i for i in range(len(UMDDetection.affordance_classes))}
umd_affordance_label2category = {i: i for i in range(len(UMDDetection.affordance_classes))}
