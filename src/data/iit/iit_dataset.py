"""
IIT Dataset Documentation:
The IIT-AFF Dataset contains images annotated with both objects and their affordances.

Dataset Structure:
1. Images:
   - Stored as JPEG files
   - Each image can contain multiple objects
   - Each object can have multiple affordances

2. Object Classes (10 + background):
   - Background
   - Bowl
   - TV/Monitor
   - Pan
   - Hammer
   - Knife
   - Cup
   - Drill
   - Racket
   - Spatula
   - Bottle

3. Affordance Classes (9 + background):
   - Background
   - Contain: ability to hold substances
   - Cut: ability to slice or separate
   - Display: ability to show visual information
   - Engine: ability to provide power/function
   - Grasp: ability to be held
   - Hit: ability to strike
   - Pound: ability to impact with force
   - Support: ability to hold/bear weight
   - Wrap-grasp: ability to be gripped around

4. Mask Files:
   - For each image with ID 'XXX' containing N objects, there will be N mask files:
     XXX_1_segmask.sm, XXX_2_segmask.sm, ..., XXX_N_segmask.sm
   - Each mask file has identical dimensions as its corresponding JPEG image
   - Pixel values in masks represent affordance classes (0 = background)

Example:
    Image "1939.jpg": 328x500 pixels
    Contains 3 objects, resulting in mask files:
    - 1939_1_segmask.sm (500x328)
    - 1939_2_segmask.sm (500x328)
    - 1939_3_segmask.sm (500x328)

Target Mask Dimensions:
    target['masks']: torch.Size([N, H, W]) where:
    - N: number of objects in the image (number of mask files)
    - H: height of the image
    - W: width of the image
    For example, if image 1939.jpg has 3 objects:
    target['masks'].shape = torch.Size([3, 500, 328])

by Anhar
"""

import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import datapoints

from src.core import register

__all__ = ["IITDetection"]


@register
class IITDetection(torch.utils.data.Dataset):
    __inject__ = ["transforms"]

    object_classes = [
        "__background__",
        "bowl",
        "tvm",
        "pan",
        "hammer",
        "knife",
        "cup",
        "drill",
        "racket",
        "spatula",
        "bottle",
    ]

    affordance_classes = [
        "__background__",
        "contain",
        "cut",
        "display",
        "engine",
        "grasp",
        "hit",
        "pound",
        "support",
        "w-grasp",
    ]

    def __init__(
        self,
        root,
        year="2012",
        image_set="train",
        transforms=None,
        use_difficult=False,
        copy_paste=None,
    ):
        self.root = root
        self.year = year
        self.image_set = image_set
        self._transforms = transforms
        self.use_difficult = use_difficult

        self.voc_root = os.path.join(root, f"VOCdevkit{year}", f"VOC{year}")
        self.imgs_path = os.path.join(self.voc_root, "JPEGImages")
        self.annos_path = os.path.join(self.voc_root, "Annotations")
        self.mask_cache_path = os.path.join(root, "cache", f"GTsegmask_VOC_{year}_{image_set}")

        self._load_image_set_index()

        self.object_dict = {class_name: i for i, class_name in enumerate(self.object_classes)}
        self.affordance_dict = {
            class_name: i for i, class_name in enumerate(self.affordance_classes)
        }

        self.image_sizes = {}  # Dictionary to store image sizes
        for img_id in self.ids:
            img_path = os.path.join(self.imgs_path, f"{img_id}.jpg")
            with Image.open(img_path) as img:
                self.image_sizes[img_id] = (img.width, img.height)

        # Optional targeted copy-paste augmentation, used as a recall lever.
        # Off (None) leaves __getitem__ identical to the un-augmented pipeline.
        # GUARD: train split only. yaml_utils.create() permanently merges
        # instance kwargs into the GLOBAL class node (yaml_utils.py:89), so
        # building the train dataloader first silently propagates copy_paste
        # to the val dataset too. That corrupts every in-training eval: pasted
        # donors are scored as false positives against clean file-GT, which
        # depressed our logged means by roughly 4.3pp until it was caught.
        self._copy_paste = None
        self._donor_index = []
        if copy_paste and image_set == "train":
            from .copy_paste import DEFAULT_DONOR_CLASSES, CopyPasteAugmentor

            cfg = dict(copy_paste)
            donor_names = cfg.pop("donor_classes", list(DEFAULT_DONOR_CLASSES))
            donor_ids = {self.object_dict[name] for name in donor_names}
            self._donor_index = self._build_donor_index(donor_ids)
            self._copy_paste = CopyPasteAugmentor(**cfg)

    def _load_image_set_index(self):
        image_set_file = os.path.join(self.voc_root, "ImageSets", "Main", f"{self.image_set}.txt")
        with open(image_set_file) as f:
            self.ids = [x.strip() for x in f.readlines()]

    def _load_raw(self, index):
        """Load one sample (PIL image + native-resolution target), no transforms."""
        img_id = self.ids[index]
        img_path = os.path.join(self.imgs_path, f"{img_id}.jpg")
        anno_path = os.path.join(self.annos_path, f"{img_id}.xml")

        img = Image.open(img_path).convert("RGB")
        target = self.parse_voc_xml(ET.parse(anno_path).getroot())

        # Load masks
        mask_count = 1
        masks = []
        while True:
            mask_path = os.path.join(self.mask_cache_path, f"{img_id}_{mask_count}_segmask.sm")
            if not os.path.exists(mask_path):
                break
            with open(mask_path, "rb") as f:
                mask = pickle.load(f)

            masks.append(mask)
            mask_count += 1

        if masks:
            masks_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            target["masks"] = datapoints.Mask(masks_tensor)
        else:
            target["masks"] = datapoints.Mask(
                torch.zeros((0, img.height, img.width), dtype=torch.uint8)
            )

        return img, target

    def __getitem__(self, index):
        img, target = self._load_raw(index)

        if self._copy_paste is not None:
            img, target = self._copy_paste(img, target, self._sample_donor)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
            # After transformations, update the 'size' in target
            h, w = img.shape[-2:]  # Get height and width from tensor shape
            target["size"] = torch.as_tensor([w, h])

        return img, target

    def _build_donor_index(self, donor_class_ids):
        """Index (dataset_index, object_position) of every donor-class instance.

        Object positions mirror parse_voc_xml's skip logic (difficult and
        degenerate boxes) so a position always addresses the same object in
        the parsed target — and, via the aligned mask files, the same
        per-object affordance mask.
        """
        index = []
        for i, img_id in enumerate(self.ids):
            anno_path = os.path.join(self.annos_path, f"{img_id}.xml")
            pos = 0
            for obj in ET.parse(anno_path).getroot().findall("object"):
                if not self.use_difficult and int(obj.find("difficult").text) == 1:
                    continue
                bbox = obj.find("bndbox")
                if float(bbox.find("xmax").text) <= float(bbox.find("xmin").text) or float(
                    bbox.find("ymax").text
                ) <= float(bbox.find("ymin").text):
                    continue
                if self.object_dict[obj.find("name").text.lower().strip()] in donor_class_ids:
                    index.append((i, pos))
                pos += 1
        return index

    def _sample_donor(self):
        """Sample one donor instance -> (PIL image, raw target, obj index) or None."""
        if not self._donor_index:
            return None
        j = int(torch.randint(len(self._donor_index), ()))
        ds_index, obj_pos = self._donor_index[j]
        img, target = self._load_raw(ds_index)
        # Guard: mask files must align 1:1 with parsed objects (holds for
        # IIT-AFF — zero difficult flags in 8835 annotations — but cheap).
        if len(target["masks"]) != len(target["labels"]) or obj_pos >= len(target["labels"]):
            return None
        return img, target, obj_pos

    def __len__(self):
        return len(self.ids)

    def parse_voc_xml(self, node):
        target = {}
        img_id = node.find("filename").text[:-4]
        target["image_id"] = torch.tensor([self.ids.index(img_id)])
        target["boxes"] = []
        target["labels"] = []
        target["area"] = []
        target["iscrowd"] = []
        target["difficult"] = []

        size = node.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        for obj in node.findall("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Validation
            if xmax <= xmin or ymax <= ymin:
                print(f"Invalid bbox for image {img_id}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                continue  # Skip invalid bounding boxes

            bbox = [xmin, ymin, xmax, ymax]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            target["boxes"].append(bbox)
            target["labels"].append(self.object_dict[obj.find("name").text.lower().strip()])
            target["area"].append(area)
            target["iscrowd"].append(0)
            target["difficult"].append(difficult)

        target["boxes"] = datapoints.BoundingBox(
            torch.tensor(target["boxes"], dtype=torch.float32),
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=(height, width),
        )
        target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
        target["area"] = torch.tensor(target["area"], dtype=torch.float32)
        target["iscrowd"] = torch.tensor(target["iscrowd"], dtype=torch.int64)
        target["difficult"] = torch.tensor(target["difficult"], dtype=torch.bool)
        target["orig_size"] = torch.as_tensor(
            [int(node.find("size").find("width").text), int(node.find("size").find("height").text)]
        )
        target["size"] = torch.as_tensor(
            [int(node.find("size").find("width").text), int(node.find("size").find("height").text)]
        )

        return target

    def extra_repr(self) -> str:
        return f"Split: {self.image_set}, Year: {self.year}"


# Create mapping dictionaries for both class types
iit_object_category2name = {i: name for i, name in enumerate(IITDetection.object_classes)}
iit_object_category2label = {i: i for i in range(len(IITDetection.object_classes))}
iit_object_label2category = {i: i for i in range(len(IITDetection.object_classes))}

iit_affordance_category2name = {i: name for i, name in enumerate(IITDetection.affordance_classes)}
iit_affordance_category2label = {i: i for i in range(len(IITDetection.affordance_classes))}
iit_affordance_label2category = {i: i for i in range(len(IITDetection.affordance_classes))}
