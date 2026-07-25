"""Tests for the targeted copy-paste augmentation (A2 recall lever).

Covers the pure augmentor (src/data/iit/copy_paste.py) on synthetic samples —
no dataset files or GPU needed — plus the donor-index position alignment
logic of IITDetection._build_donor_index via temp XML annotations.
"""

import os
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from src.data.iit.copy_paste import CopyPasteAugmentor
from src.data.iit.iit_dataset import IITDetection
from torchvision import datapoints

GRASP, WGRASP = 5, 9  # affordance labels
HAMMER, BOTTLE = 4, 10  # object labels


def _dst_sample(box=(10, 10, 50, 50), h=100, w=120):
    """One existing hammer with a grasp-labelled mask filling its box."""
    img = Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8))
    mask = torch.zeros(h, w, dtype=torch.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = GRASP
    boxes = datapoints.BoundingBox(
        torch.tensor([box], dtype=torch.float32),
        format=datapoints.BoundingBoxFormat.XYXY,
        spatial_size=(h, w),
    )
    target = {
        "boxes": boxes,
        "labels": torch.tensor([HAMMER], dtype=torch.int64),
        "masks": datapoints.Mask(mask[None]),
        "area": torch.tensor([float((x1 - x0) * (y1 - y0))]),
        "iscrowd": torch.zeros(1, dtype=torch.int64),
        "difficult": torch.zeros(1, dtype=torch.bool),
        "image_id": torch.tensor([0]),
        "orig_size": torch.tensor([w, h]),
        "size": torch.tensor([w, h]),
    }
    return img, target


def _donor_sampler():
    """Bottle donor: 80x80 image, w-grasp labels filling [20,60)x[20,60)."""
    img = Image.fromarray(np.full((80, 80, 3), 200, dtype=np.uint8))
    mask = torch.zeros(80, 80, dtype=torch.uint8)
    mask[20:60, 20:60] = WGRASP
    target = {
        "boxes": torch.tensor([[15.0, 15.0, 65.0, 65.0]]),
        "labels": torch.tensor([BOTTLE], dtype=torch.int64),
        "masks": torch.stack([mask]),
    }
    return img, target, 0


class TestCopyPasteAugmentor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_prob_zero_returns_same_objects(self):
        aug = CopyPasteAugmentor(prob=0.0)
        img, target = _dst_sample()
        img2, target2 = aug(img, target, _donor_sampler)
        self.assertIs(img2, img)
        self.assertIs(target2, target)

    def test_donor_none_keeps_sample(self):
        aug = CopyPasteAugmentor(prob=1.0, max_pastes=1)
        img, target = _dst_sample()
        img2, target2 = aug(img, target, lambda: None)
        self.assertIs(img2, img)
        self.assertIs(target2, target)

    def test_paste_appends_instance_with_donor_labels(self):
        aug = CopyPasteAugmentor(prob=1.0, max_pastes=1, max_overlap=1.0)
        img, target = _dst_sample()
        img2, target2 = aug(img, target, _donor_sampler)

        self.assertEqual(tuple(target2["boxes"].shape), (2, 4))
        self.assertEqual(int(target2["labels"][-1]), BOTTLE)
        self.assertEqual(tuple(target2["masks"].shape), (2, 100, 120))
        # NEAREST label resize -> pasted channel carries only donor labels.
        pasted = target2["masks"][1]
        self.assertEqual(set(torch.unique(pasted).tolist()), {0, WGRASP})
        # New box is the tight bbox of the pasted pixels.
        ys, xs = torch.nonzero(pasted, as_tuple=True)
        expect = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        self.assertEqual([int(v) for v in target2["boxes"][1]], [int(v) for v in expect])
        # Metadata rows extended consistently.
        for key, n in (("area", 2), ("iscrowd", 2), ("difficult", 2)):
            self.assertEqual(len(target2[key]), n)
        # Image pixels under the pasted alpha come from the donor (value 200).
        img2_np = np.asarray(img2)
        alpha = pasted.numpy() > 0
        self.assertTrue((img2_np[alpha] == 200).all())
        # Untouched keys survive.
        self.assertTrue(torch.equal(target2["orig_size"], target["orig_size"]))

    def test_occlusion_zeroes_existing_masks_under_paste(self):
        aug = CopyPasteAugmentor(prob=1.0, max_pastes=1, max_overlap=1.0)
        img, target = _dst_sample(box=(0, 0, 120, 100))  # existing object everywhere
        img2, target2 = aug(img, target, _donor_sampler)
        self.assertEqual(len(target2["labels"]), 2)
        pasted = target2["masks"][1]
        occluded = target2["masks"][0][pasted > 0]
        self.assertTrue((occluded == 0).all())
        # Outside the paste, the existing mask is untouched.
        outside = target2["masks"][0][pasted == 0]
        self.assertTrue((outside == GRASP).all())

    def test_overlap_rejection_leaves_sample_unchanged(self):
        aug = CopyPasteAugmentor(prob=1.0, max_pastes=1, max_overlap=0.0, max_tries=5)
        img, target = _dst_sample(box=(0, 0, 120, 100))  # any placement overlaps fully
        img2, target2 = aug(img, target, _donor_sampler)
        self.assertIs(img2, img)
        self.assertIs(target2, target)

    def test_multiple_pastes_extend_all_fields(self):
        aug = CopyPasteAugmentor(prob=1.0, max_pastes=3, max_overlap=1.0)
        img, target = _dst_sample()
        _, target2 = aug(img, target, _donor_sampler)
        n = len(target2["labels"])
        self.assertGreaterEqual(n, 2)
        self.assertEqual(tuple(target2["boxes"].shape), (n, 4))
        self.assertEqual(target2["masks"].shape[0], n)
        self.assertEqual(len(target2["area"]), n)

    def test_rejects_invalid_config(self):
        with self.assertRaises(ValueError):
            CopyPasteAugmentor(prob=1.5)
        with self.assertRaises(ValueError):
            CopyPasteAugmentor(max_pastes=0)
        with self.assertRaises(ValueError):
            CopyPasteAugmentor(scale_jitter=(0.0, 1.0))


_XML = """<annotation>
  <filename>{img_id}.jpg</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
  {objects}
</annotation>"""

_OBJ = """<object>
  <name>{name}</name><difficult>{difficult}</difficult>
  <bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox>
</object>"""


class TestBuildDonorIndex(unittest.TestCase):
    def test_positions_mirror_parse_skip_logic(self):
        objects = "".join(
            [
                _OBJ.format(name="hammer", difficult=0, x0=1, y0=1, x1=20, y1=20),
                _OBJ.format(name="bottle", difficult=1, x0=1, y0=1, x1=20, y1=20),  # skipped
                _OBJ.format(name="bottle", difficult=0, x0=30, y0=30, x1=30, y1=30),  # degenerate
                _OBJ.format(name="spatula", difficult=0, x0=40, y0=40, x1=60, y1=60),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "0001.xml")
            with open(path, "w") as f:
                f.write(_XML.format(img_id="0001", objects=objects))
            ET.parse(path)  # sanity: well-formed

            stub = types.SimpleNamespace(
                ids=["0001"],
                annos_path=tmp,
                use_difficult=False,
                object_dict={n: i for i, n in enumerate(IITDetection.object_classes)},
            )
            index = IITDetection._build_donor_index(stub, {9, 10})  # spatula, bottle
            # hammer occupies position 0; both bottles are skipped exactly like
            # parse_voc_xml; spatula lands at position 1.
            self.assertEqual(index, [(0, 1)])

    def test_copy_paste_guarded_to_train_split(self):
        """Regression: yaml_utils.create() merges instance kwargs into the
        GLOBAL class node, so the val dataset can receive copy_paste when the
        train dataloader is built first. The ctor must refuse it off-train
        (this corrupted every in-training eval of the 2026-07-04 A2 run)."""
        with tempfile.TemporaryDirectory() as tmp:
            voc = os.path.join(tmp, "VOCdevkit2012", "VOC2012")
            os.makedirs(os.path.join(voc, "JPEGImages"))
            os.makedirs(os.path.join(voc, "Annotations"))
            os.makedirs(os.path.join(voc, "ImageSets", "Main"))
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(
                os.path.join(voc, "JPEGImages", "0003.jpg")
            )
            objects = _OBJ.format(name="bottle", difficult=0, x0=1, y0=1, x1=20, y1=20)
            with open(os.path.join(voc, "Annotations", "0003.xml"), "w") as f:
                f.write(_XML.format(img_id="0003", objects=objects))
            for split in ("train", "test"):
                with open(os.path.join(voc, "ImageSets", "Main", f"{split}.txt"), "w") as f:
                    f.write("0003\n")

            cp = {"prob": 1.0, "max_pastes": 1}
            train_ds = IITDetection(tmp, image_set="train", copy_paste=dict(cp))
            test_ds = IITDetection(tmp, image_set="test", copy_paste=dict(cp))
            self.assertIsNotNone(train_ds._copy_paste)
            self.assertEqual(len(train_ds._donor_index), 1)
            self.assertIsNone(test_ds._copy_paste)  # guard: never on eval splits
            self.assertEqual(test_ds._donor_index, [])

    def test_donor_class_targeting(self):
        objects = _OBJ.format(name="hammer", difficult=0, x0=1, y0=1, x1=20, y1=20)
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "0002.xml"), "w") as f:
                f.write(_XML.format(img_id="0002", objects=objects))
            stub = types.SimpleNamespace(
                ids=["0002"],
                annos_path=tmp,
                use_difficult=False,
                object_dict={n: i for i, n in enumerate(IITDetection.object_classes)},
            )
            self.assertEqual(IITDetection._build_donor_index(stub, {4}), [(0, 0)])
            self.assertEqual(IITDetection._build_donor_index(stub, {10}), [])


if __name__ == "__main__":
    unittest.main()
