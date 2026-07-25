"""
Test suite for the UMDDetection dataset (UMD Part Affordance Dataset).

Gated on data availability: every test skips cleanly when the extracted
dataset (dataset/umd/part-affordance-dataset/tools) is absent, so the suite
stays green on machines without the ~3 GB download.

Covers (Stage 1, protocol-critical properties):
1. Split lists load: category split = 54 train / 51 test tools (disjoint,
   union = 105); novel split = 76 / 29 with fully held-out test categories.
2. Label -> mask binarization on a real sample: the per-instance mask equals
   the raw ``*_label.mat`` map, and per-affordance binarization
   (mask == k) matches the ranked GT (``*_label_rank.mat`` rank == 1) - the
   AffordanceNet-lineage convention.
3. Class inventory: 7 affordances + background, 17 object categories +
   background, README id order.
4. Boxes valid: XYXY, tight bbox of the non-zero label pixels, inside image.
5. Frame striding (every-3rd-frame human-GT control) and transform path.

Run:  python -m unittest tests.test_umd_dataset -v   (from repo root)
"""

import os
import unittest

import numpy as np
import torch
from src.data import UMDDetection

UMD_ROOT = "./dataset/umd"
DATA_AVAILABLE = os.path.isdir(os.path.join(UMD_ROOT, "part-affordance-dataset", "tools"))

torch.manual_seed(0)


@unittest.skipUnless(DATA_AVAILABLE, "UMD dataset not found under dataset/umd - skipping")
class TestUMDDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = UMDDetection(root=UMD_ROOT, image_set="train", split="category")
        cls.test = UMDDetection(root=UMD_ROOT, image_set="test", split="category")

    # ------------------------------------------------------------------ #
    # splits
    # ------------------------------------------------------------------ #
    def test_category_split_lists_load(self):
        self.assertEqual(len(self.train.tools), 54, "category split: 54 train tools")
        self.assertEqual(len(self.test.tools), 51, "category split: 51 test tools")
        train_set, test_set = set(self.train.tools), set(self.test.tools)
        self.assertEqual(len(train_set & test_set), 0, "train/test tools must be disjoint")
        self.assertEqual(len(train_set | test_set), 105, "105 tools total")

    def test_category_split_is_novel_instance(self):
        """Every category appears on both sides (novel-INSTANCE protocol)."""
        train_cats = {t.rsplit("_", 1)[0] for t in self.train.tools}
        test_cats = {t.rsplit("_", 1)[0] for t in self.test.tools}
        self.assertEqual(train_cats, test_cats, "category split shares all 17 categories")
        self.assertEqual(len(train_cats), 17)

    def test_novel_split_holds_out_categories(self):
        novel_train = UMDDetection(root=UMD_ROOT, image_set="train", split="novel")
        novel_test = UMDDetection(root=UMD_ROOT, image_set="test", split="novel")
        self.assertEqual(len(novel_train.tools) + len(novel_test.tools), 105)
        train_cats = {t.rsplit("_", 1)[0] for t in novel_train.tools}
        test_cats = {t.rsplit("_", 1)[0] for t in novel_test.tools}
        self.assertEqual(len(train_cats & test_cats), 0, "novel split: no shared categories")

    def test_dataset_covers_all_labeled_frames(self):
        """stride 1 indexes every *_label.mat of the split's tools."""
        expected = 0
        for tool in self.train.tools:
            tool_dir = os.path.join(UMD_ROOT, "part-affordance-dataset", "tools", tool)
            expected += sum(1 for f in os.listdir(tool_dir) if f.endswith("_label.mat"))
        self.assertEqual(len(self.train), expected)
        self.assertGreater(len(self.train), 0)

    # ------------------------------------------------------------------ #
    # classes
    # ------------------------------------------------------------------ #
    def test_class_inventory(self):
        self.assertEqual(
            len(UMDDetection.affordance_classes), 8, "7 affordances + background"
        )
        self.assertEqual(
            UMDDetection.affordance_classes,
            ["__background__", "grasp", "cut", "scoop", "contain", "pound", "support", "w-grasp"],
            "affordance ids must match the official README order (gt_label values 1..7)",
        )
        self.assertEqual(len(UMDDetection.object_classes), 18, "17 categories + background")

    # ------------------------------------------------------------------ #
    # sample structure + GT binarization
    # ------------------------------------------------------------------ #
    def _find_index(self, dataset, category):
        for i, (tool, _) in enumerate(dataset.ids):
            if tool.rsplit("_", 1)[0] == category:
                return i
        self.skipTest(f"no {category} sample in split")

    def test_sample_structure_and_boxes_valid(self):
        img, target = self.train[0]
        self.assertEqual((img.width, img.height), (640, 480), "UMD frames are 640x480")
        for key in ("image_id", "boxes", "labels", "area", "iscrowd", "masks",
                    "orig_size", "size"):
            self.assertIn(key, target)
        boxes = target["boxes"]
        self.assertEqual(tuple(boxes.shape), (1, 4), "one whole-tool instance per image")
        x0, y0, x1, y1 = boxes[0].tolist()
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 640)
        self.assertLessEqual(y1, 480)
        self.assertEqual(tuple(target["masks"].shape), (1, 480, 640))
        label_id = int(target["labels"][0])
        self.assertTrue(1 <= label_id <= 17, "object label is a 1..17 category id")

    def test_label_to_mask_binarization_matches_ranked_gt(self):
        """(mask == k) must equal (rank[:, :, k-1] == 1) on a real knife sample."""
        from scipy.io import loadmat

        idx = self._find_index(self.train, "knife")
        _, target = self.train[idx]
        mask = target["masks"][0].numpy()

        tool, frame = self.train.ids[idx]
        base = os.path.join(UMD_ROOT, "part-affordance-dataset", "tools", tool, f"{tool}_{frame}")
        raw = loadmat(base + "_label.mat")["gt_label"]
        rank = loadmat(base + "_label_rank.mat")["gt_label"]

        np.testing.assert_array_equal(mask, raw, "mask must be the raw gt_label map")
        present = set(np.unique(mask).tolist()) - {0}
        self.assertTrue(present, "knife sample should have foreground affordances")
        self.assertLessEqual(present, set(range(1, 8)), "affordance ids in 1..7")
        for k in sorted(present):
            np.testing.assert_array_equal(
                (mask == k),
                (rank[:, :, k - 1] == 1),
                f"binarized affordance {k} must equal rank-1 of the ranked GT",
            )

    def test_box_is_tight_bbox_of_label_foreground(self):
        _, target = self.train[0]
        mask = target["masks"][0].numpy()
        ys, xs = np.nonzero(mask)
        expected = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        self.assertEqual(target["boxes"][0].tolist(), expected)

    def test_object_label_matches_tool_directory(self):
        idx = self._find_index(self.train, "hammer")
        _, target = self.train[idx]
        self.assertEqual(
            int(target["labels"][0]), UMDDetection.object_classes.index("hammer")
        )

    # ------------------------------------------------------------------ #
    # frame striding + transforms
    # ------------------------------------------------------------------ #
    def test_frame_stride_subsamples(self):
        strided = UMDDetection(
            root=UMD_ROOT, image_set="test", split="category", frame_stride=3
        )
        self.assertLess(len(strided), len(self.test))
        # every 3rd frame (+/- per-tool remainder)
        self.assertAlmostEqual(len(strided) / len(self.test), 1.0 / 3.0, delta=0.01)
        frames = [int(f) for t, f in strided.ids if t == strided.ids[0][0]]
        self.assertTrue(
            all(b - a == 3 for a, b in zip(frames, frames[1:])),
            "per-tool frame numbers must step by 3",
        )

    def test_transform_path(self):
        from src.data.transforms import Compose, ConvertDtype, Resize, ToImageTensor

        transforms = Compose(
            [Resize(size=(640, 640)), ToImageTensor(), ConvertDtype(torch.float32)]
        )
        ds = UMDDetection(
            root=UMD_ROOT, image_set="train", split="category", transforms=transforms
        )
        img, target = ds[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(tuple(img.shape), (3, 640, 640))
        self.assertEqual(target["size"].tolist(), [640, 640])
        self.assertEqual(target["masks"].shape[-2:], (640, 640))
        self.assertLessEqual(int(target["masks"].max()), 7)


if __name__ == "__main__":
    unittest.main()
