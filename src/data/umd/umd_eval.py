"""UMD Part-Affordance evaluator (Myers, Teo, Fermuller & Aloimonos, ICRA 2015).

Mirrors ``src/data/iit/iit_eval.py``'s ``IITEvaluator`` public interface — the
exact surface ``det_engine``/``det_solver`` consume (``update``,
``synchronize_between_processes``, ``accumulate``, ``summarize``, ``iou_types``,
``stats``, plus ``object_classes``/``iou_thresh``/``use_07_metric`` for
checkpointing) — by subclassing it, so the shared prediction bookkeeping, bbox
AP and affordance-IoU AP logic are reused verbatim.

Two UMD-specific differences:
  1. Ground truth is sourced from the UMD ``*_label.mat`` single-label maps at
     their native 640x480 resolution (boxes = tight bbox of the non-zero tool
     pixels; masks = ``label == affordance_id``), not from VOC XML / ``.sm``
     files. Predictions arrive already resized to the same native resolution by
     the postprocessor, so shapes align.
  2. The weighted F-measure is reported with **beta^2 = 1 as the primary
     convention** (UMD/Myers 2015 states beta = 1 explicitly; AffordanceNet's
     ``evaluate_Fwb_non_rank`` calls ``WFb.m`` with its default ``Beta2 = 1``),
     while beta^2 = 0.3 (the saliency / IIT-AFF default) is computed alongside
     for parity with our IIT tables.

The F_beta^w scorer itself (``weighted_fbeta_measure``) is imported and reused —
never re-implemented. Macro aggregation matches the IIT/AffordanceNet lineage:
per-class mean over the images where that GT class is present, then mean over
classes.
"""

from collections import defaultdict

import numpy as np
from scipy.io import loadmat

from src.data.iit.iit_eval import IITEvaluator, weighted_fbeta_measure

__all__ = ["UMDEvaluator"]


class UMDEvaluator(IITEvaluator):
    """Affordance/detection evaluator for the UMD Part-Affordance dataset."""

    # F_beta^w conventions to report: (stats bucket, beta^2). beta^2 = 1 is the
    # UMD/AffordanceNet primary; 0.3 is the IIT-AFF saliency default (aux).
    FBW_BETAS = (("affordance_fbw_b1", 1.0), ("affordance_fbw_b03", 0.3))

    def __init__(self, dataset, iou_thresh=0.5, use_07_metric=False, use_affordance=True):
        super().__init__(
            dataset,
            iou_thresh=iou_thresh,
            use_07_metric=use_07_metric,
            use_affordance=use_affordance,
        )
        # UMD has no object -> valid-affordance gating table (the dense-cost
        # path that consumed the IIT table is never triggered here).
        self.valid_affordances = {}

    def reset(self):
        """Reset state, adding the two F_beta^w buckets (beta^2 = 1 and 0.3)."""
        self.image_ids = []
        self.bbox_predictions = defaultdict(list)
        self.mask_predictions = defaultdict(list)
        self._saw_affordances = False
        # Per-eval cache of native-resolution label maps, keyed by dataset
        # index. Every affordance/bbox class sweep re-reads GT for all images;
        # caching the raw label map loads each ``*_label.mat`` exactly once
        # instead of (num_classes x num_images) times. The evaluator is
        # recreated each epoch, so the cache is transient and freed afterwards.
        self._label_cache = {}
        self.stats = {
            "bbox": defaultdict(list),
            "affordance": defaultdict(list) if self.use_affordance else None,
        }
        if self.use_affordance:
            for bucket, _ in self.FBW_BETAS:
                self.stats[bucket] = defaultdict(list)

    # ------------------------------------------------------------------ #
    # ground truth sourcing (UMD label maps, native 640x480)
    # ------------------------------------------------------------------ #
    def _file_id(self, img_id):
        """UMD records image_id == dataset index directly (no stem mapping)."""
        return img_id

    def _load_label(self, img_id):
        """Native-resolution single-label affordance map for a dataset index."""
        index = int(img_id)
        cached = self._label_cache.get(index)
        if cached is None:
            path = self.dataset._frame_path(index, "label.mat")
            cached = np.asarray(loadmat(path)["gt_label"], dtype=np.uint8)
            self._label_cache[index] = cached
        return cached

    def load_ground_truth(self, cls):
        """Tight bbox of the tool's non-zero label pixels, per image of ``cls``."""
        gt_boxes = []
        for img_id in self.image_ids:
            index = int(img_id)
            tool, _ = self.dataset.ids[index]
            if self.dataset._tool_category(tool) != cls:
                continue
            label = self._load_label(img_id)
            ys, xs = np.nonzero(label)
            if len(xs) == 0:
                continue
            box = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ]
            gt_boxes.append({"bbox": box, "image_id": img_id})
        return gt_boxes

    def load_ground_truth_masks(self, aff_cls):
        """Binary GT mask (``label == aff_idx``) per image where it is present."""
        gt_masks = []
        aff_idx = self.dataset.affordance_dict[aff_cls]
        for img_id in self.image_ids:
            label = self._load_label(img_id)
            binary_mask = (label == aff_idx).astype(np.uint8)
            if binary_mask.sum() > 0:
                gt_masks.append({"mask": binary_mask, "image_id": img_id})
        return gt_masks

    # ------------------------------------------------------------------ #
    # summary (both betas; beta^2 = 1 is the headline)
    # ------------------------------------------------------------------ #
    def summarize(self):
        """Summarize bbox AP, affordance IoU AP, and F_beta^w for both betas."""
        print("Summarizing UMD results...")

        if "bbox" in self.iou_types:
            print("Evaluating Bounding Boxes...")
            self.evaluate_bboxes()
            if self.stats["bbox"]["AP"]:
                mean_bbox = np.mean(list(self.stats["bbox"]["AP"]))
                print(f"mAP for Bounding Boxes: {mean_bbox:.4f}")
            else:
                print("No Bounding Box AP to report.")

        if "affordance" in self.iou_types and self.use_affordance and self._saw_affordances:
            print("Evaluating Affordance Masks (IoU AP)...")
            self.evaluate_affordances()
            if self.stats["affordance"]["AP"]:
                mean_aff = np.mean(list(self.stats["affordance"]["AP"]))
                print(f"mAP for Affordance Masks: {mean_aff:.4f}")
            else:
                print("No Affordance Mask AP to report.")

            for bucket, beta2 in self.FBW_BETAS:
                tag = "PRIMARY" if beta2 == 1.0 else "aux"
                print(f"Evaluating Affordance F_beta^w (beta^2={beta2}, {tag})...")
                self._evaluate_fbw(beta2, bucket)
                fbw = list(self.stats[bucket].get("Fbw", []))
                if fbw:
                    print(
                        f"mean F_beta^w (beta^2={beta2}, {tag}): {np.mean(fbw):.4f}"
                    )
                else:
                    print(f"No F_beta^w (beta^2={beta2}) to report.")

    def _evaluate_fbw(self, beta2, bucket):
        """Weighted F-measure per affordance class into ``self.stats[bucket]``.

        Union GT (and predicted) binary masks per image for the class, score
        each GT-present image with the reused ``weighted_fbeta_measure``, then
        average over images (per class). A class with GT but no prediction on an
        image scores against an all-background map (i.e. counts as 0), never
        skipped — mirroring IITEvaluator so an empty model cannot inflate.
        """
        for cls in self.affordance_classes:
            if cls == "__background__":
                continue
            gt_masks = self.load_ground_truth_masks(cls)
            if not gt_masks:
                continue

            gt_by_img = defaultdict(lambda: None)
            for idx, g in enumerate(gt_masks):
                if not isinstance(g, dict) or "mask" not in g:
                    continue
                m = np.asarray(g["mask"]).astype(bool)
                iid = g.get("image_id", f"_gt{idx}")
                gt_by_img[iid] = m if gt_by_img[iid] is None else (gt_by_img[iid] | m)
            if not gt_by_img:
                continue

            pred_by_img = {}
            for idx, p in enumerate(self.mask_predictions.get(cls, [])):
                if not isinstance(p, dict) or "mask" not in p:
                    continue
                m = np.asarray(p["mask"]).astype(bool)
                iid = p.get("image_id", f"_pred{idx}")
                pred_by_img[iid] = m if iid not in pred_by_img else (pred_by_img[iid] | m)

            scores = []
            for iid, gt_m in gt_by_img.items():
                pred_m = pred_by_img.get(iid)
                if pred_m is None or pred_m.shape != gt_m.shape:
                    pred_fg = np.zeros_like(gt_m, dtype=np.float64)
                else:
                    pred_fg = pred_m.astype(np.float64)
                q = weighted_fbeta_measure(pred_fg, gt_m, beta2=beta2)
                if q is not None:
                    scores.append(q)

            if scores:
                cls_fbw = float(np.mean(scores))
                self.stats[bucket]["Fbw"].append(cls_fbw)
                print(f"F_beta^w (beta^2={beta2}) for Affordance {cls}: {cls_fbw:.4f}")
