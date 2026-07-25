import os
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch

from src.misc import dist

__all__ = ['IITEvaluator', 'weighted_fbeta_measure']

def parse_rec(filename):
    """Parse a PASCAL VOC XML file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text.lower().strip()
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(float(bbox.find('xmin').text)),
            int(float(bbox.find('ymin').text)),
            int(float(bbox.find('xmax').text)),
            int(float(bbox.find('ymax').text))
        ]
        objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall."""
    if use_07_metric:
        # 11-point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.0
    else:
        # Correct AP calculation
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

class IITEvaluator:

    __share__ = ['use_affordance']

    def __init__(self, dataset, iou_thresh=0.5, use_07_metric=False, use_affordance=True):
        """
        Initialize the IITEvaluator.

        Args:
            dataset: Reference to the IIT dataset.
            iou_thresh: IoU threshold for matching predictions to ground truth.
            use_07_metric: Whether to use VOC2007 11-point AP metric.
            use_affordance: Whether to evaluate affordance masks in addition to bounding boxes.
        """
        self.dataset = dataset
        self.iou_thresh = iou_thresh
        self.use_07_metric = use_07_metric
        self.use_affordance = use_affordance
        self.object_classes = dataset.object_classes
        self.affordance_classes = dataset.affordance_classes  # Added for affordance handling

        # Define mapping from object classes to valid affordance classes
        # This should mirror the mapping in matcher.py
        self.valid_affordances = {
            1: [1],          # bowl: contain
            2: [3],          # TV/monitor: display
            3: [1, 5],       # pan: contain, grasp
            4: [5, 7],       # hammer: grasp, pound
            5: [2, 5],       # knife: cut, grasp
            6: [1, 5],       # cup: contain, grasp
            7: [4, 5],       # drill: engine, grasp
            8: [5, 6],       # racket: grasp, hit
            9: [5, 8],       # spatula: grasp, support
            10: [1, 9]       # bottle: contain, wrap-grasp
        }

        # Validate IoU types
        self.iou_types = ['bbox']
        if self.use_affordance:
            self.iou_types.append('affordance')

        self.reset()

    def reset(self):
        """Reset the evaluator state."""
        self.image_ids = []
        self.bbox_predictions = defaultdict(list)
        self.mask_predictions = defaultdict(list)
        # True once any prediction carried an 'affordances' key — distinguishes
        # "model produced no affordance output at all" (skip aff reporting)
        # from "produced output but empty for some classes" (report 0 per
        # class with GT, so the mean isn't silently inflated).
        self._saw_affordances = False
        # Initialize stats for each IoU type
        self.stats = {
            'bbox': defaultdict(list),
            'affordance': defaultdict(list) if self.use_affordance else None,
            # F_beta^w — the IIT-AFF/AffordanceNet standard affordance metric.
            'affordance_fbw': defaultdict(list) if self.use_affordance else None,
        }

    def update(self, predictions):
        """
        Update evaluator with new predictions.
        Predictions are assumed to have affordance masks already resized to image size.
        """
        for image_id, prediction in predictions.items():
            self.image_ids.append(image_id)

            scores = prediction.get('scores', torch.empty((0,)))
            labels = prediction.get('labels', torch.empty((0,), dtype=torch.int64))
            boxes = prediction.get('boxes', torch.empty((0, 4)))

            # Object Detection Predictions
            if 'bbox' in self.iou_types:
                for box, score, label in zip(boxes, scores, labels):
                    # Skip invalid boxes
                    xmin, ymin, xmax, ymax = box.tolist()
                    if (xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0 or  # negative coordinates
                        xmax <= xmin or ymax <= ymin):                   # invalid dimensions
                        continue

                    self.bbox_predictions[self.object_classes[label]].append({
                        'image_id': image_id,
                        'score': score.item(),
                        'bbox': [xmin, ymin, xmax, ymax]
                    })

            # Affordance Segmentation Predictions
            if self.use_affordance and 'affordances' in prediction:
                self._saw_affordances = True
                affordances = prediction['affordances']  # [num_queries, H, W]

                for score, _label, mask in zip(scores, labels, affordances):
                    # Trust the multiclass affordance head's per-pixel class
                    # directly. Gating by object_label -> valid_affordances
                    # discards correct predictions whenever the affordance blob
                    # is emitted by a query the detector labelled as a
                    # different object (e.g. w-grasp was predicted but NEVER on
                    # bottle-labelled queries -> forced F_beta^w = 0). The
                    # postprocessor already only exports masks for queries that
                    # pass the detection-score gate.
                    present = torch.unique(mask)
                    for aff_class_idx in present.tolist():
                        aff_class_idx = int(aff_class_idx)
                        if aff_class_idx <= 0 or aff_class_idx >= len(self.affordance_classes):
                            continue  # skip background / out-of-range
                        affordance_class = self.affordance_classes[aff_class_idx]
                        mask_binary = (mask == aff_class_idx).cpu().numpy().astype(np.uint8)
                        self.mask_predictions[affordance_class].append({
                            'image_id': image_id,
                            'score': score.item(),
                            'mask': mask_binary  # Already resized in postprocessor
                        })

    def synchronize_between_processes(self):
        """
        Synchronize predictions from all processes (for distributed evaluation).
        """
        if not dist.is_dist_available_and_initialized():
            return

        # Gather bbox predictions
        all_bbox_predictions = dist.all_gather_object(self.bbox_predictions)
        merged_bbox_predictions = defaultdict(list)
        for preds in all_bbox_predictions:
            for cls, items in preds.items():
                merged_bbox_predictions[cls].extend(items)
        self.bbox_predictions = merged_bbox_predictions

        # Gather mask predictions
        if self.use_affordance:
            all_mask_predictions = dist.all_gather_object(self.mask_predictions)
            merged_mask_predictions = defaultdict(list)
            for preds in all_mask_predictions:
                for cls, items in preds.items():
                    merged_mask_predictions[cls].extend(items)
            self.mask_predictions = merged_mask_predictions

        # Gather image IDs
        all_image_ids = dist.all_gather_object(self.image_ids)
        self.image_ids = list(set([img_id for sublist in all_image_ids for img_id in sublist]))

    def accumulate(self):
        """
        Accumulate evaluation results from all images.
        """
        # Placeholder for potential future accumulation steps
        pass

    def summarize(self):
        """
        Summarize the evaluation results by calculating AP for each class and IoU type.
        """
        print('Summarizing results...')

        if 'bbox' in self.iou_types:
            print("Evaluating Bounding Boxes...")
            self.evaluate_bboxes()  # populates self.stats['bbox']

        # Always evaluate affordances when enabled. evaluate_affordances()
        # already assigns AP 0.0 per class that has GT but no predictions —
        # skipping it on aff_pred_count==0 would silently drop those classes
        # and inflate the reported mean (a model predicting nothing must score
        # 0, not be excused).
        if 'affordance' in self.iou_types and self.use_affordance \
                and self._saw_affordances:
            print("Evaluating Affordance Masks...")
            self.evaluate_affordances()  # populates self.stats['affordance']

        # Compute mAP for each IoU type
        if 'bbox' in self.iou_types:
            if self.stats['bbox']['AP']:
                mAP_bbox = np.mean(list(self.stats['bbox']['AP']))
                print(f"mAP for Bounding Boxes: {mAP_bbox:.4f}")
            else:
                print("No Bounding Box AP to report.")

        if 'affordance' in self.iou_types and self.use_affordance \
                and self._saw_affordances:
            if self.stats['affordance']['AP']:
                mAP_affordance = np.mean(list(self.stats['affordance']['AP']))
                print(f"mAP for Affordance Masks: {mAP_affordance:.4f}")
            else:
                print("No Affordance Mask AP to report.")

            # Weighted F-measure (the metric comparable to AffordanceNet).
            print("Evaluating Affordance F_beta^w...")
            self.evaluate_affordances_fbw()
            fbw = list(self.stats['affordance_fbw'].get('Fbw', []))
            if fbw:
                print(f"mean F_beta^w for Affordance Masks: {np.mean(fbw):.4f}")
            else:
                print("No Affordance F_beta^w to report.")

    def evaluate_bboxes(self):
        """
        Evaluate bounding boxes using PASCAL VOC metrics.

        Returns:
            List of AP values per class.
        """
        ap_per_class = []
        for cls in self.object_classes:
            if cls == '__background__':
                continue  # Skip background class
            preds = self.bbox_predictions.get(cls, [])

            # Load ground truth annotations
            gt_annotations = self.load_ground_truth(cls)
            if not gt_annotations:
                continue  # Skip if no ground truth for the class

            if not preds:
                # No predictions for this class but ground truth exists
                ap = 0.0
                self.stats['bbox']['AP'].append(ap)
                ap_per_class.append(ap)
                print(f"AP for {cls}: {ap:.4f}")
                continue  # Move to the next class

            # Sort predictions by score descending
            preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
            num_preds = len(preds_sorted)
            num_gts = len(gt_annotations)

            if num_gts == 0:
                ap = 0.0
                ap_per_class.append(ap)
                continue

            # Initialize True Positive and False Positive
            tp = np.zeros(num_preds)
            fp = np.zeros(num_preds)
            matched_gts = set()

            for i, pred in enumerate(preds_sorted):
                gt_boxes = gt_annotations
                pred_box = pred['bbox']
                ious = [box_iou_np(pred_box, gt['bbox']) for gt in gt_boxes]
                if len(ious) == 0:
                    fp[i] = 1
                    continue
                max_iou = max(ious)
                max_idx = ious.index(max_iou)
                if max_iou >= self.iou_thresh and max_idx not in matched_gts:
                    tp[i] = 1
                    matched_gts.add(max_idx)
                else:
                    fp[i] = 1

            # Compute precision and recall
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / num_gts
            prec = tp_cum / (tp_cum + fp_cum + 1e-6)
            ap = voc_ap(rec, prec, self.use_07_metric)
            self.stats['bbox']['AP'].append(ap)
            ap_per_class.append(ap)
            print(f"AP for {cls}: {ap:.4f}")

        return ap_per_class

    def evaluate_affordances(self):
        """
        Evaluate affordance masks using IoU metrics.

        Returns:
            List of AP values per affordance class.
        """
        ap_per_class = []
        for cls in self.affordance_classes:
            if cls == '__background__':
                continue  # Skip background class
            preds = self.mask_predictions.get(cls, [])

            # Load ground truth masks
            gt_masks = self.load_ground_truth_masks(cls)
            if not gt_masks:
                continue  # Skip if no ground truth for the class

            if not preds:
                # No predictions for this class but ground truth exists
                ap = 0.0
                self.stats['affordance']['AP'].append(ap)
                ap_per_class.append(ap)
                print(f"AP for Affordance {cls}: {ap:.4f}")
                continue  # Move to the next class

            # Group GT by image_id — detection AP matches a prediction only
            # against GT of the SAME image. Comparing every prediction to
            # every GT globally is both wrong (cross-image matches) and
            # O(N_pred * N_gt) over full-res masks (eval effectively hangs).
            # GT entries from load_ground_truth_masks carry 'image_id'. Any
            # without one go in a shared bucket every prediction also checks
            # (preserves the old global behavior when image_id is absent,
            # e.g. mocked unit tests).
            _NOIMG = '__noimg__'
            gt_by_img = defaultdict(list)
            for g in gt_masks:
                gt_by_img[g.get('image_id', _NOIMG)].append(g['mask'])

            # Sort predictions by score descending
            preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
            num_preds = len(preds_sorted)
            num_gts = len(gt_masks)

            # Initialize True Positive and False Positive
            tp = np.zeros(num_preds)
            fp = np.zeros(num_preds)
            matched_gts = set()  # keyed by (image_id, local gt index)

            for i, pred in enumerate(preds_sorted):
                pred_mask = pred['mask']
                img_id = pred.get('image_id', _NOIMG)
                # Same-image GT plus any image-id-less GT (test fallback).
                candidates = [(img_id, m) for m in gt_by_img.get(img_id, [])]
                if img_id != _NOIMG:
                    candidates += [(_NOIMG, m) for m in gt_by_img.get(_NOIMG, [])]
                best_iou, best_key = 0.0, None
                for j, (bucket, gt_mask) in enumerate(candidates):
                    if pred_mask.shape != gt_mask.shape:
                        continue
                    iou = mask_iou_np(pred_mask, gt_mask)
                    if iou > best_iou:
                        best_iou, best_key = iou, (bucket, j)
                key = best_key
                if best_iou >= self.iou_thresh and key is not None and key not in matched_gts:
                    tp[i] = 1
                    matched_gts.add(key)
                else:
                    fp[i] = 1

            # Compute precision and recall
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / num_gts
            prec = tp_cum / (tp_cum + fp_cum + 1e-6)
            ap = voc_ap(rec, prec, self.use_07_metric)
            self.stats['affordance']['AP'].append(ap)
            ap_per_class.append(ap)
            print(f"AP for Affordance {cls}: {ap:.4f}")

        return ap_per_class

    def evaluate_affordances_fbw(self, beta2=0.3):
        """Weighted F-measure F_beta^w per affordance class, aggregated over
        the test images (mean over images where the GT class is present),
        then averaged across classes. This is the AffordanceNet/IIT-AFF
        comparable metric.

        Returns:
            List of per-class F_beta^w values.
        """
        fbw_per_class = []
        for cls in self.affordance_classes:
            if cls == '__background__':
                continue
            gt_masks = self.load_ground_truth_masks(cls)
            if not gt_masks:
                continue

            # Union GT (and predicted) binary masks per image for this class.
            gt_by_img = defaultdict(lambda: None)
            for idx, g in enumerate(gt_masks):
                if not isinstance(g, dict) or 'mask' not in g:
                    continue  # skip malformed GT entries
                m = np.asarray(g['mask']).astype(bool)
                iid = g.get('image_id', f'_gt{idx}')
                gt_by_img[iid] = m if gt_by_img[iid] is None else (gt_by_img[iid] | m)
            if not gt_by_img:
                continue

            pred_by_img = {}
            for idx, p in enumerate(self.mask_predictions.get(cls, [])):
                if not isinstance(p, dict) or 'mask' not in p:
                    continue
                m = np.asarray(p['mask']).astype(bool)
                iid = p.get('image_id', f'_pred{idx}')
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
                self.stats['affordance_fbw']['Fbw'].append(cls_fbw)
                fbw_per_class.append(cls_fbw)
                print(f"F_beta^w for Affordance {cls}: {cls_fbw:.4f}")
        return fbw_per_class

    def _file_id(self, img_id):
        """Map a recorded image_id to the original VOC stem used in GT
        filenames. det_engine keys predictions by ``target['image_id'].item()``
        = the dataset *index*, but Annotations/`.sm` files are named by the
        original string id (e.g. index 0 -> id "5"). IIT-AFF files are
        numerically named, so the raw index silently resolves to a *valid but
        wrong* image's GT — corrupting both bbox and affordance eval. Translate
        index -> dataset.ids[index]; pass through anything already a stem.
        """
        ids = getattr(self.dataset, 'ids', None)
        if ids is not None and isinstance(img_id, (int, np.integer)) \
                and 0 <= int(img_id) < len(ids):
            return ids[int(img_id)]
        return img_id

    def load_ground_truth(self, cls):
        """
        Load ground truth bounding boxes for a specific class.

        Args:
            cls (str): Class name.

        Returns:
            List of ground truth bounding boxes.
        """
        gt_boxes = []
        for img_id in self.image_ids:
            anno_path = os.path.join(
                self.dataset.annos_path, f"{self._file_id(img_id)}.xml"
            )
            objects = parse_rec(anno_path)
            for obj in objects:
                if obj['name'] == cls:
                    # Return dicts to match evaluate_bboxes (gt['bbox']).
                    gt_boxes.append({'bbox': obj['bbox'], 'image_id': img_id})
        return gt_boxes

    def load_ground_truth_masks(self, aff_cls):
        """
        Load ground truth affordance masks for a specific affordance class.

        Args:
            aff_cls (str): Affordance class name.

        Returns:
            List of dictionaries containing ground truth masks and metadata.
        """
        gt_masks = []
        aff_idx = self.dataset.affordance_dict[aff_cls]

        for img_id in self.image_ids:
            file_id = self._file_id(img_id)
            mask_count = 1
            while True:
                mask_path = os.path.join(self.dataset.mask_cache_path, f"{file_id}_{mask_count}_segmask.sm")
                if not os.path.exists(mask_path):
                    break
                try:
                    with open(mask_path, 'rb') as f:
                        mask = pickle.load(f)
                    # Convert to binary mask for the specific affordance class
                    binary_mask = (mask == aff_idx).astype(np.uint8)
                    if binary_mask.sum() > 0:  # Only add if mask contains the affordance
                        gt_masks.append({
                            'mask': binary_mask,
                            'image_id': img_id  # Ensure image_id is included
                        })
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
                mask_count += 1
        return gt_masks

def box_iou_np(box1, box2):
    """Compute IoU between two boxes in [xmin, ymin, xmax, ymax] format."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou

def weighted_fbeta_measure(fg, gt, beta2=0.3):
    """Weighted F-beta measure F_beta^w (Margolin, Zelnik-Manor & Tal, CVPR 2014,
    "How to Evaluate Foreground Maps?"). This is the standard IIT-AFF /
    AffordanceNet affordance-segmentation metric — NOT detection AP.

    Faithful port of the reference ``WFb.m``.

    Args:
        fg: predicted foreground map, float array in [0, 1], shape [H, W].
        gt: ground-truth binary mask, shape [H, W].
        beta2: beta^2 (AffordanceNet/IIT-AFF convention: 0.3).

    Returns:
        Scalar F_beta^w in [0, 1], or ``None`` if ``gt`` has no positive
        pixels (that (image, class) pair is not scored).
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    gt = np.asarray(gt).astype(bool)
    if gt.sum() == 0:
        return None
    fg = np.asarray(fg).astype(np.float64)
    fg = np.clip(fg, 0.0, 1.0)
    dgt = gt.astype(np.float64)

    e = np.abs(fg - dgt)
    # Distance to nearest GT pixel + index of that pixel (MATLAB bwdist(GT)).
    dst, idxt = distance_transform_edt(~gt, return_indices=True)

    # Pixel dependency: propagate the error of the nearest GT pixel into bg.
    et = e.copy()
    et[~gt] = e[idxt[0][~gt], idxt[1][~gt]]
    ea = gaussian_filter(et, sigma=5.0, truncate=(7 - 1) / 2 / 5.0)
    min_e_ea = e.copy()
    cond = gt & (ea < e)
    min_e_ea[cond] = ea[cond]

    # Pixel importance.
    b = np.ones_like(dgt)
    b[~gt] = 2.0 - np.exp(np.log(1 - 0.5) / 5.0 * dst[~gt])
    ew = min_e_ea * b

    tpw = dgt.sum() - ew[gt].sum()
    fpw = ew[~gt].sum()
    eps = np.finfo(np.float64).eps
    r = 1.0 - ew[gt].mean()                       # weighted recall
    p = tpw / (eps + tpw + fpw)                   # weighted precision
    q = (1 + beta2) * (r * p) / (eps + r + beta2 * p)
    return float(q)


def mask_iou_np(mask1, mask2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / (union + 1e-6)
    return iou
