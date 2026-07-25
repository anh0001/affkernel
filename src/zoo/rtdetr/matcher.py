"""
Enhanced Hungarian Matcher for Joint Object Detection and Affordance Prediction
==============================================================================

This module implements a dual-task Hungarian Matcher that handles both object detection
and affordance segmentation for the IIT-AFF dataset. It computes optimal assignments
between predictions and ground truth targets by considering both object detection metrics and
affordance segmentation quality.

At its core, the matcher is designed to pair each prediction from your model with a corresponding
ground truth object. Here is a straightforward breakdown:
    1. Purpose:
       The matcher is job is to determine the best one-to-one correspondence between the fixed number
       of predictions (e.g., bounding boxes, class scores, and even affordance masks) and the actual
       objects present in an image (the ground truth).
    2. How It Works:
       • Cost Calculation: It computes a “cost” for every possible pairing between a prediction and a
         ground truth object. This cost can include differences in:
           - Classification: How well the predicted class matches the actual class.
           - Box Localization: How far off the predicted bounding box is from the true box.
           - Additional Factors: In this case, it also factors in affordance prediction quality.
       • Optimal Matching: Using the Hungarian algorithm, it finds the pairing that minimizes the overall
         cost. Essentially, it answers the question: “Which prediction should be compared with which ground
         truth to best evaluate the error?”
    3. Why Indices Matter:
       The matcher returns indices—pairs of numbers indicating which predicted output corresponds to which
       ground truth annotation. These indices are critical for the training process as they allow the loss
       functions to accurately match predictions with targets, especially for models that output a fixed
       number of predictions.

Key Features:
------------
1. Dual Class Hierarchies:
   - Object Classes (11 total):
     * Background + 10 object types (bowl, TV/monitor, pan, hammer, knife, etc.)
   - Affordance Classes (10 total):
     * Background + 9 affordance types (contain, cut, display, engine, etc.)

2. Object-Affordance Relationships:
   - Maintains valid affordance mappings for each object class
   - Examples:
     * Bowl -> contain
     * Knife -> cut, grasp
     * Drill -> engine, grasp

3. Cost Components:
   Object Detection:
   - Classification probability (-log p)
   - Bounding box L1 distance
   - Generalized IoU

   Affordance Prediction:
   - Mask IoU for valid affordances
   - Relationship validity penalties

...

by Anhar
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.core import register
from src.zoo.rtdetr.affordance_mask_resizer import RobustAffordanceMaskResizer

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


@register
class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher handling both object detection and affordance prediction.
    For each ground truth box/mask, matches the best predicted box/mask.
    """
    __share__ = ['use_focal_loss', 'use_affordance', 'num_affordance_classes']

    # IIT-AFF object -> valid affordance ids. Only consumed by
    # ``compute_affordance_cost`` (the dense ``affordances_mask`` matcher cost),
    # which the deployed dynamic-kernel recipe never triggers. Applied as the
    # default *only* for the IIT class count so IIT matching stays byte
    # identical; other datasets (e.g. UMD) construct with an empty table.
    _IIT_VALID_AFFORDANCES = {
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
    _IIT_NUM_AFFORDANCE_CLASSES = 10

    def __init__(self, weight_dict, use_focal_loss=False, use_affordance=False,
                 alpha=0.25, gamma=2.0, num_affordance_classes=10,
                 valid_affordances=None):
        """
        Parameters:
            weight_dict: Dict with weights for different loss components
            use_focal_loss: Whether to use focal loss for classification
            use_affordance: Whether to include affordance prediction
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            num_affordance_classes: affordance channels incl. background
                (IIT-AFF default 10 = 9 classes + bg; UMD = 8 = 7 + bg). Flows
                from the top-level config key so IIT YAMLs stay unchanged.
            valid_affordances: optional object-id -> allowed-affordance-id map
                for the dense affordance matcher cost. ``None`` derives the
                legacy IIT table when ``num_affordance_classes`` is the IIT
                count, else an empty table (no object->affordance gating).
        """
        super().__init__()
        self.cost_class = weight_dict.get('cost_class', 1)
        self.cost_bbox = weight_dict.get('cost_bbox', 5)
        self.cost_giou = weight_dict.get('cost_giou', 2)
        self.cost_affordance = weight_dict.get('cost_affordance', 3)

        self.use_focal_loss = use_focal_loss
        self.use_affordance = use_affordance
        self.alpha = alpha
        self.gamma = gamma

        assert any(cost != 0 for cost in [self.cost_class, self.cost_bbox,
                                        self.cost_giou, self.cost_affordance]), "all costs can't be 0"

        # Affordance class count now flows from config (default = IIT 10).
        self.num_affordance_classes = num_affordance_classes

        # Object->affordance validity map for the dense matcher cost. Default
        # to the IIT table only when the class count matches IIT, so IIT
        # matching is byte-identical and non-IIT datasets get no gating.
        if valid_affordances is None:
            if num_affordance_classes == self._IIT_NUM_AFFORDANCE_CLASSES:
                valid_affordances = dict(self._IIT_VALID_AFFORDANCES)
            else:
                valid_affordances = {}
        self.valid_affordances = valid_affordances

        # Initialize mask resizer for predicted masks
        if self.use_affordance:
            self.mask_resizer = RobustAffordanceMaskResizer()


    def compute_affordance_cost(self, pred_affordances, targets, image_sizes, pred_boxes):
        """
        Compute the cost matrix for affordance prediction considering object-affordance relationships.

        Args:
            pred_affordances: [batch_size, num_queries, num_affordance_classes, H, W]
            targets: list of dicts containing ground truth information
            image_sizes: list of tuples [(height, width), ...] for each image in the batch
            pred_boxes: [batch_size, num_queries, 4] - Predicted bounding boxes in xyxy format

        Returns:
            cost_affordance: [batch_size, num_queries, max_num_targets]
        """
        batch_size, num_queries, num_affordance_classes, H, W = pred_affordances.shape
        device = pred_affordances.device

        # Extract object labels from targets
        object_labels = [v["labels"] for v in targets]

        # Calculate the total number of targets across the batch
        num_objects = sum([len(labels) for labels in object_labels])

        # Initialize cost matrix with high costs (e.g., 1.0)
        cost_affordance = torch.ones((batch_size, num_queries, num_objects), device=device)

        target_offset = 0  # To keep track of the target index across the batch

        for b in range(batch_size):
            img_height, img_width = image_sizes[b]
            num_targets = len(object_labels[b])
            for t in range(num_targets):
                obj_label = object_labels[b][t].item()
                valid_affordance_classes = self.valid_affordances.get(obj_label, [])

                if not valid_affordance_classes:
                    # Assign a high cost if there are no valid affordances for this object
                    cost_affordance[b, :, target_offset] = 1.0
                    target_offset += 1
                    continue

                # Get ground truth mask
                gt_mask = targets[b]['masks'][t].to(device)  # [H, W]
                gt_mask = (gt_mask > 0).float()

                # Get ground truth bounding box
                gt_box = targets[b]['boxes'][t].tolist()  # [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = gt_box
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin

                if bbox_width <= 0 or bbox_height <= 0:
                    # print(f"Invalid bbox dimensions for image {b}, target {t}: "
                    #     f"width={bbox_width}, height={bbox_height}")
                    # Assign a high cost for all queries for this target
                    cost_affordance[b, :, target_offset] = 1.0
                    target_offset += 1
                    continue  # Skip further processing for this target

                # Process each query
                for q in range(num_queries):
                    # Extract predicted affordance mask for query q
                    pred_mask = pred_affordances[b, q]  # [num_affordance_classes, H, W]

                    # Extract affordance classes that are valid for this object
                    pred_valid_affordances = pred_mask[valid_affordance_classes]  # [num_valid_affordances, H, W]

                    if pred_valid_affordances.numel() == 0:
                        # No valid affordances predicted
                        cost_affordance[b, q, target_offset] = 1.0
                        continue

                    # Replace F.interpolate resizing with mask resizer
                    resized_masks_list = []
                    for idx in range(pred_valid_affordances.shape[0]):
                        # Use mask resizer: output is a full-size (img_height, img_width) mask.
                        # Ensure the resized mask is on the same device.
                        resized_mask = self.mask_resizer.resize_mask(
                            pred_valid_affordances[idx], gt_box, (img_height, img_width)
                        ).to(device)
                        resized_masks_list.append((resized_mask > 0.5).float())
                    full_pred_mask = torch.stack(resized_masks_list, dim=0)  # [num_valid_affordances, img_height, img_width]

                    # Compute IoU between each affordance class mask and the ground truth mask
                    ious = []
                    for a in range(full_pred_mask.shape[0]):
                        intersection = (full_pred_mask[a] * gt_mask).sum()
                        union = full_pred_mask[a].sum() + gt_mask.sum() - intersection
                        iou = intersection / (union + 1e-6)
                        ious.append(iou)

                    if ious:
                        mean_iou = torch.stack(ious).mean()
                        cost_affordance[b, q, target_offset] = -mean_iou  # Negative because we minimize cost
                    else:
                        cost_affordance[b, q, target_offset] = 1.0  # High cost for invalid affordance

                target_offset += 1

        return cost_affordance

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Forward pass computing the assignment between predictions and targets.

        Args:
            outputs: dict containing:
                'pred_logits': [batch_size, num_queries, num_object_classes]
                'pred_boxes': [batch_size, num_queries, 4]
                'affordances_mask': [batch_size, num_queries, num_affordance_classes, H, W]
            targets: list of dicts containing:
                'labels': [num_target_boxes] - Object class labels
                'boxes': [num_target_boxes, 4]
                'masks': [num_target_boxes, H, W] - Affordance masks
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten predictions
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Concat target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute classification cost
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute box costs
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                       box_cxcywh_to_xyxy(tgt_bbox))

        # Combine all costs except affordance
        C = (self.cost_bbox * cost_bbox +
             self.cost_class * cost_class +
             self.cost_giou * cost_giou)

        # Add affordance cost if enabled
        if self.use_affordance and "affordances_mask" in outputs:
            pred_affordances = outputs["affordances_mask"]  # [batch_size, num_queries, num_affordance_classes, H, W]
            image_sizes = [tuple(t["size"].tolist()) for t in targets]  # [(height, width), ...]

            cost_affordance = self.compute_affordance_cost(pred_affordances, targets, image_sizes, outputs["pred_boxes"])
            cost_affordance = cost_affordance.view(bs * num_queries, -1)

            C += self.cost_affordance * cost_affordance

        # Reshape cost matrix and compute optimal assignment
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def extra_repr(self) -> str:
        """Extra representation string."""
        return (f'Costs: class={self.cost_class}, bbox={self.cost_bbox}, '
                f'giou={self.cost_giou}, affordance={self.cost_affordance}\n'
                f'FocalLoss: {self.use_focal_loss}, Affordance: {self.use_affordance}')
