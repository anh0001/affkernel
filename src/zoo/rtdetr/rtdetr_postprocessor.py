import logging

import torch
import torch.nn as nn
import torchvision

from src.core import register
from src.zoo.rtdetr.affordance_mask_resizer import RobustAffordanceMaskResizer

logger = logging.getLogger(__name__)

__all__ = ['RTDETRPostProcessor']

@register
class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category', 'use_affordance', 'num_affordance_classes']

    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300,
                 remap_mscoco_category=False, use_affordance=False,
                 num_affordance_classes=10) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category
        self.use_affordance = use_affordance
        # Affordance channels incl. background — must equal
        # AffordanceBranch.output_dim. Flows from the top-level config key
        # (default 10 = 9 IIT-AFF classes + bg; UMD = 8 = 7 + bg).
        self.num_affordance_classes = num_affordance_classes
        self.deploy_mode = False

        # Initialize mask resizer if affordance is enabled
        if self.use_affordance:
            self.mask_resizer = RobustAffordanceMaskResizer()

    def extra_repr(self) -> str:
        return (f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, '
                f'num_top_queries={self.num_top_queries}, use_affordance={self.use_affordance}')

    def forward(self, outputs, orig_target_sizes):
        """
        Forward pass for post-processing the model outputs.

        Args:
            outputs: dict containing:
                - pred_logits: classification predictions [batch_size, num_queries, num_classes]
                - pred_boxes: bounding box predictions in cxcywh format [batch_size, num_queries, 4]
                - affordances_mask: affordance predictions [batch_size, num_queries, num_affordance_classes, H, W] (if enabled)
            orig_target_sizes: tensor of original image sizes [batch_size, 2] (width, height)

        Returns:
            List of dictionaries per batch containing:
                - labels: predicted class labels
                - boxes: predicted bounding boxes (xyxy format)
                - scores: confidence scores
                - affordances: pixel-wise affordance predictions [H, W] (if enabled)
        """
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        # --- Log warning if query counts don't match ---
        if logits.shape[1] != boxes.shape[1]:
            logger.warning("Number of queries in pred_logits (%d) does not match pred_boxes (%d)",
                           logits.shape[1], boxes.shape[1])
        # --------------------------

        # Convert boxes from center format (cxcywh) to corner format (xyxy)
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')

        # Scale boxes to the original image size
        # orig_target_sizes is [batch_size, 2] => repeat(1, 2) => [batch_size, 4], then unsqueeze
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        # Detect and fix invalid bounding boxes: width <= 0 or height <= 0
        widths = bbox_pred[..., 2] - bbox_pred[..., 0]
        heights = bbox_pred[..., 3] - bbox_pred[..., 1]
        invalid_mask = (widths <= 0) | (heights <= 0)
        if invalid_mask.any():
            logger.warning(
                "Found invalid bounding boxes with zero or negative size. "
                "You may want to filter or clip them."
            )
            # Fix invalid boxes by ensuring x2 > x1 and y2 > y1 with minimum size of 1 pixel
            bbox_pred[..., 2] = torch.max(bbox_pred[..., 2], bbox_pred[..., 0] + 1.0)
            bbox_pred[..., 3] = torch.max(bbox_pred[..., 3], bbox_pred[..., 1] + 1.0)

        batch_size = logits.shape[0]

        if self.use_focal_loss:
            # -------------------------------------------------------------------
            # 1) Apply sigmoid to logits
            # 2) (New) Filter out queries whose max conf < threshold
            # 3) Then gather top-k among remaining queries
            # -------------------------------------------------------------------
            scores_all = torch.sigmoid(logits)  # [batch_size, num_queries, num_classes]

            # Define a confidence threshold that must be exceeded by *some* class.
            # Overridable via the `confidence_threshold` attribute (default 0.6,
            # byte-identical when unset) so the eval operating point can be swept.
            confidence_threshold = getattr(self, "confidence_threshold", 0.6)

            # We will:
            #  (a) flatten each sample's scores to [num_queries * num_classes]
            #  (b) do top-k as before
            #  (c) zero out any that remain below the threshold
            #
            # => If all queries have max confidence below threshold, they become all zeros.
            scores_flat = scores_all.flatten(1)  # [batch_size, num_queries * num_classes]

            # Top-k over the flattened dimension
            # shape = [batch_size, num_top_queries]
            scores, index = torch.topk(scores_flat, self.num_top_queries, dim=-1)

            labels = index % self.num_classes  # [batch_size, num_top_queries]
            query_indices = index // self.num_classes  # [batch_size, num_top_queries]

            # Gather boxes
            boxes = bbox_pred.gather(dim=1, index=query_indices.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

            # -------------------------------------------------------------------
            # ** Apply threshold **
            # We'll interpret 'scores' as the top class conf for each top-k item,
            # so if that conf < threshold => set it to zero.
            # You can also do a per-class check, but here we assume:
            #   "scores" is the chosen top-k confidence
            #
            # If 'scores' < threshold => zero out both label & score
            # -------------------------------------------------------------------
            mask = scores >= confidence_threshold
            # For each batch, zero out those below threshold
            for b in range(batch_size):
                if not mask[b].any():
                    # If absolutely none pass threshold => all zero
                    scores[b] = 0.0
                    labels[b] = 0
                    boxes[b] = 0.0
                else:
                    # Zero out below-threshold indexes
                    scores[b, ~mask[b]] = 0.0
                    labels[b, ~mask[b]] = 0
                    boxes[b, ~mask[b]] = 0.0

        else:
            # Apply softmax to logits for standard loss
            # Exclude the last class if it is background
            scores_all = torch.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores_all.max(dim=-1)  # [batch_size, num_queries]

            # If the number of scores exceeds top queries, do top-k
            if scores.shape[1] > self.num_top_queries:
                scores, index_topk = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index_topk)
                boxes = torch.gather(boxes, dim=1,
                                     index=index_topk.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
                query_indices = index_topk
            else:
                # No need to reduce
                query_indices = torch.arange(scores.shape[1], device=logits.device)
                query_indices = query_indices.unsqueeze(0).repeat(batch_size, 1)

        # -------------------------------------------------------------------
        # Process affordances if enabled
        # -------------------------------------------------------------------
        resized_affordances = [None] * logits.shape[0]  # Initialize with batch size
        if self.use_affordance and 'aff_kernel' in outputs:
            from src.zoo.rtdetr.affordance import (
                decode_affordance_masks,
                decode_affordance_masks_embedding,
            )

            aff_feat = outputs['aff_feat']       # [B, Cf, Hf, Wf]
            kernels_all = outputs['aff_kernel']  # [B, Q, n_params]
            top_k = query_indices.shape[1]
            out_dim = self.num_affordance_classes

            # Flatten the (batch, top-k query) selection and decode masks ONLY
            # for those queries — never a dense [Q, C, H, W] tensor.
            batch_index = (
                torch.arange(batch_size, device=kernels_all.device)
                .unsqueeze(1)
                .repeat(1, top_k)
                .flatten()
            )
            flat_q = query_indices.flatten()
            sel_kernels = kernels_all[batch_index, flat_q]  # [B*top_k, n_params]
            # Dispatch on head type — 'dynamic' (default, dynamic-conv kernels)
            # vs 'embedding' (MaskFormer-style A5 ablation).
            head_type = outputs.get('aff_head_type', 'dynamic')
            decode_fn = (
                decode_affordance_masks_embedding if head_type == 'embedding'
                else decode_affordance_masks
            )
            # Box-relative geometry channels (dynamic head only): gather the
            # selected queries' NORMALISED cxcywh boxes. `boxes` has already been
            # converted to scaled xyxy above, so read from the raw pred tensor.
            sel_boxes = None
            if outputs.get('aff_box_relative', False):
                sel_boxes = outputs['pred_boxes'][batch_index, flat_q]
            decoded = decode_fn(
                aff_feat=aff_feat,
                kernels=sel_kernels,
                batch_index=batch_index,
                output_dim=out_dim,
                boxes=sel_boxes,
            )  # [B*top_k, out_dim, Hf, Wf]
            # Keep SOFT probabilities — argmax at the coarse Hf x Wf grid and
            # then nearest-upsampling hard labels destroys boundaries, which a
            # boundary-weighted metric (F_beta^w) punishes hard. Instead
            # bilinearly upsample the probability maps to full resolution and
            # argmax there.
            probs = decoded.softmax(dim=1)  # [B*top_k, C, Hf, Wf]
            probs = probs.view(batch_size, top_k, out_dim, *probs.shape[-2:])

            # Only export masks for queries that pass the detection-score gate.
            # Decoding/upsampling every top-k query and unioning them in the
            # evaluator turns weak/duplicate detections into spurious
            # foreground (score-blind union). Non-retained queries get an
            # all-background mask so per-query alignment with scores/labels in
            # the evaluator is preserved.
            score_thr = getattr(self, 'affordance_score_thresh', 0.5)
            keep_all = scores >= score_thr  # [B, top_k]

            for b in range(batch_size):
                w, h = int(orig_target_sizes[b][0]), int(orig_target_sizes[b][1])
                masks = torch.zeros(
                    (top_k, h, w), dtype=torch.long, device=probs.device
                )
                keep = keep_all[b].nonzero(as_tuple=True)[0]
                if keep.numel() > 0:
                    up = torch.nn.functional.interpolate(
                        probs[b, keep], size=(h, w),
                        mode='bilinear', align_corners=False,
                    )  # [n_keep, C, H, W]
                    masks[keep] = up.argmax(dim=1).long()
                resized_affordances[b] = masks

        # Handle deploy mode - return single tensors
        if self.deploy_mode:
            if self.use_affordance and 'aff_kernel' in outputs:
                # Stack along batch dimension to [B, num_top_queries, H, W]
                affordances_4d = torch.stack(resized_affordances, dim=0)
                return labels, boxes, scores, affordances_4d
            return labels, boxes, scores

        # Optionally remap MSCOCO label IDs
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            # Flatten, map, then reshape
            labels_flat = labels.flatten()
            mapped = [mscoco_label2category[int(x.item())] for x in labels_flat]
            labels = torch.tensor(mapped, device=boxes.device).reshape(labels.shape)

        # Build final list-of-dicts output
        results = []
        for b in range(batch_size):
            result_dict = {
                'labels': labels[b],
                'boxes': boxes[b],
                'scores': scores[b]
            }
            if self.use_affordance and resized_affordances[b] is not None:
                result_dict['affordances'] = resized_affordances[b]
            results.append(result_dict)

        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self

    @property
    def iou_types(self):
        return ('bbox', ) if not self.use_affordance else ('bbox', 'affordance')
