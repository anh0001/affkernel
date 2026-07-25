"""
AfforMeNet Criterion Module
==========================

This module extends the RTDETR (Real-Time Detection Transformer) criterion to create
AfforMeNet, which adds affordance prediction capabilities. The criterion computes
losses for both object detection and affordance segmentation tasks.

Key Extensions for AfforMeNet:
---------------------------
1. Affordance Loss Integration:
   - Multi-class affordance segmentation loss
   - Pixel-wise affordance prediction within object regions
   - Integrated with RTDETR's detection losses

2. Dual-Task Learning:
   - Balances object detection and affordance prediction
   - Maintains RTDETR's real-time performance
   - Coordinated loss computation through Hungarian matching

Loss Components:
--------------
1. Object Detection (from RTDETR):
   - Classification (CE/BCE/Focal/VFL)
   - Bounding box regression (L1 + GIoU)
   - Cardinality for prediction counting

2. Affordance Prediction (new):
   - Pixel-wise cross-entropy loss
   - Region-specific affordance learning
   - Optional auxiliary affordance predictions

Loss Computation Flow:
--------------------
1. Hungarian Matching:
   - Assigns predictions to ground truth
   - Considers both detection and affordance quality

2. Detection Losses:
   - Compute classification and box regression losses
   - Apply detection-specific weights

3. Affordance Losses:
   - Compute affordance segmentation loss
   - Consider only valid object regions
   - Apply affordance-specific weights

4. Auxiliary Losses:
   - Handle intermediate decoder outputs
   - Include affordance predictions in auxiliary layers
   - Process denoising predictions

Configuration Options:
-------------------
1. Affordance Settings:
   ```python
   criterion = SetCriterion(
       matcher=HungarianMatcher(),
       weight_dict={
           'loss_ce': 1,
           'loss_bbox': 5,
           'loss_giou': 2,
           'loss_affordance': 1.0  # New affordance loss weight
       },
       losses=['labels', 'boxes', 'cardinality', 'affordances'],  # Added affordances
       num_classes=80,
       use_affordance=True  # Enable affordance prediction
   )
   ```

2. Loss Weights:
   - Balance detection and affordance tasks
   - Adjust auxiliary loss weights
   - Configure affordance loss impact

Integration Notes:
----------------
1. Input Requirements:
   - Detection inputs: class labels, bounding boxes
   - Affordance inputs: pixel-wise affordance maps
   - Valid region masks for affordance computation

2. Output Format:
   - Detection outputs: class logits, box coordinates
   - Affordance outputs: pixel-wise affordance logits
   - Auxiliary outputs for both tasks

3. Performance Considerations:
   - Maintain real-time processing capability
   - Balance memory usage with affordance prediction
   - Optimize affordance computation efficiency

Extension Guidelines:
------------------
1. Custom Affordance Types:
   - Extend affordance loss for multiple types
   - Add new affordance metrics
   - Implement custom affordance evaluation

2. Training Enhancements:
   - Add affordance-specific auxiliary tasks
   - Implement region-based sampling
   - Create affordance-aware matching strategies

References:
----------
- RT-DETR: Base real-time detection transformer
- AfforMeNet: Affordance detection extension
- DETR: Original transformer detection architecture

by Anhar
"""

from __future__ import annotations  # py3.8: lazy annotations (dict[...]|None)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.core import register
from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.zoo.rtdetr.affordance import (
    decode_affordance_masks,
    decode_affordance_masks_embedding,
)

# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

# Affordance segmentation loss hyper-parameters. Plain pixel CE under-weights
# thin/rare affordance regions and ignores boundary quality, which the
# boundary-weighted F_beta^w metric punishes. We combine focal CE (hard-pixel
# focus) + edge weighting (boundary focus) + soft multiclass Dice (region
# overlap, robust to class imbalance).
_AFF_IGNORE_INDEX = 255
_AFF_FOCAL_GAMMA = 2.0
_AFF_BOUNDARY_W = 4.0  # extra weight on GT-boundary pixels
_AFF_DICE_W = 1.0  # weight of the Dice term relative to (focal+edge) CE
_AFF_DICE_SMOOTH = 1.0


def _affordance_boundary_weight(
    target: torch.Tensor,
    valid: torch.Tensor,
    boundary_w: float = _AFF_BOUNDARY_W,
    class_boundary_w: dict[int, float] | None = None,
) -> torch.Tensor:
    """Per-pixel weight map that up-weights GT class boundaries.

    A pixel is a boundary if any 3x3 neighbour has a different label. Boundary
    pixels get ``boundary_w`` (non-boundary get 1.0). If ``class_boundary_w`` is
    given, boundary pixels whose GT label is in the map get that class-specific
    weight *instead* of the global ``boundary_w`` — used to emphasise a single
    hard class (e.g. grasp=5) without touching the rest (global reweighting is a
    measured local optimum; per-class was untested). When ``class_boundary_w``
    is None/empty the output is byte-identical to the global-only path.
    """
    t = target.clone()
    t[~valid] = 0
    t_f = t.unsqueeze(1).float()
    # Min/max over a 3x3 window differ iff the window straddles a class edge.
    mx = F.max_pool2d(t_f, kernel_size=3, stride=1, padding=1)
    mn = -F.max_pool2d(-t_f, kernel_size=3, stride=1, padding=1)
    is_boundary = (mx != mn).squeeze(1) & valid
    weight = torch.ones_like(target, dtype=torch.float)
    weight[is_boundary] = boundary_w
    if class_boundary_w:
        for cls_idx, cls_w in class_boundary_w.items():
            weight[is_boundary & (target == cls_idx)] = cls_w
    return weight


def _affordance_seg_loss(
    src: torch.Tensor,
    target: torch.Tensor,
    focal_gamma: float = _AFF_FOCAL_GAMMA,
    boundary_w: float = _AFF_BOUNDARY_W,
    dice_w: float = _AFF_DICE_W,
    class_boundary_w: dict[int, float] | None = None,
) -> torch.Tensor:
    """Edge-weighted focal CE + soft multiclass Dice for per-instance masks.

    Args:
        src: [N, C, H, W] affordance logits.
        target: [N, H, W] int labels; ``_AFF_IGNORE_INDEX`` pixels are ignored.
        focal_gamma: focal-CE hard-pixel focusing exponent.
        boundary_w: per-pixel up-weight on GT class boundaries.
        dice_w: weight of the Dice term relative to the (focal+edge) CE term.
        class_boundary_w: optional {class_index: boundary_weight} overriding
            ``boundary_w`` on that class's boundary pixels only.

    Defaults reproduce the v2/v3 constants exactly, so unswept callers are
    byte-identical.
    """
    num_classes = src.shape[1]
    valid = target != _AFF_IGNORE_INDEX
    if valid.sum() == 0:
        return src.sum() * 0.0

    # --- Edge-weighted focal cross-entropy ---
    ce = F.cross_entropy(src, target, ignore_index=_AFF_IGNORE_INDEX, reduction="none")  # [N, H, W]
    pt = torch.exp(-ce)
    focal = ((1.0 - pt) ** focal_gamma) * ce
    w = _affordance_boundary_weight(target, valid, boundary_w, class_boundary_w)
    focal_ce = (focal * w)[valid].sum() / w[valid].sum().clamp(min=1.0)

    # --- Soft multiclass Dice (ignore-aware) ---
    prob = src.softmax(dim=1)
    tgt_safe = target.clone()
    tgt_safe[~valid] = 0
    onehot = F.one_hot(tgt_safe, num_classes).permute(0, 3, 1, 2).float()
    vmask = valid.unsqueeze(1).float()
    prob = prob * vmask
    onehot = onehot * vmask
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dims)
    denom = prob.sum(dims) + onehot.sum(dims)
    dice = 1.0 - (2.0 * inter + _AFF_DICE_SMOOTH) / (denom + _AFF_DICE_SMOOTH)
    dice = dice.mean()

    return focal_ce + dice_w * dice


def _dense_semantic_target(masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Merge per-instance full-image label maps [N, H, W] into one semantic map.

    Background stays 0. Pixels where two instances claim DIFFERENT positive
    classes are set to ``_AFF_IGNORE_INDEX`` (ambiguous supervision), as is any
    label outside [0, num_classes). An empty instance list yields an
    all-background map (correct supervision for an object-free image).
    """
    if masks.shape[0] == 0:
        return torch.zeros(masks.shape[-2:], dtype=torch.long, device=masks.device)
    m = masks.long()
    acc = torch.zeros_like(m[0])
    conflict = torch.zeros_like(acc, dtype=torch.bool)
    for inst in m:
        fg = inst > 0
        conflict |= fg & (acc > 0) & (acc != inst)
        acc = torch.where(fg & (acc == 0), inst, acc)
    acc[conflict] = _AFF_IGNORE_INDEX
    acc[acc >= num_classes] = _AFF_IGNORE_INDEX
    acc[acc < 0] = _AFF_IGNORE_INDEX
    return acc


@register
class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    __share__ = ["num_classes", "use_affordance", "num_affordance_classes"]
    __inject__ = [
        "matcher",
    ]

    # 9 affordance classes + background (IIT-AFF); matches
    # AffordanceBranch.output_dim in the model config. Class-level default kept
    # for backward compatibility; the __init__ param (config-shared) overrides
    # it per instance so UMD (8) and IIT (10) coexist.
    num_affordance_classes = 10

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        eos_coef=1e-4,
        num_classes=80,
        use_affordance=False,
        num_affordance_classes=10,
        affordance_loss="combined",
        aff_focal_gamma=_AFF_FOCAL_GAMMA,
        aff_boundary_w=_AFF_BOUNDARY_W,
        aff_dice_w=_AFF_DICE_W,
        aff_class_boundary_w=None,
        aff_aux_loss_weight=0.25,
        aff_full_res_loss=False,
        o2m_loss_weight=1.0,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            num_affordance_classes: affordance channels incl. background (IIT-AFF
                default 10; UMD 8). Must equal AffordanceBranch.output_dim.
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            affordance_loss: 'combined' (v2 default: edge-weighted focal CE + Dice) or
                'ce' (ablation: plain pixel cross-entropy, matches the pre-v2 criterion).
            aff_focal_gamma: focal-CE focusing exponent for the combined affordance loss.
            aff_boundary_w: boundary-pixel up-weight for the combined affordance loss.
            aff_dice_w: Dice-term weight (vs focal+edge CE) for the combined affordance loss.
                Defaults reproduce the v2/v3 constants, so configs that omit these are unchanged.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.use_affordance = use_affordance
        # Instance-level affordance class count (config-shared). Set before
        # _validate_class_boundary_w, which bounds-checks against it.
        self.num_affordance_classes = num_affordance_classes
        if affordance_loss not in ("combined", "ce"):
            raise ValueError(f"affordance_loss must be 'combined' or 'ce', got {affordance_loss!r}")
        self.affordance_loss = affordance_loss
        self.aff_focal_gamma = aff_focal_gamma
        self.aff_boundary_w = aff_boundary_w
        self.aff_dice_w = aff_dice_w
        # Optional per-class boundary up-weight override. Config passes a
        # {class_index: weight} map (IIT-AFF affordance labels: 0=bg, 1=contain,
        # 2=cut, 3=display, 4=engine, 5=grasp, 6=hit, 7=pound, 8=support,
        # 9=w-grasp). None => global-only boundary weighting (byte-identical to
        # every existing config). Validated at build time so a typo'd class
        # index fails loudly instead of silently no-op-ing.
        self.aff_class_boundary_w = self._validate_class_boundary_w(aff_class_boundary_w)
        # Deep-supervision aux affordance-loss scale, applied ON TOP of the
        # base loss_affordance weight for each intermediate decoder layer (the
        # criterion multiplies by weight_dict before suffixing _aux_i, so this
        # is the per-layer multiplier that keeps the 5 aux layers from
        # overpowering detection). No-op unless the model emits per-layer
        # aff_kernel (AffordanceBranch.deep_supervision=True).
        self.aff_aux_loss_weight = aff_aux_loss_weight
        # When True, supervise the affordance loss at the GT mask resolution by
        # bilinearly upsampling the decoded per-query logits (mirrors
        # loss_masks), instead of nearest-downsampling the GT to the feature
        # grid (which aliases thin structure). Train-time only; inference
        # unchanged. Per-matched-query, so the cost is small.
        self.aff_full_res_loss = aff_full_res_loss
        # Scale on the one-to-many auxiliary detection loss (H-DETR style). The
        # decoder emits 'o2m_aux_outputs' only when its num_o2m_queries>0, so
        # this is a no-op for every existing config. Train-time only.
        self.o2m_loss_weight = o2m_loss_weight

        # Remove affordance losses (incl. affordances_dense) if not using it
        if not use_affordance and any(ls.startswith("affordances") for ls in self.losses):
            self.losses = [ls for ls in self.losses if not ls.startswith("affordances")]
            # Remove affordance-related weights from weight_dict
            self.weight_dict = {
                k: v for k, v in weight_dict.items() if not k.startswith("loss_affordance")
            }

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.alpha = alpha
        self.gamma = gamma

    def _validate_class_boundary_w(self, cfg) -> dict[int, float] | None:
        """Normalise + validate the per-class boundary-weight override.

        Accepts None (disabled) or a mapping of affordance class index -> weight.
        YAML may deliver keys as str; coerce to int and bounds-check against
        ``num_affordance_classes`` so a bad index raises at construction, not
        silently. Returns None when disabled so the hot path skips the loop.
        """
        if not cfg:
            return None
        if not isinstance(cfg, dict):
            raise ValueError(
                f"aff_class_boundary_w must be a dict or None, got {type(cfg).__name__}"
            )
        out: dict[int, float] = {}
        for k, v in cfg.items():
            idx = int(k)
            if not 0 <= idx < self.num_affordance_classes:
                raise ValueError(
                    f"aff_class_boundary_w index {idx} out of range "
                    f"[0, {self.num_affordance_classes})"
                )
            out[idx] = float(v)
        return out

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1.0, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_bce": loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        # ce_loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction="none")
        # prob = F.sigmoid(src_logits) # TODO .detach()
        # p_t = prob * target + (1 - prob) * (1 - target)
        # alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_affordances(self, outputs, targets, indices, num_boxes):
        """Per-instance multiclass affordance loss.

        Decodes affordance logits *only* for Hungarian-matched queries by
        applying each matched query's dynamic kernel to the shared
        affordance-feature map, then a pixel-wise cross-entropy against the
        ground-truth affordance map (downsampled to the feature resolution).
        No dense [num_queries, C, H, W] tensor is ever materialised.
        """
        if not self.use_affordance or "aff_kernel" not in outputs:
            return {}

        batch_idx, query_idx = self._get_src_permutation_idx(indices)
        if batch_idx.numel() == 0:
            return {}

        aff_feat = outputs["aff_feat"]  # [B, Cf, Hf, Wf]
        kernels = outputs["aff_kernel"][batch_idx, query_idx]  # [num_matched, n_params]
        hf, wf = aff_feat.shape[-2:]

        # Dispatch on head type — 'dynamic' (default) is the dynamic-conv
        # head; 'embedding' is the MaskFormer-style A5 ablation alternative.
        head_type = outputs.get("aff_head_type", "dynamic")
        decode_fn = (
            decode_affordance_masks_embedding
            if head_type == "embedding"
            else decode_affordance_masks
        )
        # Box-relative geometry channels (dynamic head only): use the matched
        # queries' predicted boxes, detached so the affordance loss doesn't
        # backprop into / destabilise box regression. Predicted (not GT) boxes
        # keep train/inference geometry consistent.
        sel_boxes = None
        if outputs.get("aff_box_relative", False):
            sel_boxes = outputs["pred_boxes"][batch_idx, query_idx].detach()
        src_affordances = decode_fn(
            aff_feat=aff_feat,
            kernels=kernels,
            batch_index=batch_idx.to(aff_feat.device),
            output_dim=self.num_affordance_classes,
            boxes=sel_boxes,
        )  # [num_matched, num_affordance_classes, Hf, Wf]

        # Build matched targets. Two supervision regimes:
        #  - full-res (aff_full_res_loss): keep GT at its native (transformed)
        #    resolution and UPSAMPLE the decoded logits to it (mirrors
        #    loss_masks), so thin structure is never nearest-aliased away.
        #  - feature-grid (default/legacy): nearest-downsample GT to (hf, wf)
        #    and supervise at the coarse decode grid.
        if self.aff_full_res_loss:
            target_list = [
                t["masks"][tgt_obj_idx].float()
                for t, (_, tgt_obj_idx) in zip(targets, indices)
                if tgt_obj_idx.numel() > 0
            ]
            target_affordances = torch.cat(target_list, dim=0).to(src_affordances.device)
            th, tw = target_affordances.shape[-2:]
            if (th, tw) != (hf, wf):
                src_affordances = F.interpolate(
                    src_affordances, size=(th, tw), mode="bilinear", align_corners=False
                )
            target_affordances = target_affordances.long()
        else:
            target_list = []
            for t, (_, tgt_obj_idx) in zip(targets, indices):
                if tgt_obj_idx.numel() == 0:
                    continue
                masks = t["masks"][tgt_obj_idx].float()  # [n, H, W]
                masks = F.interpolate(masks.unsqueeze(1), size=(hf, wf), mode="nearest").squeeze(1)
                target_list.append(masks)
            target_affordances = torch.cat(target_list, dim=0).to(src_affordances.device)
            target_affordances = target_affordances.long()
        # Ignore any label outside [0, num_affordance_classes).
        target_affordances[target_affordances >= self.num_affordance_classes] = _AFF_IGNORE_INDEX
        target_affordances[target_affordances < 0] = _AFF_IGNORE_INDEX

        if self.affordance_loss == "combined":
            loss_affordance = _affordance_seg_loss(
                src_affordances,
                target_affordances,
                focal_gamma=self.aff_focal_gamma,
                boundary_w=self.aff_boundary_w,
                dice_w=self.aff_dice_w,
                class_boundary_w=self.aff_class_boundary_w,
            )
        else:  # 'ce' — pre-v2 plain pixel cross-entropy, for the LOO loss ablation.
            loss_affordance = F.cross_entropy(
                src_affordances,
                target_affordances,
                ignore_index=_AFF_IGNORE_INDEX,
            )
        return {"loss_affordance": loss_affordance}

    def loss_affordances_dense(self, outputs, targets, indices, num_boxes):
        """Image-level dense auxiliary affordance loss (box-independent).

        Supervises the AffordanceBranch ``dense_aux`` 1x1 head against the
        pixel-wise merge of every instance's full-image label map — so the
        shared trunk receives class gradients everywhere, not only where a
        Hungarian-matched query happens to look. Uses the same edge-weighted
        focal CE + Dice as the per-query loss, at GT resolution (logits
        bilinearly upsampled, mirroring aff_full_res_loss).

        Guarded: only the top-level outputs carry 'aff_dense_logits' (see
        rtdetr.py), so the aux/dn loss loops no-op here by construction.
        Hungarian ``indices`` and ``num_boxes`` are unused (image-level loss).
        """
        if not self.use_affordance or "aff_dense_logits" not in outputs:
            return {}
        logits = outputs["aff_dense_logits"]  # [B, C, Hf, Wf]
        target = torch.stack(
            [_dense_semantic_target(t["masks"], self.num_affordance_classes) for t in targets],
            dim=0,
        ).to(logits.device)
        th, tw = target.shape[-2:]
        if (th, tw) != logits.shape[-2:]:
            logits = F.interpolate(logits, size=(th, tw), mode="bilinear", align_corners=False)
        loss = _affordance_seg_loss(
            logits,
            target,
            focal_gamma=self.aff_focal_gamma,
            boundary_w=self.aff_boundary_w,
            dice_w=self.aff_dice_w,
            class_boundary_w=self.aff_class_boundary_w,
        )
        return {"loss_affordance_dense": loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # NOTE: DETR's binary-instance `loss_masks` ("masks") was removed from this
        # fork. Its helper imports were lost in the fork history, so it could only
        # ever have raised NameError, and no config selects it. Affordance masks are
        # supervised by `loss_affordances` / `loss_affordances_dense` instead.
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "bce": self.loss_labels_bce,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "affordances": self.loss_affordances,
            "affordances_dense": self.loss_affordances_dense,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        num_boxes_o2o = num_boxes  # preserve pre-DN-scaling normaliser for o2m

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    if loss == "affordances":
                        # Deep-supervision down-weight per intermediate layer
                        # (no-op when the layer carries no aff_kernel).
                        l_dict = {k: v * self.aff_aux_loss_weight for k, v in l_dict.items()}
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if "dn_aux_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]

            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # One-to-many auxiliary detection loss (H-DETR). Each GT is repeated
        # o2m_group_repeat times and Hungarian-matched against the isolated o2m
        # query group, so every GT gets that many positive queries — denser
        # supervision to lift recall on hard / low-AP hosts. Detection-only:
        # 'masks' is skipped and 'affordances' self-skips (o2m carries no
        # aff_kernel). No-op unless the decoder emitted an o2m group.
        if "o2m_aux_outputs" in outputs:
            assert "o2m_meta" in outputs, "o2m_aux_outputs requires o2m_meta"
            k_rep = outputs["o2m_meta"]["o2m_group_repeat"]
            o2m_targets = [
                {"labels": t["labels"].tile(k_rep), "boxes": t["boxes"].tile(k_rep, 1)}
                for t in targets
            ]
            o2m_num_boxes = num_boxes_o2o * k_rep
            for i, aux_outputs in enumerate(outputs["o2m_aux_outputs"]):
                indices = self.matcher(aux_outputs, o2m_targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, o2m_targets, indices, o2m_num_boxes, **kwargs
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k: v * self.o2m_loss_weight for k, v in l_dict.items()}
                    l_dict = {k + f"_o2m_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
