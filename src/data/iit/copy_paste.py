"""Targeted copy-paste augmentation for IIT-AFF, used as a recall lever.

Pastes whole object instances (RGB pixels + per-pixel affordance label map +
box) from donor images into the current training image at NATIVE resolution,
before the geometric transform chain, so downstream crop/flip/resize/sanitize
apply jointly to pasted objects.

Motivation: our error analysis found the largest recoverable component of the
remaining gap is a detection-coupling tax, i.e. GT-present (image, class) pairs
with no surviving query, concentrated on low-AP host objects (bottle->w-grasp,
spatula->support, hammer->pound, multi-tool grasp scenes). Adding one-to-many
matching showed those misses are NOT supervision-density-bound, so this lever
instead adds example DIVERSITY: the same hard hosts in new contexts, scales
and backgrounds.

Simplified from Ghiasi et al. 2021 ("Simple Copy-Paste"): no blending, random
scale jitter + random placement. Unlike COCO copy-paste, the pasted payload
carries the per-pixel AFFORDANCE label map (values 1..9), not a binary
instance mask, so the label crop is resized with NEAREST to preserve labels.
Occluded regions of existing instance masks are zeroed; existing boxes are
left untouched (standard simple-copy-paste behaviour — the residual box noise
is bounded by ``max_overlap``).

All randomness is drawn from the torch RNG (seeded globally by
``dist.set_seed`` and per-DataLoader-worker by torch), keeping runs
reproducible under ``--seed``.
"""

from __future__ import annotations  # py3.8: lazy annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import datapoints

__all__ = ["CopyPasteAugmentor", "DEFAULT_DONOR_CLASSES"]

# Low-AP host objects from the miss diagnosis (Spearman host-AP vs miss% = -0.67).
DEFAULT_DONOR_CLASSES = ("bottle", "spatula", "hammer")


class CopyPasteAugmentor:
    """Paste donor object instances into a (PIL image, IIT target) sample.

    Args:
        prob: probability of attempting copy-paste on a sample.
        max_pastes: pastes per augmented sample are drawn uniformly from
            ``[1, max_pastes]``.
        scale_jitter: ``(lo, hi)`` multiplicative jitter applied on top of the
            donor->destination native-size normalisation.
        max_overlap: max allowed fraction of ANY existing GT box covered by
            the pasted patch (intersection / existing-box area). Placements
            violating this are re-sampled up to ``max_tries`` times.
        max_tries: placement attempts before giving up on a paste.
        min_visible_px: minimum affordance-labelled pixels in the scaled
            patch for the paste to count.
        min_side: minimum pasted-patch side in pixels after scaling.
        max_frac: pasted patch is capped to this fraction of each
            destination image side.
    """

    def __init__(
        self,
        prob: float = 0.5,
        max_pastes: int = 2,
        scale_jitter: tuple[float, float] = (0.5, 1.0),
        max_overlap: float = 0.3,
        max_tries: int = 10,
        min_visible_px: int = 96,
        min_side: int = 24,
        max_frac: float = 0.8,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        if max_pastes < 1:
            raise ValueError(f"max_pastes must be >= 1, got {max_pastes}")
        if not 0.0 < scale_jitter[0] <= scale_jitter[1]:
            raise ValueError(f"invalid scale_jitter {scale_jitter}")
        self.prob = float(prob)
        self.max_pastes = int(max_pastes)
        self.scale_jitter = (float(scale_jitter[0]), float(scale_jitter[1]))
        self.max_overlap = float(max_overlap)
        self.max_tries = int(max_tries)
        self.min_visible_px = int(min_visible_px)
        self.min_side = int(min_side)
        self.max_frac = float(max_frac)

    def __call__(
        self,
        img: Image.Image,
        target: dict,
        sample_donor: Callable[[], tuple[Image.Image, dict, int] | None],
    ) -> tuple[Image.Image, dict]:
        """Return the (possibly) augmented sample; inputs untouched when off."""
        if float(torch.rand(())) >= self.prob:
            return img, target

        n_pastes = int(torch.randint(1, self.max_pastes + 1, ()))
        img_np = np.array(img, dtype=np.uint8)
        boxes = torch.as_tensor(target["boxes"]).reshape(-1, 4).clone()
        masks = torch.as_tensor(target["masks"]).clone()
        labels = [target["labels"]]
        areas = [target["area"]]
        new_masks = []
        pasted = 0

        for _ in range(n_pastes):
            donor = sample_donor()
            if donor is None:
                continue
            entry = self._paste_one(img_np, boxes, masks, new_masks, donor)
            if entry is None:
                continue
            box, label = entry
            boxes = torch.cat([boxes, box[None]], dim=0)
            labels.append(label[None])
            areas.append(((box[2] - box[0]) * (box[3] - box[1]))[None])
            pasted += 1

        if pasted == 0:
            return img, target

        h, w = img_np.shape[:2]
        out = dict(target)
        out["boxes"] = datapoints.BoundingBox(
            boxes, format=datapoints.BoundingBoxFormat.XYXY, spatial_size=(h, w)
        )
        all_masks = torch.cat([masks, torch.stack(new_masks)], dim=0) if new_masks else masks
        out["masks"] = datapoints.Mask(all_masks)
        out["labels"] = torch.cat(labels)
        out["area"] = torch.cat(areas)
        out["iscrowd"] = torch.cat([target["iscrowd"], torch.zeros(pasted, dtype=torch.int64)])
        out["difficult"] = torch.cat([target["difficult"], torch.zeros(pasted, dtype=torch.bool)])
        return Image.fromarray(img_np), out

    def _paste_one(
        self,
        img_np: np.ndarray,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        new_masks: list,
        donor: tuple[Image.Image, dict, int],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Composite one donor instance in place; return (box, label) or None."""
        donor_img, donor_target, obj_idx = donor
        prepared = self._prepare_patch(donor_img, donor_target, obj_idx, img_np.shape[:2])
        if prepared is None:
            return None
        patch, label_crop, alpha = prepared

        placement = self._find_placement(boxes, alpha.shape, img_np.shape[:2])
        if placement is None:
            return None
        y0, x0 = placement
        ph, pw = alpha.shape

        # Composite RGB, occlude existing instance masks, add the new channel.
        region = img_np[y0 : y0 + ph, x0 : x0 + pw]
        region[alpha] = patch[alpha]
        alpha_t = torch.from_numpy(alpha)
        if masks.numel() > 0:
            masks[:, y0 : y0 + ph, x0 : x0 + pw][:, alpha_t] = 0
        for m in new_masks:
            m[y0 : y0 + ph, x0 : x0 + pw][alpha_t] = 0
        channel = torch.zeros(img_np.shape[0], img_np.shape[1], dtype=torch.uint8)
        channel[y0 : y0 + ph, x0 : x0 + pw][alpha_t] = torch.from_numpy(label_crop)[alpha_t]
        new_masks.append(channel)

        ys, xs = np.nonzero(alpha)
        box = torch.tensor(
            [x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1],
            dtype=torch.float32,
        )
        label = torch.as_tensor(donor_target["labels"][obj_idx], dtype=torch.int64)
        return box, label

    def _prepare_patch(
        self,
        donor_img: Image.Image,
        donor_target: dict,
        obj_idx: int,
        dst_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Crop + rescale the donor instance -> (rgb, affordance labels, alpha)."""
        src_np = np.asarray(donor_img, dtype=np.uint8)
        src_h, src_w = src_np.shape[:2]
        x0, y0, x1, y1 = (float(v) for v in torch.as_tensor(donor_target["boxes"][obj_idx]))
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(src_w, int(np.ceil(x1))), min(src_h, int(np.ceil(y1)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None

        label_full = torch.as_tensor(donor_target["masks"][obj_idx]).numpy()
        label_crop = label_full[y0:y1, x0:x1]
        if int((label_crop > 0).sum()) < self.min_visible_px:
            return None

        dst_h, dst_w = dst_hw
        lo, hi = self.scale_jitter
        jitter = lo + (hi - lo) * float(torch.rand(()))
        scale = jitter * min(dst_h, dst_w) / min(src_h, src_w)
        pw = int(round((x1 - x0) * scale))
        ph = int(round((y1 - y0) * scale))
        # Cap to a fraction of the destination, keeping aspect ratio.
        cap = min(self.max_frac * dst_w / max(pw, 1), self.max_frac * dst_h / max(ph, 1), 1.0)
        pw, ph = int(pw * cap), int(ph * cap)
        if min(pw, ph) < self.min_side:
            return None

        rgb_crop = np.ascontiguousarray(src_np[y0:y1, x0:x1])
        rgb = np.asarray(Image.fromarray(rgb_crop).resize((pw, ph), Image.BILINEAR), dtype=np.uint8)
        # np.array (copy) keeps the buffer writable for torch.from_numpy.
        lab = np.array(
            Image.fromarray(np.ascontiguousarray(label_crop)).resize((pw, ph), Image.NEAREST),
            dtype=np.uint8,
        )
        alpha = lab > 0
        if int(alpha.sum()) < self.min_visible_px:
            return None
        return rgb, lab, alpha

    def _find_placement(
        self,
        boxes: torch.Tensor,
        patch_hw: tuple[int, int],
        dst_hw: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Sample a top-left corner whose patch rect respects max_overlap."""
        ph, pw = patch_hw
        dst_h, dst_w = dst_hw
        if ph > dst_h or pw > dst_w:
            return None
        for _ in range(self.max_tries):
            y0 = int(torch.randint(0, dst_h - ph + 1, ()))
            x0 = int(torch.randint(0, dst_w - pw + 1, ()))
            if boxes.numel() == 0 or self._max_covered(boxes, x0, y0, pw, ph) <= self.max_overlap:
                return y0, x0
        return None

    @staticmethod
    def _max_covered(boxes: torch.Tensor, x0: int, y0: int, pw: int, ph: int) -> float:
        """Largest fraction of any existing box covered by the patch rect."""
        ix0 = torch.clamp(boxes[:, 0], min=float(x0))
        iy0 = torch.clamp(boxes[:, 1], min=float(y0))
        ix1 = torch.clamp(boxes[:, 2], max=float(x0 + pw))
        iy1 = torch.clamp(boxes[:, 3], max=float(y0 + ph))
        inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
        area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=1e-6)
        return float((inter / area).max())
