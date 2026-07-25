"""Free supervision-ceiling audit (no training, no model forward).

Theory-of-Constraints E1 probe. The affordance loss supervises at the aff_feat
grid: GT is downsampled to (hf, wf) with mode='nearest' (rtdetr_criterion.py:468)
and the per-pixel loss is computed there, while eval scores at native GT
resolution. This script measures the *information ceiling* that pipeline permits:
take each native GT mask, degrade it through a candidate supervision grid + interp
(exactly as the loss would see it), reconstruct it the way the postprocessor does
(bilinear upsample to native + threshold), and score F_beta^w against the native GT.

A perfect model cannot beat this ceiling. The gap from 1.0 is the F_beta^w thrown
away purely by the supervision grid/interpolation -- i.e. the headroom each fix can
recover, all train-time-only / inference-latency-neutral:

  grid 160 nearest -> stride-4 supervision ceiling (calibration vs old ~0.815)
  grid 320 nearest -> stride-2 supervision ceiling (CURRENT; bounds our 0.854)
  grid 320 area    -> E3 (de-alias GT: nearest -> area)
  grid 640 nearest -> E5 stride-1 / E2 supervise-at-input ceiling
  grid 640 area    -> absolute ceiling at fixed 640 input

Usage:
    python tools/audit_supervision_ceiling.py -c <config> [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import src.misc.dist as dist  # noqa: E402
from src.core import YAMLConfig  # noqa: E402
from src.data.iit.iit_eval import IITEvaluator, weighted_fbeta_measure  # noqa: E402

CLASSES = ["contain", "cut", "display", "engine", "grasp", "hit", "pound", "support", "w-grasp"]
THIN = {"grasp", "cut", "pound", "w-grasp"}
# (grid, interp) supervision configs to audit.
CONFIGS = [
    (160, "nearest"),
    (320, "nearest"),
    (320, "area"),
    (640, "nearest"),
    (640, "area"),
]
INPUT_SIZE = 640  # the train/eval input lock; supervision cannot exceed this at fixed input.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument(
        "--limit", type=int, default=0, help="cap (image,class) pairs per class for a quick run"
    )
    return p.parse_args()


def degrade(gt_bool: np.ndarray, grid: int, mode: str) -> np.ndarray:
    """native GT -> input resize (nearest) -> supervision grid (mode) ->
    bilinear upsample back to native -> threshold 0.5. Mirrors loss + postproc."""
    h, w = gt_bool.shape
    t = torch.from_numpy(gt_bool.astype(np.float32))[None, None]
    # Input resize to the 640 lock the model actually sees (masks: nearest).
    t = F.interpolate(t, size=(INPUT_SIZE, INPUT_SIZE), mode="nearest")
    # Supervision grid downsample (the lever under test).
    if mode == "area":
        d = F.interpolate(t, size=(grid, grid), mode="area")
    else:
        d = F.interpolate(t, size=(grid, grid), mode="nearest")
    # Reconstruct to native res the way the postprocessor does (bilinear + argmax).
    u = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
    return (u[0, 0].numpy() >= 0.5).astype(np.float64)


def main():
    args = parse_args()
    dist.init_distributed()
    cfg = YAMLConfig(args.config)
    dataset = cfg.val_dataloader.dataset
    while hasattr(dataset, "dataset") and not hasattr(dataset, "ids"):
        dataset = dataset.dataset

    ev = IITEvaluator(dataset)
    ev.image_ids = list(dataset.ids)

    # Per-config, per-class scores for both beta values.
    betas = {"b2=1.0": 1.0, "b2=0.3": 0.3}
    acc = {bk: {cfg_: defaultdict(list) for cfg_ in CONFIGS} for bk in betas}

    for cls in CLASSES:
        gt_masks = ev.load_ground_truth_masks(cls)
        if not gt_masks:
            continue
        gt_by_img = defaultdict(lambda: None)
        for g in gt_masks:
            m = np.asarray(g["mask"]).astype(bool)
            iid = g.get("image_id")
            gt_by_img[iid] = m if gt_by_img[iid] is None else (gt_by_img[iid] | m)

        items = list(gt_by_img.items())
        if args.limit:
            items = items[: args.limit]
        print(f"[{cls}] {len(items)} images", flush=True)
        for _, gt_m in items:
            if gt_m.sum() == 0:
                continue
            for grid, mode in CONFIGS:
                rec = degrade(gt_m, grid, mode)
                for bk, b2 in betas.items():
                    q = weighted_fbeta_measure(rec, gt_m, beta2=b2)
                    if q is not None:
                        acc[bk][(grid, mode)][cls].append(q)

    for bk in betas:
        print(f"\n===== supervision ceiling  F_beta^w  ({bk}) =====")
        header = (
            "grid/interp  " + "".join(f"{c[:5]:>7}" for c in CLASSES) + f"{'MEAN':>8}{'thin4':>7}"
        )
        print(header)
        for cfg_ in CONFIGS:
            per = acc[bk][cfg_]
            row_vals = [float(np.mean(per[c])) if per.get(c) else float("nan") for c in CLASSES]
            mean = float(np.nanmean(row_vals))
            thin = float(np.nanmean([v for c, v in zip(CLASSES, row_vals) if c in THIN]))
            tag = f"{cfg_[0]:>4}-{cfg_[1][:4]:<5}"
            print(tag + " " + "".join(f"{v:7.3f}" for v in row_vals) + f"{mean:8.4f}{thin:7.3f}")


if __name__ == "__main__":
    main()
