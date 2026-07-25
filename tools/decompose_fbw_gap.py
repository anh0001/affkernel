"""Decompose the F_beta^w gap into detection-recall coupling vs segmentation quality.

The headline model sits ~0.11 below its OWN supervision ceiling (see
audit_supervision_ceiling.py). That residual has two very different causes, which
demand different fixes:

  (1) Detection-recall coupling: a GT-present (image, class) with NO surviving
      query produces an all-zero mask and scores F_beta^w = 0 exactly
      (iit_eval.py:438-446). This is a DETECTOR problem, not a mask problem.
  (2) Segmentation quality: among images where the model DID fire, how far is the
      mean F_beta^w below the ceiling? This is the mask-head / semantics problem.

This runs ONE eval forward of an existing checkpoint and reports, per class:
  n_gt        : GT-present images for the class
  n_miss      : of those, how many scored exactly 0 (no/!=shape prediction)
  miss_rate   : n_miss / n_gt  (the detection-coupling tax)
  mean_all    : mean F_beta^w over all GT-present images (== the reported metric)
  mean_fired  : mean F_beta^w over only images where the model fired (seg quality)
  recovered   : mean_all if misses were instead scored at mean_fired (upper bound
                on closing the coupling gap WITHOUT touching the mask head)

Usage:
    python tools/decompose_fbw_gap.py -c <config> -r <ckpt> [--beta2 1.0]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import src.misc.dist as dist  # noqa: E402
from src.core import YAMLConfig  # noqa: E402
from src.data import IITDetection, IITEvaluator  # noqa: E402
from src.data.iit.iit_eval import weighted_fbeta_measure  # noqa: E402
from src.solver import TASKS  # noqa: E402
from src.solver.det_engine import evaluate  # noqa: E402

CLASSES = ["contain", "cut", "display", "engine", "grasp", "hit", "pound", "support", "w-grasp"]
THIN = {"grasp", "cut", "pound", "w-grasp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--resume", "-r", required=True)
    p.add_argument("--beta2", type=float, default=1.0)
    return p.parse_args()


def union_by_img(entries):
    out = defaultdict(lambda: None)
    for e in entries:
        if not isinstance(e, dict) or "mask" not in e:
            continue
        m = np.asarray(e["mask"]).astype(bool)
        iid = e.get("image_id")
        out[iid] = m if out[iid] is None else (out[iid] | m)
    return out


def main():
    args = parse_args()
    dist.init_distributed()
    cfg = YAMLConfig(args.config, resume=args.resume)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()

    dataset = solver.val_dataloader.dataset
    for _ in range(10):
        if isinstance(dataset, (torchvision.datasets.CocoDetection, IITDetection)):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    module = solver.ema.module if solver.ema else solver.model

    _, ev = evaluate(
        module,
        solver.criterion,
        solver.postprocessor,
        solver.val_dataloader,
        dataset,
        solver.device,
        solver.output_dir,
    )
    if not isinstance(ev, IITEvaluator):
        raise ValueError(f"expected IITEvaluator, got {type(ev)}")

    b2 = args.beta2
    print(f"\n===== F_beta^w gap decomposition (beta2={b2}) =====")
    print(
        f"{'class':>9} {'n_gt':>5} {'n_miss':>6} {'miss%':>6} "
        f"{'mean_all':>9} {'mean_fired':>10} {'recovered':>9}"
    )
    agg = defaultdict(list)
    for cls in CLASSES:
        gt_masks = ev.load_ground_truth_masks(cls)
        if not gt_masks:
            continue
        gt_by_img = union_by_img(gt_masks)
        pred_by_img = union_by_img(ev.mask_predictions.get(cls, []))

        all_scores, fired_scores, n_miss = [], [], 0
        for iid, gt_m in gt_by_img.items():
            pred_m = pred_by_img.get(iid)
            if pred_m is None or pred_m.shape != gt_m.shape:
                q = weighted_fbeta_measure(np.zeros_like(gt_m, dtype=np.float64), gt_m, beta2=b2)
                if q is not None:
                    all_scores.append(q)
                n_miss += 1
            else:
                q = weighted_fbeta_measure(pred_m.astype(np.float64), gt_m, beta2=b2)
                if q is not None:
                    all_scores.append(q)
                    fired_scores.append(q)
        if not all_scores:
            continue
        n_gt = len(all_scores)
        mean_all = float(np.mean(all_scores))
        mean_fired = float(np.mean(fired_scores)) if fired_scores else 0.0
        # If every missed image were scored at mean_fired instead of ~0.
        recovered = (sum(fired_scores) + n_miss * mean_fired) / n_gt
        miss_rate = n_miss / n_gt
        agg["miss_rate"].append(miss_rate)
        agg["mean_all"].append(mean_all)
        agg["mean_fired"].append(mean_fired)
        agg["recovered"].append(recovered)
        agg["thin_mean_all" if cls in THIN else "_"].append(mean_all)
        print(
            f"{cls:>9} {n_gt:>5} {n_miss:>6} {miss_rate*100:>5.1f}% "
            f"{mean_all:>9.4f} {mean_fired:>10.4f} {recovered:>9.4f}"
        )

    print("-" * 64)
    print(
        f"{'MEAN':>9} {'':>5} {'':>6} {np.mean(agg['miss_rate'])*100:>5.1f}% "
        f"{np.mean(agg['mean_all']):>9.4f} {np.mean(agg['mean_fired']):>10.4f} "
        f"{np.mean(agg['recovered']):>9.4f}"
    )
    print(
        f"\nDetection-coupling tax  = mean_all -> recovered gap = "
        f"{np.mean(agg['recovered']) - np.mean(agg['mean_all']):+.4f} "
        f"(closeable WITHOUT touching the mask head)"
    )
    print(
        f"Segmentation-quality gap = mean_fired -> 1.0 = "
        f"{1.0 - np.mean(agg['mean_fired']):.4f} (mask-head / semantics)"
    )


if __name__ == "__main__":
    main()
