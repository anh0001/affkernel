"""Re-score affordance F_beta^w at multiple beta^2 from a single forward pass.

Metric audit (2026-06-29): our evaluator defaults to beta^2 = 0.3
(AffordanceNet / IIT-AFF convention), but the Mur-Labadia 2025 SOTA states
beta = 1 (i.e. beta^2 = 1) in its F_beta^w. Our 0.3 numbers are therefore NOT
directly comparable to their 0.844 / 0.883 / 0.906. This tool runs a checkpoint's
forward pass ONCE and reports F_beta^w at every requested beta^2 from the same
stored per-instance predictions, so the re-scored numbers are exact and
self-consistent (no retraining, no re-inference per beta).

Usage:
    python tools/rescore_fbw_beta.py -c <config> -r <checkpoint.pth> [--betas 0.3 1.0]

The headline checkpoint is the LAST-epoch EMA checkpoint (checkpoint.pth): IIT-AFF
has no validation split (val_dataloader uses image_set='test'), so a fixed
last-epoch selection is the honest choice (avoids peak-test-epoch selection).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import src.misc.dist as dist  # noqa: E402
from src.core import YAMLConfig  # noqa: E402
from src.data import IITDetection, IITEvaluator  # noqa: E402
from src.solver import TASKS  # noqa: E402
from src.solver.det_engine import evaluate  # noqa: E402

CLASSES = ["contain", "cut", "display", "engine", "grasp", "hit", "pound", "support", "w-grasp"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--resume", "-r", required=True)
    p.add_argument("--betas", nargs="+", type=float, default=[0.3, 1.0])
    return p.parse_args()


def main():
    args = parse_args()
    dist.init_distributed()
    cfg = YAMLConfig(args.config, resume=args.resume)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()  # BaseSolver.setup(): builds model/ema, loads the resume checkpoint

    dataset = solver.val_dataloader.dataset
    for _ in range(10):  # unwrap nested Subset
        if isinstance(dataset, (torchvision.datasets.CocoDetection, IITDetection)):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if not isinstance(dataset, IITDetection):
        raise ValueError(f"expected IITDetection, got {type(dataset)}")

    module = solver.ema.module if solver.ema else solver.model
    _, evaluator = evaluate(
        module, solver.criterion, solver.postprocessor,
        solver.val_dataloader, dataset, solver.device, solver.output_dir,
    )
    if not isinstance(evaluator, IITEvaluator):
        raise ValueError(f"expected IITEvaluator, got {type(evaluator)}")

    print("\n===== F_beta^w re-score =====")
    print(f"checkpoint: {args.resume}")
    results = {}
    for b2 in args.betas:
        # The forward pass already populated evaluator.mask_predictions; re-scoring
        # only recombines stored per-instance precision/recall, so reset the
        # accumulator and recompute from the same predictions.
        evaluator.stats["affordance_fbw"]["Fbw"] = []
        per_class = evaluator.evaluate_affordances_fbw(beta2=b2)
        mean = float(np.mean(per_class)) if per_class else float("nan")
        results[b2] = (mean, per_class)
        print(f"  beta^2={b2}: mean F_beta^w = {mean:.4f}")

    print("\n----- summary (mean) -----")
    for b2, (mean, _) in results.items():
        print(f"beta^2={b2}\t{mean:.4f}")


if __name__ == "__main__":
    main()
