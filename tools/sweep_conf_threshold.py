"""Sweep the eval confidence threshold and measure F_beta^w (beta=1 and 0.3).

The postprocessor gates affordance masks at conf >= threshold (default 0.6). A
GT-present (image, class) pair with no surviving query is scored 0, which couples
detection recall into the F_beta^w segmentation metric. This sweeps the threshold
to find the F_beta^w-optimal single-pass operating point (latency-neutral, the
detector forward is unchanged) and to quantify how much the 0.6 gate costs.

Usage:
    python tools/sweep_conf_threshold.py -c <config> -r <ckpt> [--thresholds 0.6 0.4 0.25 0.1 0.05]
"""

from __future__ import annotations

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
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.6, 0.4, 0.25, 0.1, 0.05])
    return p.parse_args()


def mean_fbw(ev, beta2):
    ev.stats["affordance_fbw"]["Fbw"] = []
    per = ev.evaluate_affordances_fbw(beta2=beta2)
    return (float(np.mean(per)) if per else float("nan")), per


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

    print(f"\n===== confidence-threshold sweep =====\ncheckpoint: {args.resume}")
    print(f"{'thr':>5} | {'Fbw(b2=1)':>10} | {'Fbw(b2=0.3)':>11}")
    rows = []
    for thr in args.thresholds:
        solver.postprocessor.confidence_threshold = thr
        _, ev = evaluate(
            module, solver.criterion, solver.postprocessor,
            solver.val_dataloader, dataset, solver.device, solver.output_dir,
        )
        if not isinstance(ev, IITEvaluator):
            raise ValueError(f"expected IITEvaluator, got {type(ev)}")
        m1, per1 = mean_fbw(ev, 1.0)
        m03, _ = mean_fbw(ev, 0.3)
        rows.append((thr, m1, m03, per1))
        print(f"{thr:>5.2f} | {m1:>10.4f} | {m03:>11.4f}")

    print("\n----- per-class F_beta^w (beta=1) by threshold -----")
    print("thr  " + "  ".join(f"{c[:5]:>6}" for c in CLASSES))
    for thr, _, _, per1 in rows:
        print(f"{thr:>4.2f} " + "  ".join(f"{v:6.3f}" for v in per1))
    best = max(rows, key=lambda r: r[1])
    print(f"\nbest beta=1: {best[1]:.4f} @ thr={best[0]:.2f} "
          f"(default 0.6 = {[r for r in rows if abs(r[0]-0.6)<1e-9][0][1]:.4f} if swept)"
          if any(abs(r[0]-0.6) < 1e-9 for r in rows) else f"\nbest beta=1: {best[1]:.4f} @ thr={best[0]:.2f}")


if __name__ == "__main__":
    main()
