"""Measure the realized copy-paste rate of a recipe (and optional variants).

The paste SUCCESS rate differs from ``prob * max_pastes`` because placements
are rejected against ``max_overlap`` and patches against size/visibility
floors — and small-sample reads mislead (a 16-image read said 0.38/sample
where the true 150-image rate was 1.17/sample; see A2, 2026-07-04). Run this
on >=150 samples before judging a recipe too weak or too strong.

    python tools/sweep_paste_rate.py \\
        --config configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup_copypaste.yml
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import YAMLConfig  # noqa: E402
from src.data.iit.copy_paste import CopyPasteAugmentor  # noqa: E402

DEFAULT_CONFIG = "configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup_copypaste.yml"

# Variants measured alongside the config's own recipe (label -> ctor overrides).
VARIANTS = {
    "relax-overlap-0.25": {"max_overlap": 0.25},
    "orig-A2-draft": {"max_overlap": 0.3, "max_frac": 0.8, "scale_jitter": (0.5, 1.0)},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default=DEFAULT_CONFIG)
    p.add_argument("--n", type=int, default=150, help="samples per variant (>=150 advised)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-variants", action="store_true", help="measure the config recipe only")
    return p.parse_args()


def measure(ds, aug, n: int, seed: int) -> float:
    """Mean pastes/sample with prob forced to 1 (upper bound; scale by prob)."""
    torch.manual_seed(seed)
    ds._copy_paste = aug
    extra = 0
    for i in range(n):
        img, t = ds._load_raw(i)
        _, t2 = ds._copy_paste(img, t, ds._sample_donor)
        extra += len(t2["labels"]) - len(t["labels"])
    return extra / n


def main():
    args = parse_args()
    cfg = YAMLConfig(args.config)
    ds = cfg.train_dataloader.dataset
    assert ds._copy_paste is not None, "config has no copy_paste block"
    base_kwargs = dict(vars(ds._copy_paste))
    configured_prob = base_kwargs.pop("prob")
    max_pastes = base_kwargs["max_pastes"]

    recipes = {"config recipe": {}}
    if not args.skip_variants:
        recipes.update(VARIANTS)

    print(f"{args.n} samples/variant, prob forced to 1.0 (configured prob={configured_prob})")
    for name, overrides in recipes.items():
        kwargs = dict(base_kwargs)
        kwargs.update(overrides)
        rate = measure(ds, CopyPasteAugmentor(prob=1.0, **kwargs), args.n, args.seed)
        print(
            f"{name:>20}: {rate:.2f} pastes/sample (max {max_pastes}.0) "
            f"-> ~{rate * configured_prob:.2f} at configured prob"
        )


if __name__ == "__main__":
    main()
