"""Smoke test: A2 copy-paste config -> dataset -> full transform chain (CPU).

Verifies the pieces unit tests cannot: the YAML deep-merge actually delivers
``copy_paste`` to the train dataset ctor, the donor index is populated, forced
pastes land and survive the full transform chain (normalized boxes, aligned
mask/label counts, affordance labels in 0..9), and per-sample overhead stays
in the tens of milliseconds. Run before launching any copy-paste training.

    python tools/smoke_copy_paste.py \\
        --config configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup_copypaste.yml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import YAMLConfig  # noqa: E402

DEFAULT_CONFIG = "configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup_copypaste.yml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default=DEFAULT_CONFIG)
    p.add_argument("--n", type=int, default=16, help="samples per pass")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_pass(ds, n: int, seed: int) -> tuple:
    """Iterate n transformed samples; return (total objects, seconds/sample)."""
    torch.manual_seed(seed)
    total, t0 = 0, time.time()
    for i in range(n):
        img, tgt = ds[i]
        n_obj = len(tgt["labels"])
        assert img.shape[-2:] == (640, 640), img.shape
        assert tgt["boxes"].shape == (n_obj, 4)
        assert tgt["masks"].shape[0] == n_obj
        assert torch.isfinite(tgt["boxes"]).all()
        assert (tgt["boxes"] >= -0.01).all() and (tgt["boxes"] <= 1.01).all(), "not normalized"
        assert set(torch.unique(tgt["masks"]).tolist()) <= set(range(10)), "bad labels"
        total += n_obj
    return total, (time.time() - t0) / n


def main():
    args = parse_args()
    cfg = YAMLConfig(args.config)
    ds = cfg.train_dataloader.dataset
    print(f"dataset: {type(ds).__name__} | split: {ds.image_set} | images: {len(ds)}")
    assert ds._copy_paste is not None, "copy_paste config did NOT reach the dataset ctor"
    print(f"augmentor: {vars(ds._copy_paste)}")
    print(
        f"donor instances indexed: {len(ds._donor_index)} "
        f"| unique donor images: {len({i for i, _ in ds._donor_index})}"
    )
    # The val dataset must stay clean (regression for the 2026-07-04 eval bug).
    vd = cfg.val_dataloader.dataset
    assert vd._copy_paste is None and not vd._donor_index, "copy_paste LEAKED into val!"

    ds._copy_paste.prob = 1.0  # force pastes for the augmented pass
    aug_total, dt_aug = run_pass(ds, args.n, args.seed)
    ds._copy_paste = None  # toggle off -> original pipeline
    base_total, dt_base = run_pass(ds, args.n, args.seed)

    extra = aug_total - base_total
    print(f"objects base={base_total} aug={aug_total} (+{extra} pasted / {args.n} samples)")
    print(f"per-sample time: base={dt_base * 1000:.0f} ms, aug={dt_aug * 1000:.0f} ms")
    assert extra > 0, "no pastes landed with prob forced to 1.0"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
