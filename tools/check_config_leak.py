"""Regression check: dataloader-build ORDER must not leak dataset kwargs.

``yaml_utils.create()`` permanently merges instance kwargs into the GLOBAL
class node (``src/core/yaml_utils.py`` ``_cfg.update(kwargs)``), so kwargs
present only on the train dataset node can propagate to the val dataset when
the train dataloader is built FIRST — which the solver does. We hit this with
the ``copy_paste`` kwarg: the val dataset inherited it, pasted donors were
scored as false positives against clean file-GT, and logged means sat a
constant ~4.3pp below truth for a whole training run.

The dataset-side guard (``IITDetection``: copy_paste only on the train split)
closes the known instance; this script re-checks BOTH build orders end-to-end
so any future train-only dataset kwarg gets caught before a multi-hour run.

    python tools/check_config_leak.py --config <cfg with train-only kwargs>
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import YAMLConfig  # noqa: E402

DEFAULT_CONFIG = "configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup_copypaste.yml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default=DEFAULT_CONFIG)
    return p.parse_args()


def val_state(cfg) -> tuple:
    vd = cfg.val_dataloader.dataset
    return (
        getattr(vd, "_copy_paste", None),
        len(getattr(vd, "_donor_index", [])),
        vd.image_set,
    )


def main():
    args = parse_args()
    failures = []

    # Solver order: train dataloader first (the order that leaked in A2).
    cfg = YAMLConfig(args.config)
    _ = cfg.train_dataloader
    aug, donors, split = val_state(cfg)
    print(f"train-first: val split={split} copy_paste={aug} donors={donors}")
    if aug is not None or donors:
        failures.append("train-first order leaked copy_paste into the val dataset")

    # Reverse order (fresh config object — the global node is already mutated
    # in this process, so this also covers cross-instantiation carry-over).
    cfg2 = YAMLConfig(args.config)
    aug, donors, split = val_state(cfg2)
    _ = cfg2.train_dataloader
    print(f"val-first:   val split={split} copy_paste={aug} donors={donors}")
    if aug is not None or donors:
        failures.append("val-first order leaked copy_paste into the val dataset")

    if failures:
        for f in failures:
            print(f"LEAK: {f}")
        sys.exit(1)
    print("OK: val dataset clean in both build orders")


if __name__ == "__main__":
    main()
