"""Checkpoint forensics: inspect, diff raw-vs-EMA, and repackage weight slots.

Consolidates the ad-hoc scripts written while debugging a copy-paste
augmentation run whose evaluation numbers turned out to be corrupted.
Subcommands:

  inspect <ckpt>              keys, last_epoch, per-slot tensor counts
  diff-ema <ckpt> [--top N]   largest |model - ema.module| entries (params AND
                              buffers; large BN running_var diffs flag a
                              train/eval distribution shift, e.g. heavy aug)
  strip-ema <ckpt> -o OUT     save a model-only checkpoint. WARNING: evaluating
                              it under a use_ema config evaluates a fresh
                              RANDOM-INIT EMA module (solver builds the EMA
                              before resume and only fills it from ckpt['ema'])
                              -> near-zero scores. Use raw-as-ema instead.
  raw-as-ema <ckpt> -o OUT    put the RAW weights into the ema slot — the
                              correct way to evaluate non-EMA weights through
                              the standard eval path.

    python tools/ckpt_forensics.py diff-ema output/<run>/checkpoint.pth --top 10
"""

from __future__ import annotations

import argparse

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("inspect")
    sp.add_argument("ckpt")

    sp = sub.add_parser("diff-ema")
    sp.add_argument("ckpt")
    sp.add_argument("--top", type=int, default=10)

    for name in ("strip-ema", "raw-as-ema"):
        sp = sub.add_parser(name)
        sp.add_argument("ckpt")
        sp.add_argument("-o", "--out", required=True)

    return p.parse_args()


def cmd_inspect(ck: dict) -> None:
    print("keys:", list(ck.keys()), "| last_epoch:", ck.get("last_epoch"))
    for slot in ("model", "ema"):
        if slot not in ck:
            continue
        state = ck[slot]["module"] if slot == "ema" else ck[slot]
        n_param = sum(v.numel() for v in state.values() if v.dtype.is_floating_point)
        print(f"{slot}: {len(state)} tensors, {n_param / 1e6:.1f}M float elements")


def cmd_diff_ema(ck: dict, top: int) -> None:
    m, e = ck["model"], ck["ema"]["module"]
    diffs = sorted(
        (
            (float((m[k].float() - e[k].float()).abs().max()), k)
            for k in m
            if k in e and m[k].dtype.is_floating_point
        ),
        reverse=True,
    )
    for d, k in diffs[:top]:
        print(f"maxdiff {d:.4f}  {k}")


def cmd_repack(ck: dict, out: str, raw_as_ema: bool) -> None:
    new = {"model": ck["model"], "last_epoch": ck.get("last_epoch")}
    if raw_as_ema:
        ema = ck.get("ema", {})
        new["ema"] = {
            "module": ck["model"],
            "updates": ema.get("updates", 0),
            "warmups": ema.get("warmups", 2000),
        }
    else:
        print(
            "WARNING: model-only checkpoint. Under a use_ema config the solver "
            "evaluates a random-init EMA module for this file -> near-zero "
            "scores. Prefer the raw-as-ema subcommand to evaluate raw weights."
        )
    torch.save(new, out)
    print(f"saved {out} (keys: {list(new.keys())})")


def main():
    args = parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    if args.cmd == "inspect":
        cmd_inspect(ck)
    elif args.cmd == "diff-ema":
        cmd_diff_ema(ck, args.top)
    else:
        cmd_repack(ck, args.out, raw_as_ema=(args.cmd == "raw-as-ema"))


if __name__ == "__main__":
    main()
