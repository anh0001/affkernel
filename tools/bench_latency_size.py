"""Latency/VRAM benchmark parameterized by input size (and optionally top-K).

Size-parameterized companion of ``bench_efficiency_topk.py``: measures batch-1
inference latency (model forward + postprocess) and peak VRAM at one or more
square input sizes, mirroring the eval transform (bilinear resize, /255).
Reference numbers from our runs (RTX 6000 Ada): 640 = 23.8 ms / 41.7 FPS and
800 = 32.9 ms / 30.3 FPS; scaling is sub-quadratic because the decoder and
postprocessor are resolution-independent.

    python tools/bench_latency_size.py \\
        --config configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml \\
        --resume output/rtdetr_r50vd_6x_iit_v3_stride2_deepsup/checkpoint.pth \\
        --sizes 640 704 800 --out output/latency_resolution_bench/bench.json

NOTE: eval_spatial_size (pos-embeds + anchors) must match --sizes for accuracy
measurements, but for pure latency the config's precomputed grids are only
correct at the config's own eval size; pass a probe config per size when the
timed forward must be numerically valid (see the deepsup_probe{704,800} yamls).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import YAMLConfig  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--resume", "-r", required=True)
    p.add_argument("--data-root", default="./dataset/iit/data")
    p.add_argument("--sizes", nargs="+", type=int, default=[640])
    p.add_argument("--k", type=int, default=50, help="postprocessor num_top_queries")
    p.add_argument("--latency-imgs", type=int, default=100)
    p.add_argument("--warmup-imgs", type=int, default=10)
    p.add_argument("--out", default="output/latency_size_bench.json")
    return p.parse_args()


def load_test_images(data_root: str, n: int) -> list:
    """Read the first n IIT-AFF test image stems and decode them once."""
    voc = os.path.join(data_root, "VOCdevkit2012", "VOC2012")
    with open(os.path.join(voc, "ImageSets", "Main", "test.txt")) as f:
        ids = [x.strip() for x in f][:n]
    out = []
    for img_id in ids:
        path = os.path.join(voc, "JPEGImages", f"{img_id}.jpg")
        out.append((img_id, np.asarray(Image.open(path).convert("RGB"))))
    return out


def preprocess(raw: list, size: int, device: torch.device) -> list:
    """Mirror the eval transform: bilinear square resize, float / 255."""
    pre = []
    for _, rgb in raw:
        h0, w0 = rgb.shape[:2]
        pil = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
        x = (
            torch.from_numpy(np.asarray(pil).copy())
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        pre.append((x, torch.tensor([[w0, h0]], dtype=torch.float32, device=device)))
    return pre


def bench_one(model, postproc, pre, warmup: int) -> dict:
    """Timed batch-1 loop over pre-decoded tensors; returns latency stats."""
    with torch.no_grad():
        for x, sz in pre[:warmup]:
            _ = postproc(model(x), sz)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times_ms = []
    with torch.no_grad():
        for x, sz in pre[warmup:]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = postproc(model(x), sz)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms)
    return {
        "n_imgs": int(arr.size),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "throughput_imgs_per_s": float(1000.0 / arr.mean()),
        "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = YAMLConfig(args.config, resume=args.resume)
    ckpt = torch.load(args.resume, map_location="cpu")
    state = ckpt.get("ema", {}).get("module") or ckpt.get("model")
    model = cfg.model
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    postproc = cfg.postprocessor.to(device).eval()
    postproc.num_top_queries = args.k

    raw = load_test_images(args.data_root, args.latency_imgs + args.warmup_imgs)

    rows = []
    for size in args.sizes:
        print(f"\n=== size={size} K={args.k} ===")
        row = {"size": size, "K": args.k}
        row.update(bench_one(model, postproc, preprocess(raw, size, device), args.warmup_imgs))
        print(json.dumps(row, indent=2))
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"checkpoint": args.resume, "rows": rows}, f, indent=2)
    print(f"\nSaved {args.out}")
    print("\n| size | median ms | p95 ms | throughput img/s | peak VRAM MiB |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['size']} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} | "
            f"{r['throughput_imgs_per_s']:.1f} | {r['peak_mem_MiB']:.0f} |"
        )


if __name__ == "__main__":
    main()
