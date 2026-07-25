"""A6 efficiency benchmark — top-K latency/VRAM sweep on the v2 checkpoint.

For each K in {300, 150, 100, 50}:
    * Set `postprocessor.num_top_queries = K`.
    * Measure batch-1 inference latency (model forward + postprocess) on the
      first `--latency-imgs` IIT-AFF test images: median, p50, p95.
    * Track peak VRAM during inference.

No retraining and no F_β^w-vs-K sweep — the v2 model was trained with K=50
(its postprocessor default), so its F_β^w at K=300/150/100 is not a meaningful
operating point; what matters for the paper is the latency/memory tail at
batch-1. The v2 K=50 F_β^w (0.8094) is the headline number reported elsewhere.

Output: JSON + markdown table.

    python tools/bench_efficiency_topk.py \\
        --config configs/rtdetr/rtdetr_r50vd_6x_iit_v2.yml \\
        --resume output/rtdetr_r50vd_6x_iit_v2/checkpoint0071.pth \\
        --out output/rtdetr_r50vd_6x_iit_v2/efficiency_topk.json
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
    p.add_argument("--config", required=True)
    p.add_argument("--resume", required=True)
    p.add_argument("--data-root", default="./dataset/iit/data")
    p.add_argument("--ks", nargs="+", type=int, default=[300, 150, 100, 50])
    p.add_argument("--latency-imgs", type=int, default=200)
    p.add_argument("--warmup-imgs", type=int, default=10)
    p.add_argument("--out", default="output/rtdetr_r50vd_6x_iit_v2/efficiency_topk.json")
    return p.parse_args()


def load_test_images(data_root: str, n: int) -> list[tuple[str, np.ndarray]]:
    """Read the first n IIT-AFF test image stems and decode them once."""
    voc = os.path.join(data_root, "VOCdevkit2012/VOC2012")
    with open(os.path.join(voc, "ImageSets/Main/test.txt")) as f:
        ids = [x.strip() for x in f][:n]
    out = []
    for img_id in ids:
        path = os.path.join(voc, "JPEGImages", f"{img_id}.jpg")
        img = np.asarray(Image.open(path).convert("RGB"))
        out.append((img_id, img))
    return out


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

    # Load + preprocess `latency-imgs + warmup-imgs` images once, on GPU.
    raw = load_test_images(args.data_root, args.latency_imgs + args.warmup_imgs)
    pre = []
    for _img_id, rgb in raw:
        h0, w0 = rgb.shape[:2]
        # Mirror the eval transform: bilinear resize to 640x640, float / 255.
        pil = Image.fromarray(rgb).resize((640, 640), Image.BILINEAR)
        x = (
            torch.from_numpy(np.asarray(pil).copy())
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        orig_size = torch.tensor([[w0, h0]], dtype=torch.float32, device=device)
        pre.append((x, orig_size))

    rows: list[dict] = []
    for k in args.ks:
        print(f"\n=== K={k} ===")
        postproc.num_top_queries = k

        # Warm-up — cuDNN autotune, allocator priming.
        with torch.no_grad():
            for x, sz in pre[: args.warmup_imgs]:
                _ = postproc(model(x), sz)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        times_ms: list[float] = []
        with torch.no_grad():
            for x, sz in pre[args.warmup_imgs:]:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = postproc(model(x), sz)
                torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.asarray(times_ms)
        peak_mem_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
        row = {
            "K": k,
            "n_imgs": int(arr.size),
            "median_ms": float(np.median(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()),
            "throughput_imgs_per_s": float(1000.0 / arr.mean()),
            "peak_mem_MiB": float(peak_mem_mib),
        }
        print(json.dumps(row, indent=2))
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "checkpoint": args.resume,
                "Ks": args.ks,
                "rows": rows,
                "notes": (
                    "Batch-1 inference latency only. v2 was trained at K=50 so "
                    "K>50 does not improve F_beta^w (it only adds postprocessor "
                    "cost); we report latency/VRAM as a function of K to show "
                    "how cheap inference becomes once K is matched to the "
                    "training top-K."
                ),
            },
            f, indent=2,
        )
    print(f"\nSaved {args.out}")
    print("\n| K | median ms | p95 ms | throughput img/s | peak VRAM MiB |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['K']} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} | "
              f"{r['throughput_imgs_per_s']:.1f} | {r['peak_mem_MiB']:.0f} |")


if __name__ == "__main__":
    main()
