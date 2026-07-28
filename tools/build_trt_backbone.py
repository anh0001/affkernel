"""Export the AffKernel backbone to ONNX and build a TensorRT fp16 engine.

The affordance head's dynamic convolutions cannot be exported to ONNX, but the
backbone alone can — and it is the largest single term of the forward pass on
embedded GPUs. This builds a backbone-only engine that ``tools/infer.py
--trt-backbone <plan>`` will run in place of the PyTorch backbone, leaving the
encoder, decoder and affordance head untouched in PyTorch.

    python tools/build_trt_backbone.py \\
        -c configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml \\
        -r weights/affkernel_iit_r50vd_stride2_deepsup_seed42.pth \\
        --out weights/backbone_fp16.plan

Requires the TensorRT Python bindings and ``trtexec`` (both ship with JetPack;
on Jetson the bindings live in /usr/lib/python3.*/dist-packages, so run with
PYTHONPATH set if your virtualenv does not already see them). Engine files are
specific to the GPU, TensorRT version and input size they were built for.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig  # noqa: E402

DEFAULT_TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--resume", "-r", required=True)
    p.add_argument("--out", required=True, help="Destination .plan path")
    p.add_argument("--size", type=int, default=640, help="Square input size (default: 640)")
    p.add_argument("--trtexec", default=None, help="Path to trtexec (default: PATH, then %s)" % DEFAULT_TRTEXEC)
    p.add_argument("--keep-onnx", default=None, help="Also write the intermediate ONNX here")
    return p.parse_args()


def find_trtexec(explicit):
    for cand in (explicit, shutil.which("trtexec"), DEFAULT_TRTEXEC):
        if cand and os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "trtexec not found; pass --trtexec explicitly (JetPack ships it at %s)" % DEFAULT_TRTEXEC
    )


def main():
    args = parse_args()
    trtexec = find_trtexec(args.trtexec)

    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state, strict=False)
    backbone = model.deploy().backbone.cuda().float().eval()

    dummy = torch.zeros(1, 3, args.size, args.size, device="cuda")
    with torch.no_grad():
        feats = backbone(dummy)
    out_names = [f"feat{i}" for i in range(len(feats))]
    print(f"backbone outputs: {[tuple(f.shape) for f in feats]}")

    tmpdir = tempfile.mkdtemp(prefix="affkernel_trt_")
    onnx_path = args.keep_onnx or os.path.join(tmpdir, "backbone.onnx")
    torch.onnx.export(
        backbone, dummy, onnx_path,
        input_names=["images"], output_names=out_names,
        opset_version=17, do_constant_folding=True,
    )
    print(f"exported ONNX -> {onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # fp16 I/O keeps the engine's tensors in the same dtype as the PyTorch tail,
    # so no per-frame conversion is needed on either boundary.
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={args.out}",
        "--fp16",
        "--inputIOFormats=fp16:chw",
        "--outputIOFormats=" + ",".join(["fp16:chw"] * len(out_names)),
    ]
    print("building engine (this takes a few minutes):\n  " + " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        raise SystemExit(f"trtexec failed with exit code {proc.returncode}")
    for line in proc.stdout.splitlines():
        if "Engine built" in line or "GPU Compute Time" in line:
            print(line.strip())
    if not args.keep_onnx:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
