"""Standalone AffKernel inference entry point.

Loads a trained RT-DETR + affordance checkpoint and runs RGB images through the
model + postprocessor, returning per-detection boxes, labels, scores and
full-resolution 10-class affordance masks.

Importable as a library (``AffordanceModel``) or runnable as a CLI on a single
image or a folder of images:

    python tools/infer.py -c configs/rtdetr/<config>.yml -r <checkpoint.pth> \\
        --input path/to/image_or_dir --output outputs/infer

Note the dynamic-kernel affordance head cannot be exported to ONNX, so this
PyTorch path is the supported way to run affordance inference.

IIT-AFF affordance classes (mask values 0-9):
    0 __background__  1 contain  2 cut      3 display  4 engine
    5 grasp           6 hit      7 pound    8 support  9 w-grasp
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms import ToTensor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig  # noqa: E402

AFFORDANCE_CLASSES = (
    "__background__", "contain", "cut", "display", "engine",
    "grasp", "hit", "pound", "support", "w-grasp",
)
GRASP_CLASS = 5
WGRASP_CLASS = 9
MODEL_INPUT_SIZE = 640
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Overlay colour per affordance class id (index 0 = background, never drawn).
AFFORDANCE_PALETTE = (
    (0, 0, 0),
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (170, 110, 40),
)
OVERLAY_ALPHA = 110


class _GraphedForward:
    """Wraps a fixed-input-shape model forward in a captured CUDA graph.

    Replaying a captured graph skips per-kernel launch overhead, which
    dominates the decoder cost on Jetson-class devices. The capture is
    bit-exact vs the eager forward at the same dtype.
    """

    def __init__(self, model, input_size: int, dtype: torch.dtype):
        self.model = model
        self.static_in = torch.zeros(
            1, 3, input_size, input_size, device="cuda", dtype=dtype
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):  # warmup allocations outside the graph
                model(self.static_in)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.static_out = self.model(self.static_in)

    def __call__(self, x: torch.Tensor):
        self.static_in.copy_(x)
        self.graph.replay()
        return self.static_out


class _TRTBackboneForward:
    """TensorRT backbone + CUDA-graphed PyTorch tail.

    The engine writes into persistent buffers, so the encoder / decoder /
    affordance head can be captured once against those buffers and replayed:
    per frame this is one engine enqueue plus one graph replay.
    """

    def __init__(self, model, plan_path: str):
        from src.zoo.rtdetr.trt_backbone import TRTBackbone

        self.model = model
        self.backbone = TRTBackbone(plan_path)
        n_enc = len(model.encoder.in_channels)

        def tail():
            feats = self.backbone.outputs
            # Extra finest level(s) feed the affordance branch only, mirroring
            # RTDETR.forward's routing.
            low_level_feat = feats[0] if len(feats) > n_enc else None
            encoder_output = model.encoder(list(feats[-n_enc:]))
            decoder_output = model.decoder(encoder_output, None)
            out = {
                "pred_logits": decoder_output["pred_logits"],
                "pred_boxes": decoder_output["pred_boxes"],
            }
            aff = model.affordance_branch(
                decoder_output["features"], encoder_output,
                low_level_feat=low_level_feat,
            )
            out["aff_feat"] = aff["aff_feat"]
            out["aff_kernel"] = aff["aff_kernel"]
            out["aff_meta"] = aff["aff_meta"]
            out["aff_head_type"] = aff.get("aff_head_type", "dynamic")
            out["aff_box_relative"] = aff.get("aff_box_relative", False)
            return out

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                tail()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.static_out = tail()

    @property
    def input_dtype(self) -> torch.dtype:
        return self.backbone.input_dtype

    def __call__(self, x: torch.Tensor):
        self.backbone(x)
        self.graph.replay()
        return self.static_out


@dataclass(frozen=True)
class Detection:
    """A single detected object with its affordance mask."""

    label: int
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray  # [H, W] int class indices 0-9, original image resolution


class AffordanceModel:
    """Loads a trained checkpoint and runs affordance-aware inference.

    Args:
        half: run the model in fp16 (about half the latency and a third of the
            VRAM on Jetson-class GPUs; sub-0.1% output drift).
        cudagraph: capture the fixed-shape forward pass in a CUDA graph,
            eliminating per-kernel launch overhead (bit-exact vs eager at the
            same dtype). Requires CUDA.
    """

    def __init__(self, config_path: str, checkpoint_path: str, device: str | None = None,
                 half: bool = False, cudagraph: bool = False, gpu_preprocess: bool = False,
                 trt_backbone: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        cfg = YAMLConfig(config_path, resume=checkpoint_path)
        if not cfg.yaml_cfg.get("use_affordance", False):
            raise ValueError("Config does not enable affordance head")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)

        dtype = torch.half if half else torch.float32
        model = cfg.model.deploy().to(self.device, dtype).eval()
        if half:
            # Module.to(dtype) skips plain-attribute tensors (precomputed
            # pos-embeds / anchors); align their dtype and device too.
            for mod in model.modules():
                for name, val in vars(mod).items():
                    if isinstance(val, torch.Tensor):
                        tgt = torch.half if val.dtype == torch.float32 else val.dtype
                        setattr(mod, name, val.to(self.device, tgt))
        self._dtype = dtype
        self.model = model
        if trt_backbone:
            if self.device.type != "cuda":
                raise ValueError("trt_backbone requires a CUDA device")
            self.model = _TRTBackboneForward(model, trt_backbone)
            # The engine fixes the I/O dtype; match the preprocess to it.
            self._dtype = self.model.input_dtype
        elif cudagraph:
            if self.device.type != "cuda":
                raise ValueError("cudagraph=True requires a CUDA device")
            self.model = _GraphedForward(model, MODEL_INPUT_SIZE, dtype)
        self.postprocessor = cfg.postprocessor.eval()  # non-deploy: list-of-dicts
        self._base_aff_thresh = getattr(
            self.postprocessor, "affordance_score_thresh", 0.5
        )
        self._to_tensor = ToTensor()
        # GPU preprocess: upload the raw uint8 frame and resize with bilinear
        # interpolation on-device instead of PIL bicubic on CPU. Matches the
        # eval transform's bilinear convention; saves ~10 ms/frame on Jetson.
        self._gpu_preprocess = gpu_preprocess

    @torch.no_grad()
    def infer(self, rgb: np.ndarray, score_thresh: float = 0.6) -> list[Detection]:
        """Run inference on an HxWx3 uint8 RGB image.

        Returns detections sorted by descending score, each carrying a
        full-resolution affordance mask of class indices.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {rgb.shape}")
        orig_h, orig_w = rgb.shape[:2]

        if self._gpu_preprocess:
            t = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
            im = t.permute(2, 0, 1)[None].float().div_(255.0)
            im = torch.nn.functional.interpolate(
                im, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                mode="bilinear", align_corners=False,
            ).to(self._dtype)
        else:
            from PIL import Image

            pil = Image.fromarray(rgb).resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
            im = self._to_tensor(pil)[None].to(self.device, self._dtype)
        orig_size = torch.tensor([[orig_w, orig_h]], device=self.device)

        # Don't pay to decode and upsample masks for detections this call is
        # about to discard: raise the postprocessor's affordance gate to the
        # caller's score threshold. Clamped below by the configured gate, so
        # this only ever removes work, never output.
        self.postprocessor.affordance_score_thresh = max(score_thresh, self._base_aff_thresh)

        outputs = self.model(im)
        results = self.postprocessor(outputs, orig_size)
        result = results[0]

        labels = result["labels"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        boxes = result["boxes"].cpu().numpy()
        # Copy ONLY the masks of detections that pass the score threshold, as
        # uint8 (class ids fit in a byte). Filtering + narrowing on-GPU cuts
        # the device-to-host transfer from top_k x H x W x 8 bytes to
        # n_kept x H x W x 1 byte per frame.
        kept = np.nonzero(scores >= score_thresh)[0]
        if "affordances" in result and kept.size > 0:
            kept_t = torch.as_tensor(kept, device=result["affordances"].device)
            kept_masks = (
                result["affordances"][kept_t].to(torch.uint8).cpu().numpy()
            )
            masks = dict(zip(kept.tolist(), kept_masks))
        else:
            masks = {}
        empty = np.zeros((orig_h, orig_w), dtype=np.int64)

        detections: list[Detection] = []
        order = kept[np.argsort(-scores[kept])]
        for i in order:
            detections.append(
                Detection(
                    label=int(labels[i]),
                    score=float(scores[i]),
                    box_xyxy=tuple(float(v) for v in boxes[i]),
                    mask=masks[int(i)].astype(np.int64) if int(i) in masks else empty,
                )
            )
        return detections


def grasp_point_from_mask(
    mask: np.ndarray, grasp_classes: tuple[int, ...] = (GRASP_CLASS, WGRASP_CLASS)
) -> tuple[int, int] | None:
    """Return the (u, v) pixel centroid of the grasp-affordance region, or None.

    Picks the largest connected blob's centroid if SciPy is available, else the
    raw centroid of all grasp-class pixels.
    """
    grasp_mask = np.isin(mask, grasp_classes)
    if not grasp_mask.any():
        return None
    try:
        from scipy import ndimage

        labeled, n = ndimage.label(grasp_mask)
        if n > 1:
            sizes = ndimage.sum(grasp_mask, labeled, range(1, n + 1))
            grasp_mask = labeled == (int(np.argmax(sizes)) + 1)
    except ImportError:
        pass
    ys, xs = np.nonzero(grasp_mask)
    return int(round(xs.mean())), int(round(ys.mean()))


def render(rgb: np.ndarray, detections: list):
    """Draw boxes plus per-class affordance overlays, returning a PIL image."""
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(rgb).convert("RGB")
    for det in detections:
        for class_id in np.unique(det.mask):
            if class_id <= 0:
                continue
            pixels = det.mask == class_id
            colour = AFFORDANCE_PALETTE[int(class_id) % len(AFFORDANCE_PALETTE)]
            overlay = Image.new("RGBA", canvas.size, colour + (OVERLAY_ALPHA,))
            canvas.paste(overlay, (0, 0), Image.fromarray((pixels * 255).astype(np.uint8), "L"))
    draw = ImageDraw.Draw(canvas)
    for det in detections:
        draw.rectangle(list(det.box_xyxy), outline=(255, 255, 255), width=2)
        draw.text((det.box_xyxy[0] + 2, det.box_xyxy[1] + 2),
                  f"{det.label}:{det.score:.2f}", fill=(255, 255, 0))
    return canvas


def collect_images(input_path: str) -> list:
    """Return the list of image paths named by a file or directory argument."""
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        found = sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.lower().endswith(IMAGE_SUFFIXES)
        )
        if not found:
            raise FileNotFoundError(f"No images with suffixes {IMAGE_SUFFIXES} in {input_path}")
        return found
    raise FileNotFoundError(f"--input does not exist: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AffKernel affordance inference on an image or a folder of images."
    )
    parser.add_argument("--config", "-c", required=True, help="Path to the model YAML config")
    parser.add_argument("--resume", "-r", required=True, help="Path to the trained checkpoint")
    parser.add_argument("--input", required=True, help="Input image file or directory of images")
    parser.add_argument("--output", default="outputs/infer",
                        help="Directory for annotated images (created if missing)")
    parser.add_argument("--score-thr", type=float, default=0.6,
                        help="Minimum detection score to keep (default: 0.6)")
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. cuda / cuda:0 / cpu (default: auto)")
    parser.add_argument("--half", action="store_true",
                        help="Run the model in fp16 (faster on Jetson/embedded GPUs)")
    parser.add_argument("--cudagraph", action="store_true",
                        help="Capture the forward pass in a CUDA graph (bit-exact, lower latency)")
    parser.add_argument("--gpu-preprocess", action="store_true",
                        help="Resize/normalize on the GPU (bilinear) instead of PIL on CPU")
    parser.add_argument("--trt-backbone", default=None, metavar="PLAN",
                        help="Run the backbone as a TensorRT engine built by "
                             "tools/build_trt_backbone.py (implies fp16 + CUDA graph)")
    args = parser.parse_args()

    from PIL import Image

    images = collect_images(args.input)
    os.makedirs(args.output, exist_ok=True)
    model = AffordanceModel(args.config, args.resume, device=args.device,
                            half=args.half or bool(args.trt_backbone),
                            cudagraph=args.cudagraph,
                            gpu_preprocess=args.gpu_preprocess,
                            trt_backbone=args.trt_backbone)

    for image_path in images:
        rgb = np.array(Image.open(image_path).convert("RGB"))
        detections = model.infer(rgb, score_thresh=args.score_thr)
        out_path = os.path.join(
            args.output, os.path.splitext(os.path.basename(image_path))[0] + "_aff.png"
        )
        render(rgb, detections).save(out_path)
        print(f"{image_path}: {len(detections)} detection(s) -> {out_path}")
        for det in detections:
            grasp = grasp_point_from_mask(det.mask)
            classes = sorted({int(c) for c in np.unique(det.mask) if c > 0})
            names = ", ".join(AFFORDANCE_CLASSES[c] for c in classes) or "none"
            print(f"    label={det.label} score={det.score:.3f} "
                  f"affordances=[{names}] grasp_point={grasp}")


if __name__ == "__main__":
    main()
