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


@dataclass(frozen=True)
class Detection:
    """A single detected object with its affordance mask."""

    label: int
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray  # [H, W] int class indices 0-9, original image resolution


class AffordanceModel:
    """Loads a trained checkpoint and runs affordance-aware inference."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        cfg = YAMLConfig(config_path, resume=checkpoint_path)
        if not cfg.yaml_cfg.get("use_affordance", False):
            raise ValueError("Config does not enable affordance head")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)

        self.model = cfg.model.deploy().to(self.device).eval()
        self.postprocessor = cfg.postprocessor.eval()  # non-deploy: list-of-dicts
        self._to_tensor = ToTensor()

    @torch.no_grad()
    def infer(self, rgb: np.ndarray, score_thresh: float = 0.6) -> list[Detection]:
        """Run inference on an HxWx3 uint8 RGB image.

        Returns detections sorted by descending score, each carrying a
        full-resolution affordance mask of class indices.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {rgb.shape}")
        orig_h, orig_w = rgb.shape[:2]

        from PIL import Image

        pil = Image.fromarray(rgb).resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        im = self._to_tensor(pil)[None].to(self.device)
        orig_size = torch.tensor([[orig_w, orig_h]], device=self.device)

        outputs = self.model(im)
        results = self.postprocessor(outputs, orig_size)
        result = results[0]

        labels = result["labels"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        boxes = result["boxes"].cpu().numpy()
        masks = (
            result["affordances"].cpu().numpy()
            if "affordances" in result
            else np.zeros((len(scores), orig_h, orig_w), dtype=np.int64)
        )

        detections: list[Detection] = []
        order = np.argsort(-scores)
        for i in order:
            if scores[i] < score_thresh:
                continue
            detections.append(
                Detection(
                    label=int(labels[i]),
                    score=float(scores[i]),
                    box_xyxy=tuple(float(v) for v in boxes[i]),
                    mask=masks[i].astype(np.int64),
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
    args = parser.parse_args()

    from PIL import Image

    images = collect_images(args.input)
    os.makedirs(args.output, exist_ok=True)
    model = AffordanceModel(args.config, args.resume, device=args.device)

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
