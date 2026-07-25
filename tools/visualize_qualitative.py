"""Render qualitative figures for an AffKernel checkpoint.

For a curated handful of IIT-AFF test images, lays out per-row:
    [ input image | GT affordance overlay | prediction overlay | pred boxes ]

The prediction column collapses per-query masks (top-K from the postprocessor)
into a single semantic affordance map by painting in descending detection-score
order: non-background pixels from each kept query overwrite a running canvas,
so higher-scoring detections take precedence on overlap. Boxes are drawn on
the rightmost column from the same kept detections.

Run from the repository root:

    python tools/visualize_qualitative.py \\
        --config configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml \\
        --resume weights/affkernel_iit_r50vd_stride2_deepsup_seed42.pth \\
        --out outputs/qualitative.png
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import YAMLConfig  # noqa: E402

# IIT-AFF taxonomy: 0=background then the 9 affordance classes.
AFFORDANCE_NAMES = [
    "background",
    "contain",
    "cut",
    "display",
    "engine",
    "grasp",
    "hit",
    "pound",
    "support",
    "w-grasp",
]
OBJECT_NAMES = [
    "__background__",
    "bowl", "tvm", "pan", "hammer", "knife",
    "cup", "drill", "racket", "spatula", "bottle",
]

# Tableau-style 9-class palette; index 0 is transparent (background).
# Picked to be print-friendly and distinguishable in greyscale-ish prints.
_AFF_PALETTE = np.array(
    [
        [0, 0, 0, 0],          # 0 background (transparent)
        [228, 26, 28, 200],    # 1 contain   red
        [55, 126, 184, 200],   # 2 cut       blue
        [77, 175, 74, 200],    # 3 display   green
        [152, 78, 163, 200],   # 4 engine    purple
        [255, 127, 0, 200],    # 5 grasp     orange
        [255, 255, 51, 220],   # 6 hit       yellow
        [166, 86, 40, 200],    # 7 pound     brown
        [247, 129, 191, 200],  # 8 support   pink
        [0, 206, 209, 200],    # 9 w-grasp   teal
    ],
    dtype=np.uint8,
)


def load_gt_affordance_map(mask_cache: str, img_id: str, h: int, w: int) -> np.ndarray:
    """Composite all per-instance GT segmask files for an image into one map.

    Each `<id>_<k>_segmask.sm` is a pickled HxW array with affordance class
    ids. Non-background pixels from later masks overwrite earlier ones (the
    same convention as the evaluator's union semantic map).
    """
    canvas = np.zeros((h, w), dtype=np.uint8)
    k = 1
    while True:
        path = os.path.join(mask_cache, f"{img_id}_{k}_segmask.sm")
        if not os.path.exists(path):
            break
        with open(path, "rb") as f:
            m = np.asarray(pickle.load(f), dtype=np.uint8)
        if m.shape == canvas.shape:
            nz = m > 0
            canvas[nz] = m[nz]
        k += 1
    return canvas


def paint_predicted_affordance_map(
    affordances: torch.Tensor, scores: torch.Tensor, score_thr: float = 0.5
) -> np.ndarray:
    """Reduce per-query [K, H, W] class labels to a single semantic map.

    Queries are painted in ascending score order so the highest-scoring
    detection ends up on top after overwrites. Non-background pixels from each
    query overwrite the running canvas.
    """
    aff = affordances.detach().cpu().numpy().astype(np.uint8)
    sc = scores.detach().cpu().numpy()
    order = np.argsort(sc)  # ascending; high scores written last
    canvas = np.zeros(aff.shape[1:], dtype=np.uint8)
    for idx in order:
        if sc[idx] < score_thr:
            continue
        m = aff[idx]
        nz = m > 0
        canvas[nz] = m[nz]
    return canvas


def overlay(img_rgb: np.ndarray, sem: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Alpha-blend a semantic affordance map on top of an RGB image."""
    overlay_rgba = _AFF_PALETTE[sem]                  # [H, W, 4]
    overlay_rgb = overlay_rgba[..., :3].astype(np.float32)
    a = (overlay_rgba[..., 3:4].astype(np.float32) / 255.0) * alpha
    base = img_rgb.astype(np.float32)
    out = base * (1.0 - a) + overlay_rgb * a
    return np.clip(out, 0, 255).astype(np.uint8)


def run_inference(model, postprocessor, img_rgb: np.ndarray, device):
    """Mirror the eval pipeline: resize to 640x640, forward, postprocess back."""
    h0, w0 = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb).resize((640, 640), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float().div(255.0)
    x = x.unsqueeze(0).to(device)
    orig_size = torch.tensor([[w0, h0]], dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(x)
        results = postprocessor(outputs, orig_size)
    return results[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", required=True)
    p.add_argument(
        "--data-root", default="./dataset/iit/data",
        help="IIT root containing VOCdevkit2012/ and cache/",
    )
    p.add_argument(
        "--ids", nargs="+", default=None,
        help="VOC stem ids to render. Defaults to a curated diverse set.",
    )
    p.add_argument("--out", default="outputs/qualitative.png")
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--max-boxes", type=int, default=6)
    return p.parse_args()


def _pick_default_ids(test_ids: list[str], data_root: str) -> list[str]:
    """Pick six DISTINCT test images, each "headlining" a different object class.

    For diversity we want each row of the figure to showcase a different
    object/affordance. We greedily assign one image to each class in
    `preferred` order, preferring images where that class is the *only*
    object (cleaner GT panel). If no single-class image exists, fall back to
    any image containing the class. The same image is never reused.
    """
    import xml.etree.ElementTree as ET
    anno_dir = os.path.join(data_root, "VOCdevkit2012/VOC2012/Annotations")
    preferred = ["bowl", "bottle", "hammer", "knife", "drill", "racket"]

    # First pass: build {object_name -> [ordered list of candidate ids]},
    # tagging single-class candidates first.
    candidates: dict[str, list[tuple[int, str]]] = {n: [] for n in preferred}
    for img_id in test_ids:
        xml = os.path.join(anno_dir, f"{img_id}.xml")
        if not os.path.exists(xml):
            continue
        names = [obj.find("name").text for obj in ET.parse(xml).getroot().iter("object")]
        names_set = set(names)
        for n in preferred:
            if n in names_set:
                # priority 0 = single-class image, 1 = multi-class
                prio = 0 if names_set == {n} else 1
                candidates[n].append((prio, img_id))

    used: set[str] = set()
    chosen: list[str] = []
    for n in preferred:
        # Sort by (priority, position-in-test-set) — single-class first.
        for _prio, img_id in sorted(candidates[n]):
            if img_id not in used:
                chosen.append(img_id)
                used.add(img_id)
                break
    return chosen


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = YAMLConfig(args.config, resume=args.resume)
    ckpt = torch.load(args.resume, map_location="cpu")
    # Prefer EMA weights — the eval pipeline uses them and so does the
    # reported F_beta^w = 0.8094.
    state = ckpt.get("ema", {}).get("module") or ckpt.get("model")
    model = cfg.model
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    postprocessor = cfg.postprocessor.to(device).eval()

    voc_root = os.path.join(args.data_root, "VOCdevkit2012/VOC2012")
    imgs_dir = os.path.join(voc_root, "JPEGImages")
    mask_cache = os.path.join(args.data_root, "cache/GTsegmask_VOC_2012_test")
    with open(os.path.join(voc_root, "ImageSets/Main/test.txt")) as f:
        test_ids = [x.strip() for x in f]

    ids = args.ids or _pick_default_ids(test_ids, args.data_root)
    print(f"Rendering {len(ids)} examples: {ids}")

    n = len(ids)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]
    for row, img_id in enumerate(ids):
        img_path = os.path.join(imgs_dir, f"{img_id}.jpg")
        img_rgb = np.asarray(Image.open(img_path).convert("RGB"))
        h, w = img_rgb.shape[:2]

        gt_sem = load_gt_affordance_map(mask_cache, img_id, h, w)
        result = run_inference(model, postprocessor, img_rgb, device)
        pred_sem = paint_predicted_affordance_map(
            result["affordances"], result["scores"], score_thr=args.score_thr
        )

        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"Input  ({img_id})", fontsize=10)
        axes[row, 1].imshow(overlay(img_rgb, gt_sem))
        axes[row, 1].set_title("Ground truth", fontsize=10)
        axes[row, 2].imshow(overlay(img_rgb, pred_sem))
        # Draw top boxes on the prediction panel for a complete qualitative view.
        boxes = result["boxes"].detach().cpu().numpy()
        scores_np = result["scores"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy()
        keep = np.where(scores_np >= args.score_thr)[0]
        keep = keep[np.argsort(-scores_np[keep])][: args.max_boxes]
        for k in keep:
            x1, y1, x2, y2 = boxes[k]
            axes[row, 2].add_patch(
                mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor="white", linewidth=1.5,
                )
            )
            cls = OBJECT_NAMES[int(labels[k])] if int(labels[k]) < len(OBJECT_NAMES) else str(int(labels[k]))
            axes[row, 2].text(
                x1, max(0, y1 - 4), f"{cls} {scores_np[k]:.2f}",
                color="white", fontsize=7,
                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"),
            )
        axes[row, 2].set_title("v2 prediction (F_β^w=0.81)", fontsize=10)
        for c in range(3):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

    # Shared legend for affordance classes.
    handles = [
        mpatches.Patch(color=_AFF_PALETTE[i, :3] / 255.0, label=AFFORDANCE_NAMES[i])
        for i in range(1, 10)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=9,
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.005),
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
