"""Perception-only, non-circular grasp-point evaluation on IIT-AFF.

For each test image we define the INDEPENDENT ground-truth grasp target from the
IIT-AFF GT affordance masks (largest 'grasp'/'w-grasp' connected component
centroid) -- NOT from any model output. We then compare three ways of choosing a
grasp point from an AffKernel detection:

  * affordance  : centroid of the predicted grasp/w-grasp mask  (our method)
  * bbox        : detection bounding-box center                  (baseline)
  * random_bbox : a seeded random point inside the bbox          (baseline)

Metrics (per method):
  * point-in-GT-mask %     : predicted point lies inside the GT grasp mask
  * normalized distance    : ||p - gt_target|| / bbox_diagonal  (mean / median)

This is a pure-perception evaluation: it needs only the IIT-AFF test split and a
trained checkpoint, with no simulator or robot in the loop.

    python tools/eval_grasp_point.py -c configs/rtdetr/<config>.yml \\
        -r output/<run>/checkpoint.pth --num 300
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.infer import GRASP_CLASS, WGRASP_CLASS, AffordanceModel  # noqa: E402, I001

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(REPO, "dataset/iit/data")
GRASP = (GRASP_CLASS, WGRASP_CLASS)


def largest_component_centroid(binary):
    """Centroid (u,v) of the largest connected component of a binary mask."""
    if not binary.any():
        return None
    try:
        from scipy import ndimage

        lab, n = ndimage.label(binary)
        if n > 1:
            sizes = ndimage.sum(binary, lab, range(1, n + 1))
            binary = lab == (int(np.argmax(sizes)) + 1)
    except ImportError:
        pass
    ys, xs = np.nonzero(binary)
    return float(xs.mean()), float(ys.mean())


def load_gt_grasp_mask(mask_dir, img_id, h, w):
    """Union of GT grasp/w-grasp pixels across all instance masks for an image."""
    gt = np.zeros((h, w), dtype=bool)
    i = 1
    found = False
    while True:
        p = os.path.join(mask_dir, f"{img_id}_{i}_segmask.sm")
        if not os.path.exists(p):
            break
        with open(p, "rb") as f:
            m = np.asarray(pickle.load(f))
        if m.shape == (h, w):
            gt |= np.isin(m, GRASP)
            found = True
        i += 1
    return gt if found else None


def main():
    ap = argparse.ArgumentParser(
        description="Perception-only grasp-point evaluation on the IIT-AFF test split: "
                    "compares affordance-mask, bbox-center and random-in-bbox grasp "
                    "points against GT grasp masks."
    )
    ap.add_argument("--config", "-c", required=True, help="Path to the model YAML config")
    ap.add_argument("--resume", "-r", required=True, help="Path to the trained checkpoint")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="IIT-AFF data root containing VOCdevkit2012/ and cache/ "
                         "(default: dataset/iit/data)")
    ap.add_argument("--num", type=int, default=300,
                    help="Number of test images to evaluate (default: 300)")
    ap.add_argument("--score", type=float, default=0.5,
                    help="Minimum detection score to keep (default: 0.5)")
    ap.add_argument("--device", default=None,
                    help="Torch device, e.g. cuda / cuda:0 / cpu (default: auto)")
    args = ap.parse_args()

    voc = os.path.join(args.data_root, "VOCdevkit2012", "VOC2012")
    imgs_dir = os.path.join(voc, "JPEGImages")
    test_list = os.path.join(voc, "ImageSets", "Main", "test.txt")
    mask_dir = os.path.join(args.data_root, "cache", "GTsegmask_VOC_2012_test")
    for required in (imgs_dir, test_list, mask_dir):
        if not os.path.exists(required):
            raise FileNotFoundError(
                f"Missing IIT-AFF path: {required}. Pass --data-root to point at your copy."
            )

    with open(test_list) as f:
        ids = [x.strip() for x in f]
    rng = np.random.default_rng(0)
    model = AffordanceModel(args.config, args.resume, device=args.device)

    methods = ["affordance", "bbox", "random_bbox"]
    inmask = {m: [] for m in methods}
    ndist = {m: [] for m in methods}
    n_eval = 0
    n_no_gt = 0
    n_no_det = 0

    for img_id in ids:
        if n_eval >= args.num:
            break
        ip = os.path.join(imgs_dir, f"{img_id}.jpg")
        if not os.path.exists(ip):
            continue
        rgb = np.array(Image.open(ip).convert("RGB"))
        h, w = rgb.shape[:2]
        gt_mask = load_gt_grasp_mask(mask_dir, img_id, h, w)
        if gt_mask is None or not gt_mask.any():
            n_no_gt += 1
            continue
        gt_target = np.array(largest_component_centroid(gt_mask))

        dets = model.infer(rgb, score_thresh=args.score)
        det = next((d for d in dets
                    if np.isin(d.mask, GRASP).any()), None)
        if det is None:
            n_no_det += 1
            continue

        x1, y1, x2, y2 = det.box_xyxy
        diag = float(np.hypot(x2 - x1, y2 - y1)) + 1e-6
        pts = {
            "affordance": np.array(largest_component_centroid(np.isin(det.mask, GRASP))),
            "bbox": np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
            "random_bbox": np.array([rng.uniform(x1, x2), rng.uniform(y1, y2)]),
        }
        for m, p in pts.items():
            pu, pv = int(round(p[0])), int(round(p[1]))
            inside = (0 <= pv < h and 0 <= pu < w and gt_mask[pv, pu])
            inmask[m].append(bool(inside))
            ndist[m].append(float(np.linalg.norm(p - gt_target) / diag))
        n_eval += 1

    print(f"\nEvaluated {n_eval} images "
          f"(skipped {n_no_gt} no-GT-grasp, {n_no_det} no-grasp-detection)\n")
    print(f"{'method':<13}{'point-in-GT %':>15}{'norm.dist mean':>16}{'norm.dist median':>18}")
    for m in methods:
        hit = 100 * np.mean(inmask[m]) if inmask[m] else 0
        print(f"{m:<13}{hit:>14.1f}{np.mean(ndist[m]):>16.3f}{np.median(ndist[m]):>18.3f}")


if __name__ == "__main__":
    main()
