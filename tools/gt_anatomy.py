"""CPU-only GT anatomy of an IIT-AFF split, per affordance class.

Pairs each instance segmask ``{id}_{k}_segmask.sm`` with the k-th XML object
and computes, per affordance class: carrier-object histogram, instances per
GT-present image, union area fraction, mean region thickness (2*mean EDT),
share of the carrier instance's labeled foreground, boundary-pixel share
(5-px erosion band), and multi-instance fraction. No GPU, no model — useful
for grounding class-difficulty arguments (thin-4 vs blob classes) in data.

    python tools/gt_anatomy.py --data-root ./dataset/iit/data --split test
"""

from __future__ import annotations

import argparse
import os
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from scipy import ndimage

OBJ = [
    "__background__",
    "bowl",
    "tvm",
    "pan",
    "hammer",
    "knife",
    "cup",
    "drill",
    "racket",
    "spatula",
    "bottle",
]
AFF = [
    "__background__",
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
# Object -> plausible affordance labels; used only to validate mask/XML pairing.
VALID = {
    "bowl": {1},
    "tvm": {3},
    "pan": {1, 5},
    "hammer": {5, 7},
    "knife": {2, 5},
    "cup": {1, 5},
    "drill": {4, 5},
    "racket": {5, 6},
    "spatula": {5, 8},
    "bottle": {1, 9},
}
EROSION_ITERS = 5  # boundary band width in px, matches the WFb sigma scale


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./dataset/iit/data")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "trainval"])
    return p.parse_args()


def main():
    args = parse_args()
    voc = os.path.join(args.data_root, "VOCdevkit2012", "VOC2012")
    cache = os.path.join(args.data_root, "cache", f"GTsegmask_VOC_2012_{args.split}")
    with open(os.path.join(voc, "ImageSets", "Main", f"{args.split}.txt")) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    print(f"{args.split} images: {len(ids)}")

    img_union = defaultdict(dict)  # cls -> {img: union bool mask}
    inst_count = defaultdict(lambda: defaultdict(int))  # cls -> img -> n instances
    carrier = defaultdict(lambda: defaultdict(int))  # cls -> obj -> count
    inst_share = defaultdict(list)  # cls -> share of instance fg
    pair_ok, pair_bad, n_inst_total = 0, 0, 0
    objcount_per_img = defaultdict(int)

    for iid in ids:
        xml_path = os.path.join(voc, "Annotations", f"{iid}.xml")
        names = []
        if os.path.exists(xml_path):
            names = [o.find("name").text for o in ET.parse(xml_path).findall("object")]
        k = 1
        while True:
            p = os.path.join(cache, f"{iid}_{k}_segmask.sm")
            if not os.path.exists(p):
                break
            with open(p, "rb") as f:
                m = pickle.load(f)
            n_inst_total += 1
            objcount_per_img[iid] += 1
            obj = names[k - 1] if k - 1 < len(names) else None
            present = [int(v) for v in np.unique(m) if v > 0]
            if obj is not None and obj in VALID:
                if set(present) <= VALID[obj]:
                    pair_ok += 1
                else:
                    pair_bad += 1
            fg_total = int((m > 0).sum())
            for a in present:
                cls = AFF[a]
                bm = m == a
                inst_count[cls][iid] += 1
                if obj:
                    carrier[cls][obj] += 1
                if fg_total:
                    inst_share[cls].append(bm.sum() / fg_total)
                u = img_union[cls].get(iid)
                img_union[cls][iid] = bm if u is None else (u | bm)
            k += 1

    print(f"instances: {n_inst_total}, pairing ok {pair_ok} / bad {pair_bad}")
    print(f"mean objects per image: {np.mean(list(objcount_per_img.values())):.2f}")
    print(
        f"{'class':>9} {'nImg':>5} {'inst/img':>8} {'multi%':>6} "
        f"{'area%':>6} {'thick_px':>8} {'bnd5%':>6} {'instShare':>9}  carriers"
    )
    for cls in AFF[1:]:
        imgs = img_union[cls]
        if not imgs:
            continue
        counts = [inst_count[cls][i] for i in imgs]
        multi = np.mean([c > 1 for c in counts]) * 100
        areas, thicks, bnds = [], [], []
        for _, u in imgs.items():
            a = u.sum()
            areas.append(a / u.size * 100)
            edt = ndimage.distance_transform_edt(u)
            thicks.append(2 * edt[u].mean() if a else 0.0)
            er = ndimage.binary_erosion(u, iterations=EROSION_ITERS)
            bnds.append(1 - er.sum() / max(a, 1))
        car = ",".join(f"{o}:{c}" for o, c in sorted(carrier[cls].items(), key=lambda x: -x[1]))
        print(
            f"{cls:>9} {len(imgs):>5} {np.mean(counts):>8.2f} {multi:>5.1f}% "
            f"{np.mean(areas):>5.2f}% {np.mean(thicks):>8.1f} "
            f"{np.mean(bnds) * 100:>5.1f}% {np.mean(inst_share[cls]):>9.2f}  {car}"
        )


if __name__ == "__main__":
    main()
