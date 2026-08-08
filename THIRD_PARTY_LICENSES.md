# Third-Party Licenses and Attribution

AffKernel's own contributions are released under the MIT License (see
[`LICENSE`](LICENSE)). A substantial part of the detector, however, is derived
from two Apache-2.0 projects, and those portions remain under the Apache
License, Version 2.0.

The full Apache-2.0 text is included at
[`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt). Attribution notices are
in [`NOTICE`](NOTICE). This document is the per-file inventory required by
Apache-2.0 sections 4(b) and 4(c): it records which files are derived, from
where, and that they have been modified.

## Summary

| Upstream | License | Role in AffKernel |
|---|---|---|
| [RT-DETR](https://github.com/lyuwenyu/RT-DETR) (lyuwenyu, 2023) | Apache-2.0 | Detector: backbone, hybrid encoder, transformer decoder, denoising, config system, solver, EMA, training entry point |
| [DETR](https://github.com/facebookresearch/detr) (Facebook, Inc.) | Apache-2.0 | Detection engine, COCO evaluation, distributed and logging utilities, box operations (mostly inherited via RT-DETR) |
| [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause | Reference implementations echoed in a few utility helpers; imported as a dependency, not vendored |

**All listed files have been modified** relative to their upstream originals.
Modifications include the affordance branch and its dynamic-convolution kernel
readout, the affordance loss terms and Hungarian matching cost, the IIT-AFF and
UMD dataset pipelines, the weighted F-measure evaluators, the stride-2 mask
readout, deep supervision, and assorted configuration changes.

## Files derived from RT-DETR (Apache-2.0)

Copyright (c) 2023 lyuwenyu.

- `src/core/__init__.py`
- `src/core/config.py`
- `src/core/yaml_config.py`
- `src/core/yaml_utils.py`
- `src/misc/dist.py`
- `src/misc/visualizer.py`
- `src/nn/backbone/common.py`
- `src/nn/backbone/presnet.py`
- `src/nn/backbone/utils.py`
- `src/optim/ema.py`
- `src/solver/__init__.py`
- `src/solver/solver.py`
- `src/solver/det_engine.py`
- `src/solver/det_solver.py`
- `src/zoo/rtdetr/__init__.py`
- `src/zoo/rtdetr/rtdetr.py`
- `src/zoo/rtdetr/hybrid_encoder.py`
- `src/zoo/rtdetr/denoising.py`
- `src/zoo/rtdetr/utils.py`
- `tools/train.py`

Additionally derived from RT-DETR, though their upstream headers were not
preserved in the fork this repository grew out of:

- `src/zoo/rtdetr/rtdetr_decoder.py`
- `src/zoo/rtdetr/rtdetr_criterion.py`
- `src/zoo/rtdetr/matcher.py`
- `src/data/transforms.py`
- `src/optim/optim.py`

`src/nn/backbone/presnet.py` additionally downloads ImageNet-pretrained
ResNet-vd weights at runtime from the RT-DETR authors' release artefacts
(`github.com/lyuwenyu/storage`). Those weights are fetched by the user's
machine on first run; they are not redistributed in this repository.

## Files derived from DETR (Apache-2.0)

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

- `src/data/coco/coco_dataset.py`
- `src/data/coco/coco_eval.py`
- `src/misc/dist.py`
- `src/misc/logger.py`
- `src/nn/backbone/common.py`
- `src/solver/det_engine.py`
- `src/zoo/rtdetr/box_ops.py`

Several of these reached AffKernel by way of RT-DETR, which had already
modified them; they are therefore attributable to both upstreams and are listed
in both sections where that applies.

## Further upstreams reached through RT-DETR

These are transitive lineages that the sections above did not name explicitly.
No code was taken from them directly; each arrived through RT-DETR.

- **Deformable-DETR** (SenseTime Research, Apache-2.0). The deformable-attention
  implementation behind `MSDeformableAttention` in
  `src/zoo/rtdetr/rtdetr_decoder.py` and `deformable_attention_core_func` in
  `src/zoo/rtdetr/utils.py`. Apache-2.0, so covered by the same terms as the
  RT-DETR material above.
- **Ultralytics YOLOv5** (GPL-3.0, later AGPL-3.0). `src/optim/ema.py` carries an
  upstream reference to YOLOv5's `torch_utils.py` for the exponential
  moving-average helper, by way of RT-DETR. The file implements a standard EMA
  update; the lineage is recorded here so it is disclosed explicitly rather than
  left implied by a source comment.

## Files derived from Fast R-CNN / py-faster-rcnn (MIT)

Copyright (c) 2015 Microsoft Corporation, Ross Girshick.

- `src/data/iit/iit_eval.py` - the `parse_rec()` and `voc_ap()` helpers,
  including the `use_07_metric` 11-point interpolation branch, are the classic
  PASCAL VOC evaluation routines. MIT, and therefore compatible with this
  repository's own MIT licence.

## Datasets

Neither dataset is redistributed by this repository, and neither is covered by
the licences above. Obtain each from its original source and honour its terms:

- **IIT-AFF** (Nguyen et al., IROS 2017). No licence is stated by its authors;
  they request citation of the original paper.
  <https://sites.google.com/site/iitaffdataset/>
- **UMD Part-Affordance** (Myers et al., ICRA 2015).
  <https://users.umiacs.umd.edu/~amyers/part-affordance-dataset/>

See [`docs/datasets.md`](docs/datasets.md) for acquisition instructions and the
BibTeX entries their authors ask you to use.

## Reporting an attribution problem

If you believe code in this repository is insufficiently attributed, or is
used in a way its licence does not permit, please open an issue at
<https://github.com/anh0001/affkernel/issues>. Attribution corrections are
treated as bugs and will be fixed promptly.
