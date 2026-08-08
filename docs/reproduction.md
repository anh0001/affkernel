# Reproduction Guide

Every command needed to regenerate the published numbers. All commands are run
from the repository root, in the environment described in the
[README](../README.md#installation), with the dataset laid out as described in
[`datasets.md`](datasets.md).

Reference hardware for all timing numbers: one NVIDIA RTX 6000 Ada, fp32.

---

## 1. The released model

The headline configuration is
`configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml`: a stride-2
affordance readout with deep supervision, trained for 72 epochs.

### Train

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml \
  --seed 42
```

About 15.5 hours. Checkpoints land in the config's `output_dir`.

> **`--seed` has no default.** If you omit it, `set_seed` is never called and
> the run is not reproducible. The three published seeds are 42, 7 and 123.

Multi-GPU (`sync_bn` only takes effect under DDP):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=8989 \
  tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml --seed 42
```

The seed-variant configs `..._deepsup_seed_7.yml` and `..._deepsup_seed_123.yml`
differ from the base config **only** in `output_dir`; the seed still comes from
`--seed`.

### Evaluate

`--test-only` evaluates the **EMA** weights and reports the AffordanceNet-lineage
convention, `beta^2 = 0.3`:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml \
  -r output/rtdetr_r50vd_6x_iit_v3_stride2_deepsup/checkpoint.pth \
  --test-only
```

Expected: `mean F_beta^w for Affordance Masks: 0.8582`.

For the primary reported metric (`beta^2 = 1`) plus the miss/quality
decomposition:

```bash
python tools/decompose_fbw_gap.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml \
  -r output/rtdetr_r50vd_6x_iit_v3_stride2_deepsup/checkpoint.pth \
  --beta2 1.0
```

Expected: `0.8685` overall, `0.8933` on fired (detected) instances.

Both conventions from a single forward pass:

```bash
python tools/rescore_fbw_beta.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml \
  -r output/rtdetr_r50vd_6x_iit_v3_stride2_deepsup/checkpoint.pth \
  --betas 0.3 1.0
```

### Per-seed results

| Seed | `F_beta^w` (beta^2=0.3) | `F_beta^w` (beta^2=1) | Fired-instance (beta^2=1) |
|---|---:|---:|---:|
| 42 | 0.8582 | 0.8685 | 0.8933 |
| 7 | 0.8567 | 0.8668 | 0.8945 |
| 123 | 0.8574 | 0.8673 | 0.8914 |
| **mean** | **0.8574 ± 0.0008** | **0.8675 ± 0.0009** | 0.8931 |

---

## 2. On the beta convention

This is the single most important protocol detail when comparing across papers.

The weighted F-measure of Margolin et al. takes a `beta` parameter. Two
conventions are in circulation:

- **`beta^2 = 0.3`** weights precision more heavily. Inherited from the released
  AffordanceNet evaluation code rather than stated in that paper, and carried
  forward by its descendants.
- **`beta^2 = 1`** weights precision and recall equally. Used by recent
  transformer baselines.

They are not interchangeable, and the gap between them is larger than most of
the differences being argued about in this literature (for this model, 0.8574
vs 0.8675). **Never compare a number computed at one convention against a
number computed at the other.** This repository reports both, and
`tools/rescore_fbw_beta.py` lets you recompute either from one pass so that any
comparison can be made at a matched convention.

Where a peer's convention is not stated in their paper, treat the comparison as
reconstructed rather than exact.

---

## 3. Latency and throughput

```bash
CUDA_VISIBLE_DEVICES=0 python tools/bench_latency_size.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml \
  -r output/rtdetr_r50vd_6x_iit_v3_stride2_deepsup/checkpoint.pth \
  --sizes 640 --latency-imgs 100 --warmup-imgs 10 \
  --out outputs/bench.json
```

Expected at 640x640: median 16.3 ms, p95 16.4 ms, 61.5 img/s, peak 442 MiB.
(The superseded code path measured 24.1 ms and 1319 MiB at stride-2; if you see
those figures you are not running the current path.)

Query-budget sweep (the top-*K* efficiency table):

```bash
python tools/bench_efficiency_topk.py -c <config> -r <checkpoint>
```

> For a numerically valid forward pass at a resolution other than 640, use the
> matching probe config (`..._deepsup_probe704.yml`, `..._probe800.yml`). The
> config's `eval_spatial_size` bakes in positional embeddings and anchors, so
> simply passing a different `--sizes` value changes timing only, not a valid
> accuracy measurement.

---

## 4. The readout-resolution ladder

The detector is identical across these rows. Only the affordance readout stride
changes, so the comparison isolates readout resolution from detector capacity.

| Row | Config | `F_beta^w` (beta^2=1) |
|---|---|---:|
| stride-8 | `rtdetr_r50vd_6x_iit_v3.yml` | 0.7518 |
| stride-4 | `rtdetr_r50vd_6x_iit_v3_stride2.yml` with `mask_upsample_factor: 2` | 0.8269 |
| stride-2 | `rtdetr_r50vd_6x_iit_v3_stride2.yml` | 0.8542 ± 0.0016 |
| stride-2 + deep sup. | `rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml` | 0.8675 ± 0.0009 |

The knob is `AffordanceBranch.mask_upsample_factor`: 1 = stride-8, 2 = stride-4,
4 = stride-2. Raising it from stride-4 to stride-2 costs +1.1 ms (+7%) and
+21 MiB, which is small relative to the accuracy it buys, and that is the point
of the result.

---

## 5. UMD Part-Affordance

Train (about one day wall-clock, including per-epoch evaluation over 14,020
test frames):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_umd.yml --seed 42
```

Evaluate:

```bash
python tools/decompose_fbw_gap.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_umd.yml \
  -r output/rtdetr_r50vd_6x_umd/checkpoint.pth --beta2 1.0
```

Expected: 0.8752 ± 0.0049 on the full test split, 0.8799 ± 0.0048 on the
human-annotated subset. Those are three-seed means (42, 7, 123); the single
seed-42 command above reproduces one draw from that spread, not the mean.

The human-annotated subset is identified as every third frame; that selection
is a documented approximation of the original protocol. Peer UMD numbers in the
literature come from a compilation whose split, ground-truth handling and beta
convention are not stated, so our rows use a reconstructed AffordanceNet-lineage
protocol and cross-paper UMD comparisons carry that caveat.

Note that `tools/rescore_fbw_beta.py` is IIT-only and will reject a UMD dataset.

---

## 6. Grasp-point actionability

Compares the affordance-selected contact point against the detection-box centre
on 300 IIT-AFF images:

```bash
python tools/eval_grasp_point.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml \
  -r output/rtdetr_r50vd_6x_iit_v3_stride2_deepsup/checkpoint.pth
```

> **Known issue, numbers withheld.** The loop in `tools/eval_grasp_point.py`
> increments its counter only after a successful detection and skips images with
> no detection, so the denominator is "images with a detection", not "images".
> The reported hit rate is therefore conditional and overstates the
> unconditional rate. The margin over the box centre survives either way, but
> the absolute figures are being re-measured and are withheld until then.

The pose-jittered Isaac Sim arm evaluation requires an Isaac Sim 4.2
installation and a robot description that are not part of this repository;
those scripts are kept in the authors' research repository. In the paper that
evaluation reports 68.3% (95% CI [50.0, 85.8]) for the affordance point against
44.2% ([25.8, 62.5]) for the box centre, over 24 images x 5 jittered poses.

---

## 7. Diagnostics

```bash
# ground-truth statistics: class balance, instance counts, mask areas
python tools/gt_anatomy.py -c <config>

# what fraction of the metric is reachable given the supervision resolution
python tools/audit_supervision_ceiling.py -c <config>

# operating-point sweep over the detection confidence threshold
python tools/sweep_conf_threshold.py -c <config> -r <checkpoint>

# inspect, diff or repack a checkpoint (EMA vs raw weights)
python tools/ckpt_forensics.py inspect <checkpoint>

# guard against the YAML-kwargs leak that can silently corrupt a config
python tools/check_config_leak.py -c <config>
```

---

## 8. Protocol notes that affect comparability

**No validation split.** IIT-AFF and UMD both ship only train and test. This
codebase therefore fixes checkpoint selection a priori: the last epoch, EMA
weights. No number reported here was obtained by scanning epochs for the best
test score. Results from a protocol that does select on test are not comparable.

That safeguard is deliberately narrow, and it should not be read as a clean bill
of health. It fixes *which epoch* is reported; it does not control the broader
adaptive use of the test split. The same IIT-AFF test set that produces the
headline also informed the recipe as it evolved, the component comparisons, and
the choice of reported configuration. The IIT-AFF results are therefore
exploratory evidence from a single benchmark rather than a confirmatory estimate
of generalisation.

**EMA weights are what is evaluated.** `--test-only` and every scoring tool read
`ema.module`. A checkpoint repacked into a bare `{"model": ...}` dict will
evaluate a freshly initialised EMA module under an EMA-enabled config and score
near zero. Use `tools/ckpt_forensics.py raw-as-ema` if you need to move raw
weights into the EMA slot.

**Backbone weights are fetched at runtime.** `PResNet.pretrained: True` triggers
a download from the RT-DETR release artefacts on first model construction,
including for `--test-only` and ONNX export. Set `pretrained: False` for an
offline run.

**Multi-scale training is on.** In addition to the 640x640 resize in the
transform pipeline, training applies a random square resize in `[480, 800]` per
iteration inside the model's forward pass. Evaluation is fixed at 640x640.

**fp32 is the default; the optimized paths are measured separately.** All
headline accuracy and the headline latency are fp32. The fp16 + CUDA-graph +
folded-BatchNorm path (12.8 ms, 78.4 img/s) and the TensorRT-backbone path used
for the Jetson AGX Orin deployment (43.4 ms, 23.1 FPS, 0.23 GiB) are measured
and documented in the README, not untested.

---

## 9. Exporting to ONNX

```bash
python tools/export_onnx.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml \
  -r output/rtdetr_r50vd_6x_iit_v3_stride2_deepsup/checkpoint.pth \
  --check -f outputs/affkernel_deepsup.onnx
```

The exporter reads `checkpoint['ema']['module']` when present. With
`use_affordance` enabled the graph has four outputs: `labels`, `boxes`,
`scores`, `affordances`. The `affordances` tensor is
`[B, num_top_queries, H, W]` of int64 argmax class-id maps, not per-class
probabilities.
