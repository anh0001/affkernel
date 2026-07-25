"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import sys
from typing import Iterable

import torch
import torch.amp
import torchvision

from src.data import CocoEvaluator, IITDetection, IITEvaluator, UMDDetection, UMDEvaluator
from src.misc import MetricLogger, SmoothedValue, reduce_dict


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    print("Train one epoch...")

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = kwargs.get('print_freq', 10)

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    print("Evaluate...")
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # iou_types = postprocessors.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    if isinstance(base_ds, torchvision.datasets.CocoDetection):
        iou_types = postprocessors.iou_types
        evaluator = CocoEvaluator(base_ds, iou_types)
    elif isinstance(base_ds, UMDDetection):
        # UMD before IIT: UMDEvaluator subclasses IITEvaluator, but UMDDetection
        # is a distinct class so this branch is reached only for UMD.
        iou_types = postprocessors.iou_types
        evaluator = UMDEvaluator(base_ds, use_affordance=postprocessors.use_affordance)
    elif isinstance(base_ds, IITDetection):
        iou_types = postprocessors.iou_types
        # NOTE: IITEvaluator's 2nd positional arg is iou_thresh (float), not
        # iou_types — passing iou_types here made `max_iou >= self.iou_thresh`
        # compare float >= list and crash. Use keyword args.
        evaluator = IITEvaluator(base_ds, use_affordance=postprocessors.use_affordance)
    else:
        raise ValueError(f"Unsupported dataset type: {type(base_ds)}")

    # NOTE: upstream DETR's panoptic evaluation branch is not wired up in this
    # fork (no PanopticEvaluator import, no 'panoptic' postprocessor); the dead
    # placeholder was removed.

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # GT (XML boxes and `.sm` segmasks) lives at the ORIGINAL image
        # resolution. `t["size"]` is overwritten to the post-transform
        # model-input size (e.g. 640x640) by IITDetection, so postprocessing
        # against it puts predictions in 640-space while GT is in original
        # space — shape mismatch zeroes every affordance AP/F_beta^w and
        # tanks bbox AP. Use `orig_size` (true original W,H) when present.
        target_sizes = torch.stack(
            [t.get("orig_size", t["size"]) for t in targets], dim=0
        )
        results = postprocessors(outputs, target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        # Prepare results with image_id and corresponding output
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        # if coco_evaluator is not None:
        #     coco_evaluator.update(res)

        # Update evaluator with predictions
        evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # accumulate predictions from all images
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    stats = {}
    if isinstance(evaluator, CocoEvaluator):
        for iou_type in iou_types:
            stats[f'coco_eval_{iou_type}'] = evaluator.coco_eval[iou_type].stats.tolist()
    elif isinstance(evaluator, UMDEvaluator):
        # UMD before IIT (subclass). Per-class AP lists plus BOTH F_beta^w
        # conventions (beta^2=1 primary, beta^2=0.3 aux).
        for iou_type in evaluator.iou_types:
            s = evaluator.stats.get(iou_type)
            if s is not None:
                stats[f'umd_eval_{iou_type}'] = list(s.get('AP', []))
        for bucket, _beta2 in evaluator.FBW_BETAS:
            fbw = evaluator.stats.get(bucket)
            if fbw is not None and fbw.get('Fbw'):
                stats[f'umd_eval_{bucket}'] = list(fbw['Fbw'])
    elif isinstance(evaluator, IITEvaluator):
        # evaluator.stats[iou_type] is a defaultdict(list) keyed by 'AP'
        # (no .tolist()); expose the per-class AP list directly.
        for iou_type in evaluator.iou_types:
            s = evaluator.stats.get(iou_type)
            if s is not None:
                stats[f'iit_eval_{iou_type}'] = list(s.get('AP', []))
        fbw = evaluator.stats.get('affordance_fbw')
        if fbw is not None and fbw.get('Fbw'):
            stats['iit_eval_affordance_fbw'] = list(fbw['Fbw'])

    return stats, evaluator
