'''
by lyuwenyu
'''
import datetime
import json
import time

import torch
import torchvision

from src.data import CocoEvaluator, IITDetection, IITEvaluator, UMDDetection, UMDEvaluator
from src.misc import dist

from .det_engine import evaluate, train_one_epoch
from .solver import BaseSolver


class DetSolver(BaseSolver):

    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        # Handle dataset type
        dataset = self.val_dataloader.dataset
        for _ in range(10):  # Handle potential nested Subset wrappers
            if isinstance(dataset, (torchvision.datasets.CocoDetection, IITDetection, UMDDetection)):
                break
            if isinstance(dataset, torch.utils.data.Subset):
                dataset = dataset.dataset

        if isinstance(dataset, torchvision.datasets.CocoDetection):
            base_ds = dataset.coco
        elif isinstance(dataset, (IITDetection, UMDDetection)):
            base_ds = dataset
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # Handle the stats
            if 'coco_eval_bbox' in test_stats:
                # COCO dataset
                best_stat['epoch'] = epoch if test_stats['coco_eval_bbox'][0] > best_stat.get('coco_eval_bbox', [0])[0] else best_stat['epoch']
                best_stat['coco_eval_bbox'] = max(best_stat.get('coco_eval_bbox', [0]), test_stats['coco_eval_bbox'], key=lambda x: x[0])
            elif 'iit_eval_bbox' in test_stats:
                # IIT dataset
                best_stat['epoch'] = epoch if test_stats['iit_eval_bbox'][0] > best_stat.get('iit_eval_bbox', [0])[0] else best_stat['epoch']
                best_stat['iit_eval_bbox'] = max(best_stat.get('iit_eval_bbox', [0]), test_stats['iit_eval_bbox'], key=lambda x: x[0])
            elif 'umd_eval_bbox' in test_stats:
                # UMD dataset
                best_stat['epoch'] = epoch if test_stats['umd_eval_bbox'][0] > best_stat.get('umd_eval_bbox', [0])[0] else best_stat['epoch']
                best_stat['umd_eval_bbox'] = max(best_stat.get('umd_eval_bbox', [0]), test_stats['umd_eval_bbox'], key=lambda x: x[0])

            print('best_stat: ', best_stat)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if isinstance(evaluator, CocoEvaluator):
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)
                elif isinstance(evaluator, IITEvaluator):
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        # Save the evaluation statistics
                        torch.save({
                            'stats': evaluator.stats,
                            'object_classes': evaluator.object_classes,
                            'iou_thresh': evaluator.iou_thresh,
                            'use_07_metric': evaluator.use_07_metric
                        }, self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'Training time {total_time_str}')


    def val(self):
        self.eval()

        # Determine the dataset type
        dataset = self.val_dataloader.dataset
        for _ in range(10):  # Handle potential nested Subset wrappers
            if isinstance(dataset, (torchvision.datasets.CocoDetection, IITDetection, UMDDetection)):
                break
            if isinstance(dataset, torch.utils.data.Subset):
                dataset = dataset.dataset

        if isinstance(dataset, torchvision.datasets.CocoDetection):
            base_ds = dataset.coco
        elif isinstance(dataset, (IITDetection, UMDDetection)):
            base_ds = dataset
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

        module = self.ema.module if self.ema else self.model
        test_stats, evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)

        if self.output_dir:
            if isinstance(evaluator, CocoEvaluator):
                if "bbox" in evaluator.coco_eval:
                    torch.save(evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
            elif isinstance(evaluator, IITEvaluator):
                torch.save({
                    'stats': evaluator.stats,
                    'object_classes': evaluator.object_classes,
                    'iou_thresh': evaluator.iou_thresh,
                    'use_07_metric': evaluator.use_07_metric
                }, self.output_dir / "eval.pth")
            elif isinstance(evaluator, UMDEvaluator):
                torch.save({'stats': evaluator.stats}, self.output_dir / "eval.pth")

        return test_stats
