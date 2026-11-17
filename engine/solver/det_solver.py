"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def state_dict(self):
        state = super().state_dict()
        if not getattr(self.cfg, 'reload_best_at_stop_epoch', True):
            best = getattr(self, '_checkpoint_best_stat', None)
            if best is not None:
                state['checkpoint_best_stat'] = dict(best)
        if getattr(self.cfg, 'save_task_best', False):
            state['task_best_stats'] = getattr(self, '_task_best_stats', {})
        state['gradient_accumulation_steps'] = getattr(self.cfg, 'gradient_accumulation_steps', 1)
        return state

    def load_state_dict(self, state):
        if getattr(self, 'optimizer', None) is not None:
            saved_accumulation = state.get('gradient_accumulation_steps', 1)
            requested_accumulation = getattr(self.cfg, 'gradient_accumulation_steps', 1)
            if saved_accumulation != requested_accumulation:
                raise ValueError('Resume requires unchanged gradient accumulation; '
                                 'start a new run for a different training schedule')
        super().load_state_dict(state)
        self._checkpoint_best_stat = state.get('checkpoint_best_stat')
        self._task_best_stats = state.get('task_best_stats', {})

    def _save_task_checkpoints(self, test_stats, epoch):
        """Track both tasks without mixing epochs or selecting on test data."""
        import math
        if not dist_utils.is_main_process():
            return
        def finite(value):
            if value is None:
                return None
            value = float(value)
            return value if math.isfinite(value) else None
        bbox = test_stats.get('coco_eval_bbox', [])
        metrics = {
            'box_ap': finite(bbox[0]) if len(bbox) > 0 else None,
            'box_ap50': finite(bbox[1]) if len(bbox) > 1 else None,
            'box_ap75': finite(bbox[2]) if len(bbox) > 2 else None,
            'raw_grid_f1': finite(test_stats.get('grid_dense_fg_f1')),
            'raw_grid_iou': finite(test_stats.get('grid_dense_fg_miou')),
            'raw_grid_threshold': 0.6,
        }
        self._task_best_stats = getattr(self, '_task_best_stats', {}) or {}
        improved = []
        for name, metric in (('best_box', 'box_ap'), ('best_grid', 'raw_grid_f1')):
            value = metrics[metric]
            previous = self._task_best_stats.get(name)
            if value is not None and value >= 0 and (previous is None or value > previous['value']):
                self._task_best_stats[name] = dict(epoch=epoch, metric=metric, value=value,
                    model_source='ema' if self.ema else 'model', metrics=dict(metrics))
                improved.append(name)
        if self.output_dir:
            # Update both histories before serializing either checkpoint.
            for name in improved:
                dist_utils.save_on_master(self.state_dict(), self.output_dir / f'{name}.pth')
            (self.output_dir / 'best_checkpoints.json').write_text(
                json.dumps(self._task_best_stats, indent=2, allow_nan=False), encoding='utf-8')

    def fit(self, ):
        self.train()
        args = self.cfg
        reload_best_at_stop = getattr(args, 'reload_best_at_stop_epoch', True)
        accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        if type(accumulation_steps) is not int or accumulation_steps < 1:
            raise ValueError('gradient_accumulation_steps must be a positive integer')

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = (len(self.train_dataloader) + accumulation_steps - 1) // accumulation_steps
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            print(f'Optimizer updates/epoch: {iter_per_epoch}; gradient accumulation: {accumulation_steps}')
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                args.lr_gamma,
                iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter,
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch
            )
            self.self_lr_scheduler = True

        n_trainable_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_trainable_parameters}')

        n_non_trainable_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_non_trainable_parameters}')

        def get_metric_value(test_stats, metric_name):
            """
            Unified metric getter for best checkpoint selection.

            Supports:
            - list metrics, e.g. coco_eval_bbox -> use [0]
            - scalar float/int metrics, e.g. grid_gcbd_f1
            - tensor scalar metrics
            """
            if metric_name not in test_stats:
                return None

            v = test_stats[metric_name]

            if isinstance(v, list):
                return float(v[0]) if len(v) > 0 else None
            if isinstance(v, (int, float)):
                return float(v)
            if hasattr(v, 'item'):
                return float(v.item())

            return None

        def to_serializable(v):
            if hasattr(v, 'item'):
                return v.item()
            elif isinstance(v, (list, tuple)):
                return [to_serializable(x) for x in v]
            elif isinstance(v, dict):
                return {k: to_serializable(val) for k, val in v.items()}
            return v

        def sanitize_name(name):
            """
            Make class/metric names safer for flat JSON logging keys.
            """
            return str(name).replace(' ', '_').replace('/', '_')

        # Choose metric used to save best checkpoint
        # Examples:
        #   coco_eval_bbox
        #   grid_gcbd_f1
        #   grid_dense_fg_miou
        best_metric_name = getattr(args, 'best_metric', 'coco_eval_bbox')
        best_metric_value = float('-inf')
        best_stat = {
            'epoch': -1,
            'best_metric_name': best_metric_name,
            'best_metric_value': best_metric_value,
        }
        restored_best = getattr(self, '_checkpoint_best_stat', None)
        if (not reload_best_at_stop and restored_best is not None
                and restored_best.get('best_metric_name') == best_metric_name):
            best_stat = dict(restored_best)
            best_metric_value = best_stat['best_metric_value']
        self._checkpoint_best_stat = best_stat

        # evaluate again before resume training
        if self.last_epoch > 0 or (not reload_best_at_stop and self.last_epoch == 0):
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                output_dir=str(self.output_dir) if self.output_dir else None
            )

            metric_value = get_metric_value(test_stats, best_metric_name)
            if metric_value is not None and metric_value > best_metric_value:
                best_stat['epoch'] = self.last_epoch
                best_stat['best_metric_value'] = metric_value
                best_metric_value = metric_value

            print(f'best_stat: {best_stat}')
            if getattr(args, 'save_task_best', False):
                self._save_task_checkpoints(test_stats, self.last_epoch)

        start_time = time.time()
        start_epoch = self.last_epoch + 1

        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                if reload_best_at_stop:
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')
                else:
                    print(f'Continuous training at epoch {epoch}: augmentation changes; '
                          'model, optimizer and EMA continue without reload.')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                gradient_accumulation_steps=accumulation_steps,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            if reload_best_at_stop:
                self.last_epoch += 1
            else:
                self.last_epoch = epoch

            if self.output_dir and reload_best_at_stop and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every checkpoint_freq epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                output_dir=str(self.output_dir) if self.output_dir else None
            )


            # TensorBoard logging

            if self.writer and dist_utils.is_main_process():
                for k, v in test_stats.items():
                    # validation losses
                    if k.startswith('val_'):
                        if isinstance(v, (int, float)):
                            self.writer.add_scalar(f'Val/{k}', v, epoch)
                        elif hasattr(v, 'item'):
                            self.writer.add_scalar(f'Val/{k}', v.item(), epoch)

                    # flattened grid metrics
                    elif k.startswith('grid_'):
                        if isinstance(v, (int, float)):
                            self.writer.add_scalar(f'Grid/{k}', v, epoch)
                        elif hasattr(v, 'item'):
                            self.writer.add_scalar(f'Grid/{k}', v.item(), epoch)

                    # coco_eval_bbox / coco_eval_masks style list metrics
                    elif isinstance(v, list):
                        for i, vv in enumerate(v):
                            self.writer.add_scalar(f'Test/{k}_{i}', vv, epoch)


            # Best checkpoint selection

            metric_value = get_metric_value(test_stats, best_metric_name)
            best_path = None
            if metric_value is None:
                if dist_utils.is_main_process():
                    print(f"[Warning] best_metric '{best_metric_name}' not found in test_stats.")
            elif metric_value > best_metric_value:
                best_metric_value = metric_value
                best_stat['epoch'] = epoch
                best_stat['best_metric_value'] = metric_value

                if self.output_dir:
                    if not reload_best_at_stop:
                        best_path = self.output_dir / 'best.pth'
                    elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        best_path = self.output_dir / 'best_stg2.pth'
                    else:
                        best_path = self.output_dir / 'best_stg1.pth'

            if getattr(args, 'save_task_best', False):
                self._save_task_checkpoints(test_stats, epoch)
            if best_path is not None:
                dist_utils.save_on_master(self.state_dict(), best_path)
            # Continuous runs checkpoint after validation so resume retains both
            # current best histories, including improvements on this epoch.
            if self.output_dir and not reload_best_at_stop:
                paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for path in paths:
                    dist_utils.save_on_master(self.state_dict(), path)

            print(f'best_stat: {best_stat}')


            # Flatten eval_grid for log.txt

            grid_log_stats = {}
            if 'eval_grid' in test_stats:
                grid_data = test_stats['eval_grid']

                gcbd = grid_data.get('gcbd_metrics', {})
                dense = grid_data.get('dense_metrics', {})

                # GCBD summary
                if 'all' in gcbd:
                    for k, v in gcbd['all'].items():
                        grid_log_stats[f'test_grid_gcbd_all_{sanitize_name(k)}'] = to_serializable(v)

                # GCBD per-class
                if 'per_class' in gcbd:
                    for cls_name, metrics in gcbd['per_class'].items():
                        cls_name_safe = sanitize_name(cls_name)
                        for k, v in metrics.items():
                            grid_log_stats[f'test_grid_gcbd_{cls_name_safe}_{sanitize_name(k)}'] = to_serializable(v)

                # Dense foreground-only summary
                if 'foreground_only' in dense:
                    fg = dense['foreground_only']
                    for k, v in fg.items():
                        if k != 'per_class':
                            grid_log_stats[f'test_grid_dense_fg_{sanitize_name(k)}'] = to_serializable(v)

                # Dense with-background summary + per-class
                if 'include_background' in dense:
                    bg = dense['include_background']
                    for k, v in bg.items():
                        if k != 'per_class':
                            grid_log_stats[f'test_grid_dense_all_{sanitize_name(k)}'] = to_serializable(v)

                    per_class = bg.get('per_class', [])
                    for idx, metrics in enumerate(per_class):
                        for k, v in metrics.items():
                            grid_log_stats[f'test_grid_dense_class_{idx}_{sanitize_name(k)}'] = to_serializable(v)


            # Write log.txt

            log_stats = {
                **{f'train_{k}': to_serializable(v) for k, v in train_stats.items()},
                **{f'test_{k}': to_serializable(v) for k, v in test_stats.items() if k != 'eval_grid'},
                **grid_log_stats,
                'epoch': epoch,
                'n_parameters': n_trainable_parameters,
                'best_metric_name': best_metric_name,
                'best_metric_value': best_metric_value,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        export_grids = self.cfg.yaml_cfg.get('grid_export', False)
        if export_grids and dist_utils.get_world_size() != 1:
            raise ValueError('Use one process for grid_export: python train.py --test-only ...')
        if export_grids and self.cfg.yaml_cfg.get('remap_mscoco_category', False):
            raise ValueError('Crack grid export requires zero-based detection category 0')
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            output_dir=str(self.output_dir) if self.output_dir else None,
            grid_export_metadata=(dict(checkpoint=self.cfg.resume,
                config=self.cfg.yaml_cfg.get('config'),
                model_source='ema' if self.ema else 'model') if export_grids else None),
        )

        if self.output_dir and coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval,
                self.output_dir / "eval.pth"
            )

        return
