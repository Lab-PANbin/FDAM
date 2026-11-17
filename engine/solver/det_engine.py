"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..data.dataset.grid_eval import GridEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _grid_logit_stats(outputs):
    """Observe pre-clamp logits; a large loss can hide zero clamp gradients."""
    logits = outputs.get('pred_grid_logits')
    if logits is None or logits.numel() == 0:
        return {}
    logits = logits.detach().float()
    return {
        'grid_logit_min': logits.amin(),
        'grid_logit_max': logits.amax(),
        'grid_clip_frac': ((logits < -20) | (logits > 20)).float().mean(),
    }


def _get_bare_model(model, ema: 'ModelEMA | None'):
    """Unwrap distributed and EMA model containers."""
    m = ema.module if ema is not None else model

    return getattr(m, 'module', m)


def train_one_epoch(
    self_lr_scheduler,
    lr_scheduler,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    import os

    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq          = kwargs.get('print_freq', 10)
    writer: SummaryWriter = kwargs.get('writer', None)
    ema: ModelEMA       = kwargs.get('ema', None)
    scaler: GradScaler  = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)

    accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
    if type(accumulation_steps) is not int or accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be a positive integer')
    updates_per_epoch = (len(data_loader) + accumulation_steps - 1) // accumulation_steps
    cur_iters = epoch * updates_per_epoch
    optimizer.zero_grad()

    # Opt-in, per-rank diagnostics for the first batches of epoch zero.
    trace_steps = int(os.environ.get('FDAM_TRACE_STEPS', '0'))
    def trace_step(step, stage):
        if epoch == 0 and step < trace_steps:
            print(f'[Train rank={dist_utils.get_rank()} epoch={epoch} step={step}] '
                  f'{stage}', force=True, flush=True)


    bare_model = _get_bare_model(model, ema)

    if hasattr(bare_model, '_current_epoch'):
        bare_model._current_epoch = epoch


        grid_detach_epochs = getattr(bare_model, 'grid_detach_epochs', None)
        if grid_detach_epochs is not None and dist_utils.is_main_process():
            if epoch == 0:
                print(f"[DEIM] Grid enhancement detach=True  "
                      f"(epoch 0 → {grid_detach_epochs - 1})")
            elif epoch == grid_detach_epochs:
                print(f"[DEIM] Grid enhancement detach=False "
                      f"(bidirectional coupling starts at epoch {epoch})")


    trace_step(0, 'waiting for data')
    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if epoch == 0 and i < trace_steps:
            image_ids = [t['image_id'].reshape(-1).tolist() for t in targets
                         if 'image_id' in t]
            trace_step(i, f'data ready, image_ids={image_ids}; transfer to device')
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        global_step = epoch * len(data_loader) + i
        metas = dict(
            epoch=epoch,
            step=i,
            global_step=global_step,
            epoch_step=len(data_loader),
        )
        window_start = (i // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, len(data_loader) - window_start)
        update_now = (i + 1) % accumulation_steps == 0 or i + 1 == len(data_loader)

        trace_step(i, 'forward')
        if scaler is not None:
            with torch.autocast(device_type=torch.device(device).type, cache_enabled=True):
                outputs = model(samples, targets=targets)


            if (outputs.get('pred_boxes') is not None and
                    (torch.isnan(outputs['pred_boxes']).any()
                     or torch.isinf(outputs['pred_boxes']).any())):
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in state.items():
                    new_key = key.replace('module.', '')
                    new_state[new_key] = value
                dist_utils.save_on_master({'model': new_state}, "./NaN.pth")

            trace_step(i, 'criterion')
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            trace_step(i, 'backward')
            scaler.scale(loss / window_size).backward()

        else:
            outputs = model(samples, targets=targets)
            trace_step(i, 'criterion')
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            trace_step(i, 'backward')
            (loss / window_size).backward()

        if update_now:
            trace_step(i, 'gradient clipping / optimizer update')
            if max_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            step_succeeded = True
            if scaler is not None:
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_succeeded = scaler.get_scale() >= old_scale
            else:
                optimizer.step()
            optimizer.zero_grad()

            if step_succeeded:
                if ema is not None:
                    trace_step(i, 'EMA update')
                    ema.update(model)
                if self_lr_scheduler:
                    optimizer = lr_scheduler.step(cur_iters + i // accumulation_steps, optimizer)
                elif lr_warmup_scheduler is not None:
                    lr_warmup_scheduler.step()


        trace_step(i, 'reduce loss metrics')
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values()).detach()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(**_grid_logit_stats(outputs))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if writer and dist_utils.is_main_process() and global_step % 10 == 0:


            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)


            grid_enhancer = getattr(bare_model, 'grid_enhancer', None)
            if grid_enhancer is not None and hasattr(grid_enhancer, 'get_alpha_stats'):
                for k, v in grid_enhancer.get_alpha_stats().items():
                    writer.add_scalar(f'GridEnhancer/{k}', v, global_step)


            if hasattr(bare_model, '_current_epoch') and hasattr(bare_model, 'grid_detach_epochs'):
                is_bidirectional = int(
                    bare_model._current_epoch >= bare_model.grid_detach_epochs
                )
                writer.add_scalar('GridEnhancer/bidirectional', is_bidirectional, global_step)

        trace_step(i, 'batch complete')
        trace_step(i + 1, 'waiting for data')


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if dist_utils.is_main_process():
        grid_enhancer = getattr(bare_model, 'grid_enhancer', None)
        if grid_enhancer is not None and hasattr(grid_enhancer, 'get_alpha_stats'):
            alpha_stats = grid_enhancer.get_alpha_stats()
            alpha_str = '  '.join(f'{k}={v:+.4f}' for k, v in alpha_stats.items())
            detach_epochs = getattr(bare_model, 'grid_detach_epochs', '?')
            mode = (
                'bidirectional'
                if bare_model._current_epoch >= bare_model.grid_detach_epochs
                else 'detach'
            )
            print(f"[GridEnhancer] epoch={epoch}  mode={mode}  {alpha_str}  "
                  f"(bidirectional starts at epoch {detach_epochs})")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _valid_eval_indices(data_loader, local_start, batch_size):
    """Exclude DistributedSampler padding while retaining equal forward counts."""
    sampler = getattr(data_loader, 'sampler', None)
    if isinstance(sampler, torch.utils.data.DistributedSampler) and not sampler.drop_last:
        return [i for i in range(batch_size)
                if (local_start + i) * sampler.num_replicas + sampler.rank < len(sampler.dataset)]
    return list(range(batch_size))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    output_dir=None,
    compute_loss=True,
    grid_export_metadata=None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    if coco_evaluator is not None:
        coco_evaluator.cleanup()
        if output_dir is not None:
            coco_evaluator.output_dir = output_dir


    class_names = None
    try:
        if coco_evaluator is not None and hasattr(coco_evaluator, 'coco_gt'):
            cats = coco_evaluator.coco_gt.cats
            if cats:
                sorted_cats = sorted(cats.items(), key=lambda x: x[0])
                class_names = [cat['name'] for _, cat in sorted_cats]
    except Exception as e:
        print(f"Warning: Could not extract class names from COCO evaluator: {e}")

    grid_evaluator = GridEvaluator(
        num_classes=None,
        class_names=class_names,
        device=device,
        output_dir=output_dir,
        conf_thresh=0.6,
    )

    grid_exporter = None
    if grid_export_metadata is not None:
        if dist_utils.get_world_size() != 1:
            raise ValueError('grid_export requires one process to avoid padded duplicate samples')
        if output_dir is None:
            raise ValueError('grid_export requires an output directory')
        from pathlib import Path
        from ..data.dataset.grid_export import GridPredictionExporter
        grid_exporter = GridPredictionExporter(
            Path(output_dir) / 'grid_predictions', data_loader.dataset, grid_export_metadata)

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = coco_evaluator.iou_types if coco_evaluator is not None else []
    local_eval_count = 0
    evaluated_boxes = False
    evaluated_grid = False

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        metric_logger.update(**{f'val_{k}': v for k, v in _grid_logit_stats(outputs).items()})


        if compute_loss and criterion is not None:
            loss_dict = criterion(outputs, targets)
            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values())
            metric_logger.update(
                val_loss=loss_value,
                **{f'val_{k}': v for k, v in loss_dict_reduced.items()},
            )

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        has_boxes = outputs.get('pred_boxes') is not None and outputs.get('pred_logits') is not None
        has_grid = outputs.get('pred_grid_logits') is not None
        evaluated_boxes = evaluated_boxes or has_boxes
        evaluated_grid = evaluated_grid or has_grid
        if has_boxes:
            results = postprocessor(outputs, orig_target_sizes)
        else:
            # Preserve image-wise export records without inventing box predictions.
            results = [dict(boxes=samples.new_zeros((0, 4)), scores=samples.new_zeros((0,)),
                            labels=torch.zeros(0, dtype=torch.long, device=samples.device))
                       for _ in targets]
        valid_indices = _valid_eval_indices(data_loader, local_eval_count, len(targets))
        local_eval_count += len(targets)
        metric_targets = [targets[i] for i in valid_indices]
        metric_results = [results[i] for i in valid_indices]

        if coco_evaluator is not None and has_boxes:
            res = {
                target['image_id'].item(): output
                for target, output in zip(targets, results)
            }
            coco_evaluator.update(res)

            if hasattr(coco_evaluator, 'update_confusion_matrix'):
                coco_evaluator.update_confusion_matrix(metric_targets, metric_results)

        # COCO AP already deduplicates image IDs when synchronizing. Grid metrics
        # and detection confusion counts must explicitly discard padded entries.
        # Even an empty local batch initializes grid confusion on every rank,
        # keeping distributed reduction collectives aligned for tiny datasets.
        grid_outputs = outputs
        if len(valid_indices) != len(targets) and outputs.get('pred_grid_logits') is not None:
            grid_outputs = {'pred_grid_logits': outputs['pred_grid_logits'][valid_indices]}
        if has_grid:
            grid_evaluator.update(grid_outputs, metric_targets)

        if grid_exporter is not None:
            logits = outputs.get('pred_grid_logits')
            if logits is None:
                raise ValueError('Model did not return pred_grid_logits')
            grid_gt = grid_evaluator.box2grid(logits, targets, use_expansion=True, grid_delta=0.125)
            grid_exporter.update(logits.sigmoid(), grid_gt, targets, results, samples.shape[-2:])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if not evaluated_boxes:
        coco_evaluator = None
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    if evaluated_grid and dist_utils.is_dist_available_and_initialized():
        grid_evaluator.synchronize_between_processes()

    grid_stats = {}
    if evaluated_grid and dist_utils.is_main_process():
        grid_stats = grid_evaluator.summarize(plot_curves=True)

    stats = {}

    for k, meter in metric_logger.meters.items():
        if k.startswith('val_'):
            stats[k] = meter.global_avg

    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    if grid_stats:
        stats['eval_grid'] = grid_stats

        gcbd_all  = grid_stats.get('gcbd_metrics',  {}).get('all', {})
        dense_fg  = grid_stats.get('dense_metrics', {}).get('foreground_only', {})
        dense_bg  = grid_stats.get('dense_metrics', {}).get('include_background', {})

        if gcbd_all:
            stats['grid_gcbd_precision']        = gcbd_all.get('precision', 0.0)
            stats['grid_gcbd_recall']            = gcbd_all.get('recall', 0.0)
            stats['grid_gcbd_f1']                = gcbd_all.get('f1', 0.0)
            stats['grid_gcbd_map50']             = gcbd_all.get('map50', 0.0)
            stats['grid_gcbd_map50_95']          = gcbd_all.get('map50_95', 0.0)
            stats['grid_gcbd_best_confidence']   = gcbd_all.get('best_confidence', 0.0)

        if dense_fg:
            stats['grid_dense_fg_precision'] = dense_fg.get('macro_precision', 0.0)
            stats['grid_dense_fg_recall']    = dense_fg.get('macro_recall', 0.0)
            stats['grid_dense_fg_f1']        = dense_fg.get('macro_f1', 0.0)
            stats['grid_dense_fg_miou']      = dense_fg.get('miou', 0.0)

        if dense_bg:
            stats['grid_dense_all_precision'] = dense_bg.get('macro_precision', 0.0)
            stats['grid_dense_all_recall']    = dense_bg.get('macro_recall', 0.0)
            stats['grid_dense_all_f1']        = dense_bg.get('macro_f1', 0.0)
            stats['grid_dense_all_miou']      = dense_bg.get('miou', 0.0)

    if grid_exporter is not None:
        bbox = stats.get('coco_eval_bbox')
        box_metrics = dict(zip(('ap', 'ap50', 'ap75'), bbox[:3])) if bbox is not None else None
        grid_exporter.finish(box_metrics=box_metrics)

    return stats, coco_evaluator
