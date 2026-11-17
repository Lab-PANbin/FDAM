"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Grid Classification Evaluator
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist


def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """Compute AP per class (GCBD style)."""
    if len(conf) == 0 or len(target_cls) == 0:
        return (
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
            np.zeros((0, 10), dtype=np.float32), np.array([], dtype=int), 0.0
        )

    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    px = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, 10), dtype=np.float32)
    p = np.zeros((nc, 1000), dtype=np.float32)
    r = np.zeros((nc, 1000), dtype=np.float32)

    for ci, c in enumerate(unique_classes):
        mask = pred_cls == c
        n_l = nt[ci]
        n_p = mask.sum()

        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[mask]).cumsum(0)
        tpc = tp[mask].cumsum(0)

        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc + eps)

        r[ci] = np.interp(-px, -conf[mask], recall[:, 0], left=0)
        p[ci] = np.interp(-px, -conf[mask], precision[:, 0], left=1)

        ap_val, _, _ = compute_ap(recall[:, 0], precision[:, 0])

        # Grid task has no IoU-threshold sweep like box detection.
        # Keep 10 identical columns for interface compatibility.
        ap[ci, :] = ap_val

    f1 = 2 * p * r / (p + r + eps)

    idx = smooth(f1.mean(0), 0.1).argmax()
    ar = r.mean(0)

    # Match GCBD style: if best-F1 threshold gives too low recall, push recall > 0.7 if possible
    if ar[idx] < 0.7:
        index = (ar > 0.7).nonzero()[0]
        if len(index):
            idx = index.max()

    p_best, r_best, f1_best = p[:, idx], r[:, idx], f1[:, idx]

    tp_count = (r_best * nt).round()
    fp_count = (tp_count / (p_best + eps) - tp_count).round()
    best_conf = float(idx / len(px))

    return tp_count, fp_count, p_best, r_best, f1_best, ap, unique_classes.astype(int), best_conf


class GridEvaluator:
    def __init__(
        self,
        num_classes=None,
        class_names=None,
        device='cuda',
        output_dir=None,
        conf_thresh=0.3,
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.output_dir = output_dir
        self.conf_thresh = conf_thresh

        # GCBD-style stats
        self.stats_grid = []
        self.seen = 0

        # Dense confusion matrix: shape (nc + 1, nc + 1), last index is background
        self.confusion_matrix = None


    def box2grid(self, pred_grid, targets_grid, cp=1.0, use_expansion=False, grid_delta=0.125):
        """Convert GT boxes to GT grid labels."""
        B, C, H, W = pred_grid.shape
        device = pred_grid.device
        grid = torch.zeros_like(pred_grid, device=device)

        for i, t in enumerate(targets_grid):
            if 'labels' not in t or 'boxes' not in t:
                continue

            labels = t['labels']
            boxes = t['boxes']

            grid_mask = labels < 0
            if not grid_mask.any():
                continue

            g_labels = labels[grid_mask]
            g_boxes = boxes[grid_mask]

            # -1 -> 0, -2 -> 1, ...
            c = (g_labels.abs() - 1).long()

            cx = g_boxes[:, 0]
            cy = g_boxes[:, 1]
            w = g_boxes[:, 2]
            h = g_boxes[:, 3]

            if use_expansion:
                offsets = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1),  (0, 0),  (0, 1),
                    (1, -1),  (1, 0),  (1, 1)
                ]

                for dh, dw in offsets:
                    y = ((cy + dh * grid_delta * h) * H).long()
                    x = ((cx + dw * grid_delta * w) * W).long()

                    valid = (c >= 0) & (c < C) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
                    if valid.any():
                        grid[i, c[valid], y[valid], x[valid]] = cp
            else:
                # GCBD-style eval: center only
                x = (cx * W).long().clamp(0, W - 1)
                y = (cy * H).long().clamp(0, H - 1)

                valid = (c >= 0) & (c < C)
                if valid.any():
                    grid[i, c[valid], y[valid], x[valid]] = cp

        return grid

    def update(self, outputs, targets):
        pred_logits = outputs.get('pred_grid_logits', None)
        if pred_logits is None:
            return

        B, C, H, W = pred_logits.shape

        if self.num_classes is None:
            self.num_classes = C

        preds_grid = torch.sigmoid(pred_logits)

        # GT grid used for grid evaluation
        # use_expansion=True to match training (3×3 expansion, grid_delta=0.125)
        labels_grid = self.box2grid(
            preds_grid, targets, cp=1.0, use_expansion=True, grid_delta=0.125
        )

        # Collect GT class labels in same traversal order
        target_cls_list = []
        for c_idx in range(C):
            n_positive = int(labels_grid[:, c_idx, :, :].sum().item())
            class_label = -(c_idx + 1)
            target_cls_list.extend([class_label] * n_positive)
        # for t in targets:
        #     if 'labels' not in t:
        #         continue
        #     labels = t['labels']
        #     grid_mask = labels < 0
        #     if grid_mask.any():
        #         g_labels = labels[grid_mask]
        #         target_cls_list.extend(g_labels.detach().cpu().tolist())

        target_cls = torch.tensor(target_cls_list, dtype=torch.float32) if target_cls_list else torch.tensor([])

        # GCBD-style low threshold
        eval_conf_thres = 0.001
        pred_index = preds_grid > eval_conf_thres

        pred_cls = torch.zeros_like(preds_grid)
        for i in range(C):
            pred_cls[:, i, ...] = -i - 1

        correct = labels_grid[pred_index].reshape(-1, 1).bool()
        conf = preds_grid[pred_index]
        pred_cls_flat = pred_cls[pred_index]

        self.stats_grid.append((
            correct.cpu(),
            conf.cpu(),
            pred_cls_flat.cpu(),
            target_cls.cpu()
        ))

        self.seen += B

        self._update_dense_confusion_matrix(preds_grid, labels_grid, self.conf_thresh)

    def summarize(self, plot_curves=True):
        print("\n" + "-" * 100)
        print(" Grid Classification Results")
        print("-" * 100)

        gcbd_metrics = self._summarize_gcbd(plot_curves=plot_curves)
        dense_metrics = self._summarize_dense(plot_cm=plot_curves)

        return {
            'gcbd_metrics': gcbd_metrics,
            'dense_metrics': dense_metrics
        }

    def _summarize_gcbd(self, plot_curves=True):
        print("\n[1] GCBD-style metrics")

        if not self.stats_grid:
            print("No grid predictions found.")
            return {}

        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats_grid)]

        if len(stats) == 0 or len(stats[0]) == 0:
            print("No valid predictions found.")
            return {}

        tp, conf, pred_cls, target_cls = stats

        if len(target_cls) == 0:
            print("No GT grid targets found.")
            return {}

        tp_count, fp_count, p, r, f1, ap, unique_classes, best_conf = ap_per_class(
            tp, conf, pred_cls, target_cls
        )

        if len(unique_classes) == 0:
            print("No valid classes for GCBD-style evaluation.")
            return {}

        nt = np.bincount((-target_cls - 1).astype(int), minlength=self.num_classes or 1)

        ap50 = ap[:, 0]
        ap50_95 = ap.mean(1)

        mp, mr = p.mean(), r.mean()
        mf1 = f1.mean()
        map50, map50_95 = ap50.mean(), ap50_95.mean()

        names = {}
        for i, c in enumerate(unique_classes):
            c_idx = int(-c - 1)
            if self.class_names and c_idx < len(self.class_names):
                names[i] = self.class_names[c_idx]
            else:
                names[i] = f"Class_{c_idx}"

        print(f"\n{'Task':<12} {'Class':<15} {'Images':>8} {'Instances':>10} {'P':>8} {'R':>8} {'F1':>8} {'mAP50':>10} {'mAP50-95':>12}")
        print("-" * 100)

        total_instances = sum(nt[int(-c - 1)] for c in unique_classes)
        print(f"{'grid-gcbd':<12} {'all':<15} {self.seen:>8} {int(total_instances):>10} {mp:>8.3f} {mr:>8.3f} {mf1:>8.3f} {map50:>10.3f} {map50_95:>12.3f}")

        results = {
            'all': {
                'precision': float(mp),
                'recall': float(mr),
                'f1': float(mf1),
                'map50': float(map50),
                'map50_95': float(map50_95),
                'best_confidence': float(best_conf),
                'num_images': int(self.seen),
                'num_instances': int(total_instances),
                'note': 'For grid classification, mAP50 and mAP50-95 are identical because IoU threshold sweep is not applied.'
            },
            'per_class': {}
        }

        if len(unique_classes) > 0:
            print("-" * 100)
            for i, c in enumerate(unique_classes):
                c_idx = int(-c - 1)
                c_name = names.get(i, f"Class_{c_idx}")
                n_instances = int(nt[c_idx])

                print(f"{'grid-gcbd':<12} {c_name:<15} {self.seen:>8} {n_instances:>10} {p[i]:>8.3f} {r[i]:>8.3f} {f1[i]:>8.3f} {ap50[i]:>10.3f} {ap50_95[i]:>12.3f}")

                results['per_class'][c_name] = {
                    'precision': float(p[i]),
                    'recall': float(r[i]),
                    'f1': float(f1[i]),
                    'map50': float(ap50[i]),
                    'map50_95': float(ap50_95[i]),
                    'num_instances': n_instances
                }

        print("-" * 100)

        if plot_curves and self.output_dir:
            self._plot_curves(stats, names, unique_classes, ap)

        return results

    def _summarize_dense(self, plot_cm=True):
        print("\n[2] Dense grid metrics (segmentation-style)")

        if self.confusion_matrix is None:
            print("No dense confusion matrix found.")
            return {}

        cm = self.confusion_matrix.copy()

        metrics_fg_only = self._compute_dense_metrics_from_cm(cm, include_background=False)
        metrics_with_bg = self._compute_dense_metrics_from_cm(cm, include_background=True)

        labels = self._get_labels(include_background=True)

        print(f"\n{'Task':<12} {'Scope':<16} {'mP':>8} {'mR':>8} {'mF1':>8} {'mIoU':>10}")
        print("-" * 70)
        print(f"{'grid-dense':<12} {'fg-only':<16} "
              f"{metrics_fg_only['macro_precision']:>8.3f} "
              f"{metrics_fg_only['macro_recall']:>8.3f} "
              f"{metrics_fg_only['macro_f1']:>8.3f} "
              f"{metrics_fg_only['miou']:>10.3f}")
        print(f"{'grid-dense':<12} {'with-bg':<16} "
              f"{metrics_with_bg['macro_precision']:>8.3f} "
              f"{metrics_with_bg['macro_recall']:>8.3f} "
              f"{metrics_with_bg['macro_f1']:>8.3f} "
              f"{metrics_with_bg['miou']:>10.3f}")
        print("-" * 70)

        print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10} {'#GT':>10}")
        print("-" * 70)
        for idx, cls_result in enumerate(metrics_with_bg['per_class']):
            print(f"{labels[idx]:<15} "
                  f"{cls_result['precision']:>10.3f} "
                  f"{cls_result['recall']:>10.3f} "
                  f"{cls_result['f1']:>10.3f} "
                  f"{cls_result['iou']:>10.3f} "
                  f"{cls_result['#GT']:>10d}")
        print("-" * 70)

        if plot_cm and self.output_dir:
            self._plot_confusion_matrix(cm)

        return {
            'foreground_only': metrics_fg_only,
            'include_background': metrics_with_bg
        }

    def _compute_dense_metrics_from_cm(self, cm, include_background=True, eps=1e-16):
        """
        rows = GT, cols = Pred
        cm shape = (num_classes + 1, num_classes + 1), last index is background
        """
        n_total = cm.shape[0]
        valid_indices = list(range(n_total)) if include_background else list(range(n_total - 1))

        per_class = []
        for c in range(n_total):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            gt = cm[c, :].sum()

            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            iou = tp / (tp + fp + fn + eps)

            per_class.append({
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'iou': float(iou),
                '#GT': int(gt),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
            })

        macro_precision = float(np.mean([per_class[i]['precision'] for i in valid_indices])) if valid_indices else 0.0
        macro_recall = float(np.mean([per_class[i]['recall'] for i in valid_indices])) if valid_indices else 0.0
        macro_f1 = float(np.mean([per_class[i]['f1'] for i in valid_indices])) if valid_indices else 0.0
        miou = float(np.mean([per_class[i]['iou'] for i in valid_indices])) if valid_indices else 0.0

        return {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'miou': miou,
            'per_class': per_class,
        }

    def _get_labels(self, include_background=True):
        nc = self.num_classes or 0
        if self.class_names and len(self.class_names) >= nc:
            labels = list(self.class_names[:nc])
        else:
            labels = [f'Class_{i}' for i in range(nc)]

        if include_background:
            labels += ['Background']
        return labels

    def _update_dense_confusion_matrix(self, preds_grid, labels_grid, conf_thresh):
        """
        Update dense confusion matrix from one batch.

        preds_grid:  (B, C, H, W), sigmoid probs
        labels_grid: (B, C, H, W), binary GT
        """
        B, C, H, W = preds_grid.shape
        nc = self.num_classes or C

        if self.confusion_matrix is None:
            self.confusion_matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)

        max_probs, pred_classes = preds_grid.max(dim=1)
        pred_classes = pred_classes.clone()
        pred_classes[max_probs < conf_thresh] = nc

        gt_classes = torch.full((B, H, W), nc, dtype=torch.long, device=preds_grid.device)
        for c in range(C):
            gt_classes[labels_grid[:, c, :, :] > 0] = c

        gt_flat = gt_classes.reshape(-1)
        pred_flat = pred_classes.reshape(-1)

        indices = gt_flat * (nc + 1) + pred_flat
        counts = torch.bincount(indices, minlength=(nc + 1) ** 2)
        cm_batch = counts.reshape(nc + 1, nc + 1).cpu().numpy()

        self.confusion_matrix += cm_batch

    def _plot_curves(self, stats, names, unique_classes, ap):
        if not self.output_dir:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        tp, conf, pred_cls, target_cls = stats

        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        _, nt = np.unique(target_cls, return_counts=True)
        px = np.linspace(0, 1, 1000)
        py = []
        eps = 1e-16

        for ci, c in enumerate(unique_classes):
            mask = pred_cls == c
            n_l = nt[ci]
            n_p = mask.sum()

            if n_p == 0 or n_l == 0:
                py.append(np.zeros_like(px))
                continue

            fpc = (1 - tp[mask]).cumsum(0)
            tpc = tp[mask].cumsum(0)
            recall = tpc / (n_l + eps)
            precision = tpc / (tpc + fpc + eps)

            _, mpre, mrec = compute_ap(recall[:, 0], precision[:, 0])
            py.append(np.interp(px, mrec, mpre))

        if len(py) == 0:
            return

        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        py = np.stack(py, axis=1)

        if 0 < len(names) < 21:
            for i, y in enumerate(py.T):
                c_name = names.get(i, f"Class_{i}")
                ax.plot(px, y, linewidth=1, label=f'{c_name} {ap[i, 0]:.3f}')
        else:
            ax.plot(px, py, linewidth=1, color='grey')

        ax.plot(px, py.mean(1), linewidth=3, color='blue',
                label=f'all classes {ap[:, 0].mean():.3f} mAP@0.5')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.set_title('Grid Classification PR Curve')

        pr_path = os.path.join(self.output_dir, 'grid_PR_curve.png')
        fig.savefig(pr_path, dpi=250)
        plt.close(fig)
        print(f"PR curve saved to: {pr_path}")

    def synchronize_between_processes(self):
        from ...misc import dist_utils

        if not dist_utils.is_dist_available_and_initialized():
            return

        all_stats = dist_utils.all_gather(self.stats_grid)
        merged_stats = []
        for stats_list in all_stats:
            merged_stats.extend(stats_list)
        self.stats_grid = merged_stats

        all_seen = dist_utils.all_gather(self.seen)
        self.seen = sum(all_seen)

        if self.confusion_matrix is not None:
            cm_tensor = torch.tensor(self.confusion_matrix, device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)
            self.confusion_matrix = cm_tensor.cpu().numpy()

        print(f"[GridEval] Synchronized stats from processes: {self.seen} images total")

    def cleanup(self):
        self.stats_grid = []
        self.confusion_matrix = None
        self.seen = 0

    def _plot_confusion_matrix(self, confusion_matrix, save_dir=None,
                               title='Grid Dense Confusion Matrix', color_theme='plasma'):
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if save_dir is None:
            save_dir = self.output_dir
        if save_dir is None:
            return

        os.makedirs(save_dir, exist_ok=True)

        labels = self._get_labels(include_background=True)
        num_labels = len(labels)

        per_label_sums = confusion_matrix.sum(axis=1, keepdims=True)
        per_label_sums[per_label_sums == 0] = 1
        normalized_matrix = confusion_matrix.astype(np.float32) / per_label_sums * 100.0

        fig_size = max(8, 1.5 * num_labels)
        fig, ax = plt.subplots(figsize=(fig_size + 1.5, fig_size), dpi=150)

        cmap = plt.get_cmap(color_theme)
        im = ax.imshow(normalized_matrix, cmap=cmap, vmin=0, vmax=100, aspect='equal')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        plt.colorbar(mappable=im, cax=cax)

        ax.set_title(title, fontdict={'weight': 'bold', 'size': 16}, pad=20)
        ax.set_ylabel('Ground Truth', fontdict={'size': 14}, labelpad=10)
        ax.set_xlabel('Prediction', fontdict={'size': 14}, labelpad=10)

        ax.set_xticks(np.arange(num_labels))
        ax.set_yticks(np.arange(num_labels))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticklabels(labels, fontsize=12, rotation=90, va='center')
        plt.setp(ax.get_xticklabels(), ha='center', rotation_mode='anchor')

        for i in range(num_labels):
            for j in range(num_labels):
                value = normalized_matrix[i, j]
                text = f"{value:.1f}%"
                text_color = 'white' if value < 50 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        color=text_color, size=11, weight='bold')

        fig.tight_layout()

        cm_path = os.path.join(save_dir, 'grid_confusion_matrix.png')
        fig.savefig(cm_path, format='png')
        plt.close(fig)
        print(f"Grid confusion matrix saved to: {cm_path}")

        cm_txt_path = os.path.join(save_dir, 'grid_confusion_matrix.txt')
        with open(cm_txt_path, 'w') as f:
            f.write(f"Grid Dense Confusion Matrix (conf_thresh={self.conf_thresh})\n")
            f.write("Rows: Ground Truth, Cols: Prediction\n")
            f.write(f"Labels: {labels}\n\n")
            f.write("Raw counts:\n")
            f.write(np.array2string(confusion_matrix, separator=', '))
            f.write("\n\nNormalized (%):\n")
            f.write(np.array2string(normalized_matrix, precision=1, separator=', '))
        print(f"Grid confusion matrix data saved to: {cm_txt_path}")
