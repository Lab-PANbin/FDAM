"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.ops import box_iou

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...core import register
from ...misc import dist_utils
__all__ = ['CocoEvaluator',]


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, output_dir=None, conf_thresh=0.3, iou_thresh=0.5):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)

        # Filter out negative category IDs (grid labels) for multi-task training
        # Only keep detection classes (category_id >= 0)
        if hasattr(coco_gt, 'dataset') and 'annotations' in coco_gt.dataset:
            coco_gt.dataset['annotations'] = [
                ann for ann in coco_gt.dataset['annotations']
                if ann.get('category_id', 0) >= 0
            ]
            # Filter categories as well
            if 'categories' in coco_gt.dataset:
                coco_gt.dataset['categories'] = [
                    cat for cat in coco_gt.dataset['categories']
                    if cat.get('id', 0) >= 0
                ]
            # Rebuild index
            coco_gt.createIndex()

        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types
        self.output_dir = output_dir
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}


        self.confusion_data = []


        self.num_classes = len(coco_gt.getCatIds()) if hasattr(coco_gt, 'getCatIds') else 0
        self.class_names = None
        self.cat_id_to_idx = {}  # category_id -> 0-indexed class index
        if hasattr(coco_gt, 'cats') and coco_gt.cats:
            sorted_cats = sorted(coco_gt.cats.items(), key=lambda x: x[0])
            self.class_names = [cat['name'] for _, cat in sorted_cats]
            self.num_classes = len(self.class_names)

            for idx, (cat_id, _) in enumerate(sorted_cats):
                self.cat_id_to_idx[cat_id] = idx

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        self.confusion_data = []


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs


        if dist_utils.is_dist_available_and_initialized():
            all_confusion_data = dist_utils.all_gather(self.confusion_data)
            merged_confusion_data = []
            for data_list in all_confusion_data:
                merged_confusion_data.extend(data_list)
            self.confusion_data = merged_confusion_data

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self, plot_confusion_matrix=True):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()


        if plot_confusion_matrix and self.output_dir and self.confusion_data and dist_utils.is_main_process():
            cm = self._compute_confusion_matrix()
            self._plot_confusion_matrix(cm)

    def update_confusion_matrix(self, targets, results):
        """Accumulate predictions and targets for the confusion matrix."""
        for target, result in zip(targets, results):
            gt_boxes = target['boxes'].clone()
            gt_labels = target['labels']

            pred_boxes = result['boxes']
            pred_labels = result['labels']
            pred_scores = result['scores']


            pos_mask = gt_labels >= 0
            gt_boxes = gt_boxes[pos_mask]
            gt_labels = gt_labels[pos_mask]


            keep = pred_scores >= self.conf_thresh
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]


            if len(gt_boxes) > 0 and 'orig_size' in target:
                orig_size = target['orig_size']  # [W, H]
                orig_w, orig_h = orig_size[0].float(), orig_size[1].float()


                cx, cy, w, h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]


                cx = cx * orig_w
                cy = cy * orig_h
                w = w * orig_w
                h = h * orig_h

                # cxcywh -> xyxy
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                gt_boxes = torch.stack([x1, y1, x2, y2], dim=1)

            self._analyze_per_img_dets(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)

    def _analyze_per_img_dets(self, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores):
        """Match detections for one image and accumulate confusion counts."""
        num_gt = len(gt_boxes)
        num_det = len(pred_boxes)

        if num_gt == 0 and num_det == 0:
            return

        if num_det == 0:

            for label in gt_labels:
                gt_idx = self._label_to_idx(label.item())
                self.confusion_data.append((gt_idx, -1))
            return

        if num_gt == 0:

            for label in pred_labels:
                pred_idx = self._label_to_idx(label.item())
                self.confusion_data.append((-1, pred_idx))
            return


        ious = box_iou(gt_boxes, pred_boxes)  # (M, N)


        gt_used = torch.zeros(num_gt, dtype=torch.bool, device=gt_boxes.device)
        det_used = torch.zeros(num_det, dtype=torch.bool, device=pred_boxes.device)

        if ious.numel() > 0:
            vals, inds = torch.sort(ious.flatten(), descending=True)

            for v, ind in zip(vals, inds):
                if v < self.iou_thresh:
                    break

                gt_i = ind // num_det
                det_i = ind % num_det

                if not gt_used[gt_i] and not det_used[det_i]:
                    gt_label = gt_labels[gt_i].item()
                    det_label = pred_labels[det_i].item()

                    gt_idx = self._label_to_idx(gt_label)
                    det_idx = self._label_to_idx(det_label)

                    self.confusion_data.append((gt_idx, det_idx))

                    gt_used[gt_i] = True
                    det_used[det_i] = True


        for i in range(num_det):
            if not det_used[i]:
                det_label = pred_labels[i].item()
                det_idx = self._label_to_idx(det_label)
                self.confusion_data.append((-1, det_idx))


        for i in range(num_gt):
            if not gt_used[i]:
                gt_label = gt_labels[i].item()
                gt_idx = self._label_to_idx(gt_label)
                self.confusion_data.append((gt_idx, -1))

    def _label_to_idx(self, label):
        """Map a category identifier to its zero-based class index."""
        if label in self.cat_id_to_idx:
            return self.cat_id_to_idx[label]

        return label

    def _compute_confusion_matrix(self):
        """Compute the detection confusion matrix."""
        nc = self.num_classes
        # +1 for background
        cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)

        for gt_idx, pred_idx in self.confusion_data:


            gt_row = gt_idx if gt_idx >= 0 else nc  # -1 -> nc (background row)
            pred_col = pred_idx if pred_idx >= 0 else nc  # -1 -> nc (background col)
            cm[gt_row, pred_col] += 1

        return cm

    def _plot_confusion_matrix(self, confusion_matrix, title='Detection Confusion Matrix', color_theme='plasma'):
        """Plot the detection confusion matrix."""
        if self.output_dir is None:
            return

        os.makedirs(self.output_dir, exist_ok=True)


        nc = self.num_classes
        if self.class_names and len(self.class_names) >= nc:
            labels = list(self.class_names[:nc]) + ['Background']
        else:
            labels = [f'Class_{i}' for i in range(nc)] + ['Background']

        num_labels = len(labels)


        per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
        per_label_sums[per_label_sums == 0] = 1
        normalized_matrix = confusion_matrix.astype(np.float32) / per_label_sums * 100


        fig_size = max(8, 1.5 * num_labels)
        fig, ax = plt.subplots(figsize=(fig_size + 1.5, fig_size), dpi=150)

        cmap = plt.get_cmap(color_theme)
        im = ax.imshow(normalized_matrix, cmap=cmap, vmin=0, vmax=100, aspect='equal')


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        plt.colorbar(mappable=im, cax=cax)


        title_font = {'weight': 'bold', 'size': 16}
        ax.set_title(title, fontdict=title_font, pad=20)
        label_font = {'size': 14}
        ax.set_ylabel('Ground Truth', fontdict=label_font, labelpad=10)
        ax.set_xlabel('Prediction', fontdict=label_font, labelpad=10)


        ax.set_xticks(np.arange(num_labels))
        ax.set_yticks(np.arange(num_labels))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticklabels(labels, fontsize=12, rotation=90, va='center')

        plt.setp(ax.get_xticklabels(), ha='center', rotation_mode='anchor')


        for i in range(num_labels):
            for j in range(num_labels):
                value = normalized_matrix[i, j]
                text = f"{value:.1f}%"


                if color_theme == 'plasma':
                    text_color = 'white' if value < 50 else 'black'
                else:
                    text_color = 'white'

                ax.text(j, i, text, ha='center', va='center', color=text_color, size=11, weight='bold')

        fig.tight_layout()


        cm_path = os.path.join(self.output_dir, 'detection_confusion_matrix.png')
        fig.savefig(cm_path, format='png')
        plt.close(fig)
        print(f"Detection confusion matrix saved to: {cm_path}")


        cm_txt_path = os.path.join(self.output_dir, 'detection_confusion_matrix.txt')
        with open(cm_txt_path, 'w') as f:
            f.write(f"Detection Confusion Matrix (conf_thresh={self.conf_thresh}, iou_thresh={self.iou_thresh})\n")
            f.write(f"Rows: Ground Truth, Cols: Prediction\n")
            f.write(f"Labels: {labels}\n\n")
            f.write("Raw counts:\n")
            f.write(np.array2string(confusion_matrix, separator=', '))
            f.write("\n\nNormalized (%):\n")
            f.write(np.array2string(normalized_matrix, precision=1, separator=', '))
        print(f"Detection confusion matrix data saved to: {cm_txt_path}")

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            # Filter out negative category IDs (grid labels) - only keep detection predictions
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                    if labels[k] >= 0  # Only include detection classes
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
