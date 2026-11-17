"""
FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy
import math

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register


class FeatureDirectionLoss(nn.Module):
    """Optional feature-direction consistency loss for longitudinal and transverse cracks."""
    def __init__(self, lc_idx=0, tc_idx=1, weight=1.0, feature_channels=256):
        super().__init__()
        self.lc_idx = lc_idx
        self.tc_idx = tc_idx
        self.weight = weight
        self.eps = 1e-6


        self.vertical_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 4, (7, 1), padding=(3, 0),
                      groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 4, 1, 1, bias=False),
        )


        self.horizontal_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 4, (1, 7), padding=(0, 3),
                      groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 4, 1, 1, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, pred_logits, features, target_grid=None):
        """Apply the module to the input tensors."""
        B, C, H, W = pred_logits.shape


        if features.shape[2:] != (H, W):
            features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)


        if self.vertical_conv[0].weight.device != features.device or \
           self.vertical_conv[0].weight.dtype != features.dtype:
            self.vertical_conv = self.vertical_conv.to(device=features.device, dtype=features.dtype)
            self.horizontal_conv = self.horizontal_conv.to(device=features.device, dtype=features.dtype)


        v_response = self.vertical_conv(features)    # (B, 1, H, W)
        h_response = self.horizontal_conv(features)  # (B, 1, H, W)


        direction_diff = v_response - h_response
        vertical_dominance = torch.sigmoid(direction_diff)  # (B, 1, H, W)


        probs = torch.sigmoid(pred_logits)
        p_lc = probs[:, self.lc_idx:self.lc_idx+1]  # (B, 1, H, W)
        p_tc = probs[:, self.tc_idx:self.tc_idx+1]  # (B, 1, H, W)


        if target_grid is not None:

            lc_mask = (target_grid[:, self.lc_idx:self.lc_idx+1] > 0.5).float()
            tc_mask = (target_grid[:, self.tc_idx:self.tc_idx+1] > 0.5).float()


            lc_loss = F.binary_cross_entropy(
                vertical_dominance * lc_mask + 0.5 * (1 - lc_mask),
                lc_mask,  # target = 1 for LC positions
                reduction='none'
            )
            lc_loss = (lc_loss * lc_mask).sum() / (lc_mask.sum() + self.eps)


            tc_loss = F.binary_cross_entropy(
                vertical_dominance * tc_mask + 0.5 * (1 - tc_mask),
                torch.zeros_like(tc_mask),  # target = 0 for TC positions
                reduction='none'
            )
            tc_loss = (tc_loss * tc_mask).sum() / (tc_mask.sum() + self.eps)
        else:


            lc_loss = (p_lc * F.binary_cross_entropy(vertical_dominance, torch.ones_like(vertical_dominance), reduction='none')).mean()

            tc_loss = (p_tc * F.binary_cross_entropy(vertical_dominance, torch.zeros_like(vertical_dominance), reduction='none')).mean()

        loss = self.weight * (lc_loss + tc_loss)

        return loss


DirectionConsistencyLoss = FeatureDirectionLoss


class DetectionGridConsistencyLoss(nn.Module):
    """Optional detection-guided grid consistency loss after warm-up."""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

        self._conf_threshold = 0.3
        self._warmup_epochs = 10

    def forward(self, pred_boxes, pred_logits, pred_grid, epoch=0):
        """Apply the module to the input tensors."""

        if epoch < self._warmup_epochs:
            return torch.tensor(0.0, device=pred_grid.device)

        B, num_queries, num_det_classes = pred_logits.shape
        _, C, H, W = pred_grid.shape
        device = pred_grid.device


        det_probs = torch.sigmoid(pred_logits)
        det_conf, det_cls = det_probs.max(dim=-1)


        valid_mask = det_conf > self._conf_threshold

        if not valid_mask.any():
            return torch.tensor(0.0, device=device)


        det_grid = torch.zeros(B, C, H, W, device=device)

        for b in range(B):
            valid_idx = valid_mask[b]
            if not valid_idx.any():
                continue

            boxes = pred_boxes[b, valid_idx]
            confs = det_conf[b, valid_idx]
            classes = det_cls[b, valid_idx].clamp(0, C - 1)


            gx = (boxes[:, 0] * W).long().clamp(0, W - 1)
            gy = (boxes[:, 1] * H).long().clamp(0, H - 1)


            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx = (gx + dx).clamp(0, W - 1)
                    ny = (gy + dy).clamp(0, H - 1)
                    dist_weight = 1.0 / (1.0 + (dx**2 + dy**2)**0.5)

                    for i in range(len(nx)):
                        c = classes[i].item()
                        det_grid[b, c, ny[i], nx[i]] = max(
                            det_grid[b, c, ny[i], nx[i]].item(),
                            (confs[i] * dist_weight).item()
                        )


        pos_mask = det_grid > 0
        if not pos_mask.any():
            return torch.tensor(0.0, device=device)

        loss = F.binary_cross_entropy_with_logits(
            pred_grid[pos_mask],
            det_grid[pos_mask].detach(),
            reduction='mean'
        )

        return self.weight * loss


@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        use_uncertainty_weighting=False,
        grid_focal_alpha=0.25,
        grid_focal_gamma=2.0,
        grid_focal_weight=0.75,
        # Per-class weighting for grid loss
        grid_class_weight_method=None,  # 'inverse', 'sqrt_inverse', or None
        grid_prior_probs=None,  # Prior probabilities for each class
        # Direction Consistency Loss parameters
        use_direction_loss=False,
        direction_loss_weight=0.1,
        direction_lc_idx=0,
        direction_tc_idx=1,
        grid_delta=0.125,
        cp=1.0,  # positive class value (with label smoothing)
        cn=0.0,  # negative class value (with label smoothing)
        # OHEM and background suppression parameters
        grid_use_ohem=False,  # Online Hard Example Mining
        grid_ohem_ratio=3.0,  # neg:pos ratio for OHEM
        grid_bg_suppression=0.0,  # Background suppression weight (0 = disabled)

        use_det_grid_consistency=False,  # Enable Detection-Grid consistency constraint
        det_grid_consistency_weight=1.0,
        # Adaptive Importance Sampling for Grid Loss (inspired by EdgeCrafter/PointRend)
        grid_use_adaptive_sampling=False,  # Enable uncertainty+boundary weighted sampling
        grid_adaptive_uncertainty_gain=1.0,  # Weight for uncertainty term (higher = more focus on hard pixels)
        grid_adaptive_boundary_gain=1.0,  # Weight for boundary term (higher = more focus on crack edges)
        grid_adaptive_boundary_kernel=3,  # Dilation kernel size for boundary detection
        ):
        """Create the criterion."""
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set

        # Uncertainty weighting for multi-task learning
        self.use_uncertainty_weighting = use_uncertainty_weighting

        self.grid_focal_alpha = grid_focal_alpha
        self.grid_focal_gamma = grid_focal_gamma
        self.grid_focal_weight = grid_focal_weight

        # Per-class weights for grid loss (will be set from grid_head's prior_probs)
        self.grid_class_weights = None  # Will be computed from prior_probs

        # Initialize class weights if prior_probs provided
        if grid_prior_probs is not None and grid_class_weight_method is not None:
            self.set_grid_class_weights(grid_prior_probs, method=grid_class_weight_method)


        print(f"[DEIMCriterion] Grid Focal Loss: alpha={grid_focal_alpha}, gamma={grid_focal_gamma}")

        self.grid_delta = grid_delta
        self.cn = cn
        self.cp = cp

        # OHEM and background suppression
        self.grid_use_ohem = grid_use_ohem
        self.grid_ohem_ratio = grid_ohem_ratio
        self.grid_bg_suppression = grid_bg_suppression

        if self.grid_use_ohem:
            print(f"[DEIMCriterion] Grid OHEM enabled: neg:pos ratio={grid_ohem_ratio}")
        if self.grid_bg_suppression > 0:
            print(f"[DEIMCriterion] Grid Background Suppression enabled: weight={grid_bg_suppression}")

        # Adaptive Importance Sampling for Grid Loss (inspired by EdgeCrafter / PointRend)
        self.grid_use_adaptive_sampling = grid_use_adaptive_sampling
        self.grid_adaptive_uncertainty_gain = grid_adaptive_uncertainty_gain
        self.grid_adaptive_boundary_gain = grid_adaptive_boundary_gain
        self.grid_adaptive_boundary_kernel = grid_adaptive_boundary_kernel
        if self.grid_use_adaptive_sampling:
            print(f"[DEIMCriterion] Grid Adaptive Sampling enabled: "
                  f"uncertainty_gain={grid_adaptive_uncertainty_gain}, "
                  f"boundary_gain={grid_adaptive_boundary_gain}, "
                  f"boundary_kernel={grid_adaptive_boundary_kernel}")

        # Direction Consistency Loss for LC/TC classification
        self.use_direction_loss = use_direction_loss
        self.direction_feature_channels = 256
        if self.use_direction_loss:
            self.direction_loss = FeatureDirectionLoss(
                lc_idx=direction_lc_idx,
                tc_idx=direction_tc_idx,
                weight=direction_loss_weight,
                feature_channels=self.direction_feature_channels
            )
            print(f"[DEIMCriterion] Feature Direction Loss enabled: weight={direction_loss_weight}")


        self.use_det_grid_consistency = use_det_grid_consistency
        if self.use_det_grid_consistency:
            self.det_grid_consistency_loss = DetectionGridConsistencyLoss(
                weight=det_grid_consistency_weight
            )
            print(f"[DEIMCriterion] Detection-Grid Consistency Loss enabled: weight={det_grid_consistency_weight}")

        if self.use_uncertainty_weighting:
            # Learnable log variance for each task
            # Detection tasks: vfl, bbox, giou, fgl, ddf
            self.log_var_detection = nn.Parameter(torch.zeros(1))
            # Grid task
            self.log_var_grid = nn.Parameter(torch.zeros(1))

    def set_grid_class_weights(self, prior_probs, method='inverse'):
        """
        Set per-class weights for grid loss based on prior probabilities.

        Args:
            prior_probs: List or tensor of prior probabilities for each class
            method: Weighting method
                - 'inverse': weight = 1 / prior_prob (normalized)
                - 'effective': weight = (1 - beta) / (1 - beta^n) where n = num_samples
                - 'sqrt_inverse': weight = 1 / sqrt(prior_prob) (softer)
        """
        if prior_probs is None:
            return

        prior_probs = torch.tensor(prior_probs, dtype=torch.float32)
        eps = 1e-8
        prior_probs = torch.clamp(prior_probs, eps, 1.0)

        if method == 'inverse':
            # Inverse frequency weighting
            weights = 1.0 / prior_probs
        elif method == 'sqrt_inverse':
            # Softer inverse weighting
            weights = 1.0 / torch.sqrt(prior_probs)
        elif method == 'effective':
            # Effective number of samples weighting (Class-Balanced Loss)
            beta = 0.9999
            effective_num = 1.0 - torch.pow(beta, prior_probs * 1e6)  # Approximate sample count
            weights = (1.0 - beta) / (effective_num + eps)
        else:
            weights = torch.ones_like(prior_probs)

        # Normalize weights so mean = 1 (preserves overall loss scale)
        weights = weights / weights.mean()

        self.grid_class_weights = weights
        print(f"[DEIMCriterion] Grid class weights ({method}): {weights.tolist()}")

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target


        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    def box2grid(self, pred_grid, boxes, labels, grid_delta=0.125, cp=0.95, cn=0.05):
        """Maps box center points to grid cells and expands to 3x3 neighborhood."""
        B, C, H, W = pred_grid.shape
        device = pred_grid.device

        # Initialize with negative class value
        grid = torch.full_like(pred_grid, cn)

        if boxes.numel() == 0:
            return grid

        # Extract components
        b = boxes[:, 0].long()  # batch index
        c = boxes[:, 1].long()  # class index (already 0-based)
        cx = boxes[:, 2]  # center x (normalized)
        cy = boxes[:, 3]  # center y (normalized)
        w = boxes[:, 4]   # width (normalized)
        h = boxes[:, 5]   # height (normalized)

        # 3x3 neighborhood offsets
        # offsets = [(-1, -1), (-1, 0), (-1, 1),
        #            (0, -1),  (0, 0),  (0, 1),
        #            (1, -1),  (1, 0),  (1, 1)]

        offsets = [(-1, -1), (-1, 1), (1, 1), (1, -1), (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

        for dh, dw in offsets:
            # Compute offset position based on box size
            y_float = (cy + dh * grid_delta * h) * H
            x_float = (cx + dw * grid_delta * w) * W

            # Convert to integer indices
            y = y_float.long()
            x = x_float.long()

            # Filter valid indices (check all boundaries)
            # valid = (b >= 0) & (b < B) & (c >= 0) & (c < C) & \
            #         (y >= 0) & (y < H) & (x >= 0) & (x < W)
            valid = (b >= 0) & (b < B) & (c >= 0) & (c < C)

            if valid.any():
                # grid[b[valid], c[valid], y[valid], x[valid]] = cp
                b_v, c_v, y_v, x_v = b[valid], c[valid], y[valid], x[valid]
                spatial_valid = (y_v >= 0) & (y_v < H) & (x_v >= 0) & (x_v < W)

                if spatial_valid.any():
                    grid[b_v[spatial_valid], c_v[spatial_valid],
                         y_v[spatial_valid], x_v[spatial_valid]] = cp

        return grid

    def _compute_adaptive_importance(self, pred_probs, target_grid):
        """Compute per-pixel importance weights inspired by EdgeCrafter / PointRend."""
        B, C, H, W = pred_probs.shape
        device = pred_probs.device

        # --- Uncertainty: model is uncertain when prob ≈ 0.5 ---
        # |2*p - 1| ranges [0, 1]; 0 = most uncertain, 1 = most certain
        certainty = (2.0 * pred_probs - 1.0).abs()
        uncertainty = 1.0 - certainty  # [0, 1], 1 = most uncertain

        # Only consider uncertainty at pixels that are actually supervised
        # (both positive and negative matter, but background-only regions less so)
        is_supervised = (target_grid > 0.01) | (target_grid < -0.01)  # any label present
        # For a simpler approach, we apply uncertainty everywhere

        # --- Boundary: near positive-negative transitions ---
        # Use max-pool dilation to find boundary regions in target_grid
        kernel_size = self.grid_adaptive_boundary_kernel
        padding = kernel_size // 2

        # For each class channel, dilate the positive region
        target_float = target_grid.float()
        # max_pool2d on target gives dilated positive mask
        dilated = F.max_pool2d(
            target_float, kernel_size=kernel_size, stride=1, padding=padding
        )
        eroded = -F.max_pool2d(
            -target_float, kernel_size=kernel_size, stride=1, padding=padding
        )
        boundary_mask = (dilated - eroded).clamp(0.0, 1.0)  # 1 on boundaries, 0 elsewhere

        # --- Combine ---
        importance = 1.0 \
            + self.grid_adaptive_uncertainty_gain * uncertainty \
            + self.grid_adaptive_boundary_gain * boundary_mask

        # Clamp to avoid extreme values
        importance = importance.clamp(1.0, 10.0)

        return importance.detach()

    def loss_grid(self, outputs, targets_list):
        """
        Key differences from previous implementation:
        Uses box CENTER POINT instead of entire box region
        Expands positive samples to 3x3 neighborhood around center
        Simpler and more effective for crack detection
        """
        pred_grid_logits = outputs.get('pred_grid_logits')  # (B, num_classes, H, W)
        if pred_grid_logits is None:
            return {}

        pred_grid_logits = torch.clamp(pred_grid_logits, min=-20.0, max=20.0)
        bs, num_grid_classes, H, W = pred_grid_logits.shape
        device = pred_grid_logits.device


        grid_features = outputs.get('grid_features')

        # Collect all grid boxes into a single tensor for vectorized processing
        # Format: [batch_idx, class_id, cx, cy, w, h]
        grid_boxes_list = []

        for i, target in enumerate(targets_list):
            if 'labels' not in target:
                continue

            all_labels = target['labels']
            all_boxes = target['boxes']  # cx, cy, w, h (normalized)

            # Get grid boxes (negative labels)
            grid_indices = all_labels < 0
            if not grid_indices.any():
                continue

            grid_labels = all_labels[grid_indices]
            boxes = all_boxes[grid_indices]

            # Convert negative labels to 0-based: -1->0, -2->1, etc.
            class_ids = (abs(grid_labels) - 1).float()

            # Create batch index column
            batch_idx = torch.full((len(boxes),), i, dtype=torch.float32, device=device)

            # Stack: [batch_idx, class_id, cx, cy, w, h]
            grid_box_data = torch.stack([
                batch_idx,
                class_ids,
                boxes[:, 0],  # cx
                boxes[:, 1],  # cy
                boxes[:, 2],  # w
                boxes[:, 3],  # h
            ], dim=1)

            grid_boxes_list.append(grid_box_data)

        # Concatenate all grid boxes
        if grid_boxes_list:
            all_grid_boxes = torch.cat(grid_boxes_list, dim=0)
        else:
            all_grid_boxes = torch.empty((0, 6), device=device)

        target_grid = self.box2grid(
            pred_grid_logits,
            all_grid_boxes,
            labels=None,
            grid_delta=self.grid_delta,
            cp=self.cp,
            cn=self.cn
        )

        # Check if using simple BCE loss (gamma=0 disables focal weighting)
        use_simple_bce = (self.grid_focal_gamma == 0.0)

        if use_simple_bce:
            # Simple BCE with mean reduction (no focal weighting)
            loss = F.binary_cross_entropy_with_logits(
                pred_grid_logits, target_grid, reduction='mean'
            )
        else:
            # Focal Loss implementation with per-class weighting
            pred_probs = torch.sigmoid(pred_grid_logits)

            # BCE loss (per-element)
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_grid_logits, target_grid, reduction='none'
            )

            # Focal weight: (1 - p_t)^gamma
            p_t = pred_probs * target_grid + (1 - pred_probs) * (1 - target_grid)
            focal_weight = (1 - p_t) ** self.grid_focal_gamma

            # Per-class weighting based on inverse prior probability
            # This gives higher weight to rare classes (like LC)
            if self.grid_class_weights is not None:
                # grid_class_weights shape: (num_classes,) -> expand to (1, C, 1, 1)
                class_weights = self.grid_class_weights.view(1, -1, 1, 1).to(device=device, dtype=pred_probs.dtype)
                # Apply class weights to positive samples only
                # For negatives, use uniform weight
                pos_class_weight = class_weights * target_grid + (1 - target_grid)
            else:
                pos_class_weight = 1.0

            # Alpha weighting for positive/negative balance
            alpha_t = self.grid_focal_alpha * target_grid + (1 - self.grid_focal_alpha) * (1 - target_grid)

            # Combined focal loss with per-class weighting
            focal_loss = pos_class_weight * alpha_t * focal_weight * bce_loss

            # ----- Adaptive Importance Sampling (EdgeCrafter / PointRend inspired) -----
            if self.grid_use_adaptive_sampling:
                importance = self._compute_adaptive_importance(pred_probs.detach(), target_grid)
                focal_loss = focal_loss * importance


            # Identify positive and negative samples
            pos_mask = target_grid > 0.5
            neg_mask = ~pos_mask
            num_pos = pos_mask.sum().clamp(min=1.0)

            # Apply OHEM if enabled
            if self.grid_use_ohem and neg_mask.any():
                # Online Hard Example Mining: keep only top-k hard negatives
                num_neg_to_keep = int(num_pos.item() * self.grid_ohem_ratio)
                num_neg_to_keep = max(num_neg_to_keep, 1)

                # Get negative losses
                neg_losses = focal_loss[neg_mask]

                if len(neg_losses) > num_neg_to_keep:
                    # Keep only the hardest negatives (highest loss)
                    _, topk_indices = torch.topk(neg_losses.view(-1), num_neg_to_keep)
                    neg_loss_ohem = neg_losses.view(-1)[topk_indices].sum()
                else:
                    neg_loss_ohem = neg_losses.sum()

                # Positive loss (all positives)
                pos_loss = focal_loss[pos_mask].sum()

                loss = (pos_loss + neg_loss_ohem) / num_pos
            else:
                # Standard focal loss
                loss = focal_loss.sum() / num_pos

            # Background suppression: penalize high predictions in background regions
            if self.grid_bg_suppression > 0:
                # Background = negative samples with high predictions
                bg_pred = pred_probs[neg_mask]
                # Soft penalty: (pred - threshold)^2 for pred > threshold
                bg_threshold = 0.3
                bg_violation = F.relu(bg_pred - bg_threshold)
                bg_loss = (bg_violation ** 2).mean()
                loss = loss + self.grid_bg_suppression * bg_loss

        losses = {"loss_grid": loss}

        # Add Feature Direction Loss if enabled
        if self.use_direction_loss and grid_features is not None:

            feat_channels = grid_features.shape[1]
            if feat_channels != self.direction_feature_channels:

                self.direction_feature_channels = feat_channels
                self.direction_loss = FeatureDirectionLoss(
                    lc_idx=self.direction_loss.lc_idx,
                    tc_idx=self.direction_loss.tc_idx,
                    weight=self.direction_loss.weight,
                    feature_channels=feat_channels
                ).to(device=grid_features.device, dtype=grid_features.dtype)
                print(f"[DEIMCriterion] Reinitialized FeatureDirectionLoss for {feat_channels} channels")

            dir_loss = self.direction_loss(pred_grid_logits, grid_features, target_grid)
            losses["loss_grid_dir"] = dir_loss

        return losses

    def forward(self, outputs, targets, epoch=0, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # A grid-only model has no detection outputs or Hungarian matching.
        if self.losses == ['grid']:
            losses = {k: v * self.weight_dict.get(k, 1.0)
                      for k, v in self.loss_grid(outputs, targets).items()}
            if self.use_uncertainty_weighting:
                losses = self._apply_uncertainty_weighting(losses)
            return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

        # Filter targets for detection (remove grid labels)
        det_targets = []
        for target in targets:
            if 'labels' in target:
                det_indices = target['labels'] >= 0
                det_labels = target['labels'][det_indices]
                det_boxes = target['boxes'][det_indices]
            else:
                det_labels = torch.tensor([], device=target['boxes'].device)
                det_boxes = torch.empty((0, 4), device=target['boxes'].device)

            det_target = {
                'labels': det_labels,
                'boxes': det_boxes
            }
            for key in ['image_id', 'area', 'iscrowd', 'orig_size', 'size']:
                if key in target:
                    det_target[key] = target[key]
            det_targets.append(det_target)

        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, det_targets, epoch=epoch)['indices']
        self._clear_cache()

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # (moved before aux_outputs processing so num_boxes is available)
        num_boxes = sum(len(t["labels"]) for t in det_targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, det_targets, epoch=epoch)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, det_targets, epoch=epoch)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            # In eval mode, aux_outputs may not be present - use main indices only
            indices_go = indices
            num_boxes_go = num_boxes

        # Check if we have any detection targets
        # has_detection_targets = num_boxes > 1.0  # num_boxes is clamped to min=1, so check > 1

        # Debug: Count positive and negative labels (commented out to avoid affecting loss)
        # num_pos_labels = sum((t['labels'] >= 0).sum().item() for t in targets if 'labels' in t)
        # num_neg_labels = sum((t['labels'] < 0).sum().item() for t in targets if 'labels' in t)

        # Compute all the requested losses, main loss
        losses = {}
        # device = next(iter(outputs.values())).device


        for loss in self.losses:
            if loss == 'grid':
                continue
            # Skip detection losses if no detection targets
            # if not has_detection_targets:
            #    continue
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, det_targets, indices_in)
            l_dict = self.get_loss(loss, outputs, det_targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss == 'grid':
                        continue
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, det_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, det_targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                if loss == 'grid':
                    continue
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, det_targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, det_targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(det_targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = det_targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    if loss == 'grid':
                        continue
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], det_targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss == 'grid':
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, det_targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, det_targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    if loss == 'grid':
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, det_targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, det_targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Compute grid loss if requested (using original targets which contain grid labels)
        if 'grid' in self.losses:
            l_dict = self.loss_grid(outputs, targets)
            # Apply weights from weight_dict, use 1.0 for losses not in weight_dict (e.g., direction loss)
            weighted_l_dict = {}
            for k, v in l_dict.items():
                if k in self.weight_dict:
                    weighted_l_dict[k] = v * self.weight_dict[k]
                else:
                    # Direction consistency loss uses its own internal weight
                    weighted_l_dict[k] = v
            losses.update(weighted_l_dict)

            # Detection-Grid Consistency Loss

            if self.use_det_grid_consistency and 'pred_grid_logits' in outputs:
                pred_boxes = outputs.get('pred_boxes')
                pred_logits = outputs.get('pred_logits')
                pred_grid = outputs.get('pred_grid_logits')

                if pred_boxes is not None and pred_logits is not None and pred_grid is not None:
                    consistency_loss = self.det_grid_consistency_loss(
                        pred_boxes, pred_logits, pred_grid, epoch=epoch
                    )
                    losses['loss_det_grid_consistency'] = consistency_loss

        # Apply uncertainty weighting for multi-task learning
        if self.use_uncertainty_weighting:
            losses = self._apply_uncertainty_weighting(losses)

        # Keep the returned loss dictionary finite.
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def _apply_uncertainty_weighting(self, losses):
        """
        Apply uncertainty weighting to balance multiple tasks.
        Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses"

        Formula: L_weighted = (1 / (2 * σ²)) * L + log(σ)
                            = (1 / (2 * exp(log_var))) * L + log_var / 2
        """
        weighted_losses = {}

        # Separate detection and grid losses
        detection_loss = 0.0
        grid_loss = 0.0

        for key, value in losses.items():
            # Grid-related losses (including direction consistency loss)
            if 'grid' in key:
                grid_loss += value
            else:
                # All other losses are detection-related
                detection_loss += value

        # Apply uncertainty weighting
        # Detection: L_det / (2 * exp(log_var_det)) + log_var_det / 2
        precision_det = torch.exp(-self.log_var_detection)
        weighted_detection = precision_det * detection_loss + self.log_var_detection / 2

        # Grid: L_grid / (2 * exp(log_var_grid)) + log_var_grid / 2
        precision_grid = torch.exp(-self.log_var_grid)
        weighted_grid = precision_grid * grid_loss + self.log_var_grid / 2

        # Reconstruct losses dict with weighted values
        # Scale individual losses proportionally
        if detection_loss > 0:
            detection_scale = weighted_detection / (detection_loss + 1e-8)
        else:
            detection_scale = 1.0

        if grid_loss > 0:
            grid_scale = weighted_grid / (grid_loss + 1e-8)
        else:
            grid_scale = 1.0

        for key, value in losses.items():
            if 'grid' in key:
                weighted_losses[key] = value * grid_scale



            else:
                weighted_losses[key] = value * detection_scale

        # Add uncertainty parameters to losses for monitoring
        weighted_losses['uncertainty_detection'] = self.log_var_detection.detach()
        weighted_losses['uncertainty_grid'] = self.log_var_grid.detach()
        weighted_losses['weight_detection'] = precision_det.detach()
        weighted_losses['weight_grid'] = precision_grid.detach()

        return weighted_losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
