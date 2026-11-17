"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

from ..core import register
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid, bias_init_with_prob

from .dfine_decoder import MSDeformableAttention, LQE, Integral
from .dfine_utils import weighting_function, distance2bbox
from .deim_utils import RMSNorm, SwiGLUFFN, Gate, MLP

__all__ = ['DEIMTransformer']

class BoxConditionedGridContextInteraction(nn.Module):
    """
    Box-conditioned Grid Context Aggregation.

    Main changes:
      1. Use grid confidence weighted pooling instead of plain mean pooling.
      2. Use small non-zero initial gate, so the module can learn from the start.
      3. Designed mainly for score refinement, not box regression.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        grid_dim: int = None,
        roi_size: int = 5,
        alpha_max: float = 0.2,
        detach_grid: bool = True,
        detach_boxes: bool = True,
        min_box_size: float = 1e-3,
        dropout: float = 0.0,
        use_conf_weight: bool = True,
        conf_power: float = 2.0,
        init_alpha_ratio: float = 0.1,
        conf_is_logits: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.grid_dim = hidden_dim if grid_dim is None else grid_dim
        self.roi_size = int(roi_size)
        self.alpha_max = float(alpha_max)
        self.detach_grid = detach_grid
        self.detach_boxes = detach_boxes
        self.min_box_size = float(min_box_size)
        self.use_conf_weight = use_conf_weight
        self.conf_power = float(conf_power)
        self.conf_is_logits = conf_is_logits

        if self.grid_dim != hidden_dim:
            self.grid_proj = nn.Conv2d(
                self.grid_dim,
                hidden_dim,
                kernel_size=1,
                bias=False,
            )
        else:
            self.grid_proj = nn.Identity()

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)

        self.context_to_film = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        # Extra grid statistics:
        #   mean confidence inside box
        #   max confidence inside box
        #   normalized box area
        self.stats_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Important:
        # Do NOT initialize actual alpha to exact zero.
        # If alpha is exactly zero, context_to_film receives almost no useful
        # gradient at the beginning.
        init_alpha_ratio = min(max(init_alpha_ratio, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_alpha_ratio / (1.0 - init_alpha_ratio))
        self.alpha_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def _sample_roi(
        self,
        feat: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat: Tensor[B, C, H, W]
            boxes: Tensor[B, N, 4], normalized cxcywh

        Returns:
            roi_feat: Tensor[B, N, C, K, K]
        """
        B, C, H, W = feat.shape
        _, N, _ = boxes.shape
        K = self.roi_size

        boxes = boxes.clamp(0.0, 1.0)

        cx, cy, bw, bh = boxes.unbind(dim=-1)

        bw = bw.clamp(min=self.min_box_size)
        bh = bh.clamp(min=self.min_box_size)

        x1 = (cx - 0.5 * bw).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * bh).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * bw).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * bh).clamp(0.0, 1.0)

        lin = torch.linspace(
            0.5 / K,
            1.0 - 0.5 / K,
            K,
            device=feat.device,
            dtype=feat.dtype,
        )

        yy, xx = torch.meshgrid(lin, lin, indexing='ij')

        xs = x1[..., None, None] + xx.view(1, 1, K, K) * (x2 - x1)[..., None, None]
        ys = y1[..., None, None] + yy.view(1, 1, K, K) * (y2 - y1)[..., None, None]

        sample_grid = torch.stack(
            [xs * 2.0 - 1.0, ys * 2.0 - 1.0],
            dim=-1,
        )  # B, N, K, K, 2

        roi_feats = []

        for b in range(B):
            grid_b = sample_grid[b].reshape(1, N * K, K, 2)

            sampled = F.grid_sample(
                feat[b:b + 1],
                grid_b,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )  # 1, C, N*K, K

            sampled = sampled.view(C, N, K, K).permute(1, 0, 2, 3).contiguous()
            roi_feats.append(sampled)

        roi_feats = torch.stack(roi_feats, dim=0)  # B, N, C, K, K

        return roi_feats

    def forward(
        self,
        queries: torch.Tensor,
        grid_feat: torch.Tensor,
        boxes: torch.Tensor,
        grid_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """Args:
            queries:
                Tensor[B, N, C]"""
        if grid_feat is None or boxes is None:
            return queries

        if self.detach_grid:
            grid_feat = grid_feat.detach()
            if grid_logits is not None:
                grid_logits = grid_logits.detach()

        if self.detach_boxes:
            boxes = boxes.detach()

        grid_feat = self.grid_proj(grid_feat)

        roi_feat = self._sample_roi(grid_feat, boxes)  # B, N, C, K, K

        roi_conf = None

        if self.use_conf_weight and grid_logits is not None:
            if isinstance(grid_logits, (list, tuple)):
                # Use the first/highest-resolution logits if a list is passed.
                grid_logits = grid_logits[0]

            if isinstance(grid_logits, dict):
                grid_logits = grid_logits.get(
                    'grid_logits',
                    grid_logits.get('pred_grid_logits', None),
                )

            if grid_logits is not None:
                if grid_logits.shape[1] > 1:
                    grid_logits = grid_logits.max(dim=1, keepdim=True).values

                if grid_logits.shape[-2:] != grid_feat.shape[-2:]:
                    grid_logits = F.interpolate(
                        grid_logits,
                        size=grid_feat.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )

                if self.conf_is_logits:
                    grid_conf = grid_logits.sigmoid()
                else:
                    grid_conf = grid_logits

                grid_conf = grid_conf.clamp(0.0, 1.0)

                roi_conf = self._sample_roi(grid_conf, boxes)  # B, N, 1, K, K

        if roi_conf is not None:
            weight = roi_conf.clamp(min=1e-4).pow(self.conf_power)

            context = (roi_feat * weight).sum(dim=(-1, -2)) / (
                weight.sum(dim=(-1, -2)) + 1e-6
            )  # B, N, C

            conf_mean = roi_conf.mean(dim=(-1, -2)).squeeze(-1)  # B, N
            conf_max = roi_conf.amax(dim=(-1, -2)).squeeze(-1)   # B, N
        else:
            context = roi_feat.mean(dim=(-1, -2))  # B, N, C

            conf_mean = torch.zeros(
                boxes.shape[:2],
                device=boxes.device,
                dtype=boxes.dtype,
            )
            conf_max = torch.zeros_like(conf_mean)

        context = self.context_norm(context)

        gamma, beta = self.context_to_film(context).chunk(2, dim=-1)
        gamma = gamma.tanh()
        beta = beta.tanh()

        q = self.query_norm(queries)
        delta = q * gamma + beta

        box_area = (boxes[..., 2] * boxes[..., 3]).clamp(0.0, 1.0)
        stats = torch.stack([conf_mean, conf_max, box_area], dim=-1)
        stats_delta = self.stats_proj(stats).tanh()

        delta = delta + stats_delta

        alpha = self.alpha_max * torch.sigmoid(self.alpha_logit)

        return queries + alpha * delta


class GridBoxRefiner(nn.Module):
    """Grid-conditioned bounded box refinement without changing decoder queries."""

    def __init__(self, hidden_dim=256, grid_dim=None, roi_size=5, alpha_max=0.05,
                 detach_grid=True, detach_boxes=True, min_box_size=1e-3,
                 use_conf_weight=True, conf_power=1.0, init_alpha_ratio=0.1,
                 conf_is_logits=True):
        super().__init__()
        self.grid_dim = hidden_dim if grid_dim is None else grid_dim
        self.roi_size = int(roi_size)
        self.alpha_max = float(alpha_max)
        self.detach_grid = detach_grid
        self.detach_boxes = detach_boxes
        self.min_box_size = float(min_box_size)
        self.use_conf_weight = use_conf_weight
        self.conf_power = float(conf_power)
        self.conf_is_logits = conf_is_logits

        self.grid_proj = nn.Conv2d(self.grid_dim, hidden_dim, 1, bias=False) if self.grid_dim != hidden_dim else nn.Identity()
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.stats_proj = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))
        self.delta_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(inplace=True), nn.Linear(hidden_dim, 4))
        init.constant_(self.delta_head[-1].weight, 0)
        init.constant_(self.delta_head[-1].bias, 0)

        init_alpha_ratio = min(max(init_alpha_ratio, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_alpha_ratio / (1.0 - init_alpha_ratio))
        self.alpha_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def _sample_roi(self, feat, boxes):
        B, C, H, W = feat.shape
        _, N, _ = boxes.shape
        K = self.roi_size
        boxes = boxes.clamp(0.0, 1.0)
        cx, cy, bw, bh = boxes.unbind(dim=-1)
        bw = bw.clamp(min=self.min_box_size)
        bh = bh.clamp(min=self.min_box_size)
        x1 = (cx - 0.5 * bw).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * bh).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * bw).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * bh).clamp(0.0, 1.0)
        lin = torch.linspace(0.5 / K, 1.0 - 0.5 / K, K, device=feat.device, dtype=feat.dtype)
        yy, xx = torch.meshgrid(lin, lin, indexing='ij')
        xs = x1[..., None, None] + xx.view(1, 1, K, K) * (x2 - x1)[..., None, None]
        ys = y1[..., None, None] + yy.view(1, 1, K, K) * (y2 - y1)[..., None, None]
        sample_grid = torch.stack([xs * 2.0 - 1.0, ys * 2.0 - 1.0], dim=-1)
        roi_feats = []
        for b in range(B):
            grid_b = sample_grid[b].reshape(1, N * K, K, 2)
            sampled = F.grid_sample(feat[b:b + 1], grid_b, mode='bilinear', padding_mode='zeros', align_corners=False)
            roi_feats.append(sampled.view(C, N, K, K).permute(1, 0, 2, 3).contiguous())
        return torch.stack(roi_feats, dim=0)

    def forward(self, boxes, grid_feat, grid_logits=None):
        if grid_feat is None or boxes is None:
            return boxes
        sample_boxes = boxes.detach() if self.detach_boxes else boxes
        if self.detach_grid:
            grid_feat = grid_feat.detach()
            if grid_logits is not None:
                grid_logits = grid_logits.detach()
        grid_feat = self.grid_proj(grid_feat)
        roi_feat = self._sample_roi(grid_feat, sample_boxes)

        roi_conf = None
        if self.use_conf_weight and grid_logits is not None:
            if isinstance(grid_logits, (list, tuple)):
                grid_logits = grid_logits[0]
            if isinstance(grid_logits, dict):
                grid_logits = grid_logits.get('grid_logits', grid_logits.get('pred_grid_logits', None))
            if grid_logits is not None:
                if grid_logits.shape[1] > 1:
                    grid_logits = grid_logits.max(dim=1, keepdim=True).values
                if grid_logits.shape[-2:] != grid_feat.shape[-2:]:
                    grid_logits = F.interpolate(grid_logits, size=grid_feat.shape[-2:], mode='bilinear', align_corners=False)
                grid_conf = grid_logits.sigmoid() if self.conf_is_logits else grid_logits
                roi_conf = self._sample_roi(grid_conf.clamp(0.0, 1.0), sample_boxes)

        if roi_conf is not None:
            weight = roi_conf.clamp(min=1e-4).pow(self.conf_power)
            context = (roi_feat * weight).sum(dim=(-1, -2)) / (weight.sum(dim=(-1, -2)) + 1e-6)
            conf_mean = roi_conf.mean(dim=(-1, -2)).squeeze(-1)
            conf_max = roi_conf.amax(dim=(-1, -2)).squeeze(-1)
        else:
            context = roi_feat.mean(dim=(-1, -2))
            conf_mean = torch.zeros(boxes.shape[:2], device=boxes.device, dtype=boxes.dtype)
            conf_max = torch.zeros_like(conf_mean)

        box_area = (sample_boxes[..., 2] * sample_boxes[..., 3]).clamp(0.0, 1.0)
        stats = torch.stack([conf_mean, conf_max, box_area], dim=-1)
        context = self.context_norm(context + self.stats_proj(stats))
        delta = self.delta_head(context).tanh()
        alpha = self.alpha_max * torch.sigmoid(self.alpha_logit)
        refined_xy = boxes[..., :2] + alpha * delta[..., :2]
        refined_wh = boxes[..., 2:] * torch.exp(alpha * delta[..., 2:])
        refined = torch.cat([refined_xy, refined_wh], dim=-1)
        return refined.clamp(0.0, 1.0)

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.,
        activation='relu',
        n_levels=4,
        n_points=4,
        cross_attn_method='default',
        layer_scale=None,
        use_gateway=False,
    ):
        super(TransformerDecoderLayer, self).__init__()

        if layer_scale is not None:
            print(f"     --- Wide Layer@{layer_scale} ---")
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)

        # ffn
        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        target,
        reference_points,
        value,
        spatial_shapes,
        attn_mask=None,
        query_pos_embed=None,
    ):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )

        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = target + self.dropout2(target2)
            target = self.norm2(target)

        # ffn
        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates
    across multiple layers, utilizing attention mechanisms, location quality
    estimators, and distribution refinement techniques to improve bounding box
    accuracy and robustness.
    """

    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
        act='relu',
    ):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max

        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [
                copy.deepcopy(decoder_layer_wide)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )

        self.lqe_layers = nn.ModuleList([
            copy.deepcopy(LQE(4, 64, 2, reg_max, act=act))
            for _ in range(num_layers)
        ])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value

        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)

        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(
            self.reg_max,
            self.up,
            self.reg_scale,
            deploy=True,
        )
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]]
        )

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
        grid_feat=None,
        grid_logits=None,
        grid_interaction=None,
        grid_box_refiner=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0

        # NOTE:
        # memory here can be the original encoder memory or the grid-enhanced
        # memory, depending on what DEIMTransformer.forward passes in.
        # In the proposed value-only enhancement design, this memory is the
        # enhanced memory and is only used by cross-attention.
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []

        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(
                    query_pos_embed,
                    scale_factor=self.layer_scale,
                )
                value = self.value_op(
                    memory,
                    None,
                    query_pos_embed.shape[-1],
                    memory_mask,
                    spatial_shapes,
                )
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(
                output,
                ref_points_input,
                value,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
            )

            if i == 0:
                # Initial bounding box predictions with inverse sigmoid refinement.
                pre_bboxes = F.sigmoid(
                    pre_bbox_head(output) + inverse_sigmoid(ref_points_detach)
                )
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections.
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(
                ref_points_initial,
                integral(pred_corners, project),
                reg_scale,
            )


            # Box-conditioned Grid Context Aggregation


            use_box_grid_this_layer = (
                grid_interaction is not None
                and grid_feat is not None
                and (i == self.eval_idx or i == self.num_layers - 1)
                )
            use_grid_box_refiner_this_layer = (
                grid_box_refiner is not None
                and grid_feat is not None
                and (i == self.eval_idx or i == self.num_layers - 1)
                )

            output_for_score = output

            if use_box_grid_this_layer:
                interacted_output = grid_interaction(
                    queries=output,
                    grid_feat=grid_feat,
                    boxes=inter_ref_bbox,
                    grid_logits=grid_logits,
                    )
                output_for_score = interacted_output

            if use_grid_box_refiner_this_layer:
                inter_ref_bbox = grid_box_refiner(
                    boxes=inter_ref_bbox,
                    grid_feat=grid_feat,
                    grid_logits=grid_logits,
                )

            if self.training or i == self.eval_idx:
                # scores = score_head[i](output)
                scores = score_head[i](output_for_score)

                # LQE does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)

                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


@register()
class DEIMTransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method='default',
        query_select_method='default',
        reg_max=32,
        reg_scale=4.,
        layer_scale=1,
        mlp_act='relu',
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        use_box_grid_interaction=False,
        box_grid_dim=None,
        box_grid_roi_size=3,
        box_grid_alpha_max=0.1,
        box_grid_detach_grid=True,
        box_grid_detach_boxes=True,
        box_grid_use_conf_weight=True,
        box_grid_conf_power=2.0,
        box_grid_init_alpha_ratio=0.1,
        use_grid_box_refiner=False,
    ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''

        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        self.use_box_grid_interaction = use_box_grid_interaction
        self.use_grid_box_refiner = use_grid_box_refiner

        if self.use_box_grid_interaction:
            self.box_grid_interaction = BoxConditionedGridContextInteraction(hidden_dim=hidden_dim,
                                                                             grid_dim=box_grid_dim if box_grid_dim is not None else hidden_dim,
                                                                             roi_size=box_grid_roi_size,
                                                                             alpha_max=box_grid_alpha_max,
                                                                             detach_grid=box_grid_detach_grid,
                                                                             detach_boxes=box_grid_detach_boxes,
                                                                             dropout=dropout,
                                                                             use_conf_weight=box_grid_use_conf_weight,
                                                                             conf_power=box_grid_conf_power,
                                                                             init_alpha_ratio=box_grid_init_alpha_ratio,
                                                                             conf_is_logits=True)
            print(f"     --- Use Box-Conditioned Grid Context Interaction: "
                  f"grid_dim={box_grid_dim if box_grid_dim is not None else hidden_dim}, "
                  f"roi_size={box_grid_roi_size}, "
                  f"alpha_max={box_grid_alpha_max}, "
                  f"detach_grid={box_grid_detach_grid}, "
                  f"detach_boxes={box_grid_detach_boxes},"
                  f"use_conf_weight={box_grid_use_conf_weight}, "
                  f"conf_power={box_grid_conf_power}, "
                  f"init_alpha_ratio={box_grid_init_alpha_ratio} ---")
        else:
            self.box_grid_interaction = None

        if self.use_grid_box_refiner:
            self.grid_box_refiner = GridBoxRefiner(hidden_dim=hidden_dim,
                                                   grid_dim=box_grid_dim if box_grid_dim is not None else hidden_dim,
                                                   roi_size=box_grid_roi_size,
                                                   alpha_max=box_grid_alpha_max,
                                                   detach_grid=box_grid_detach_grid,
                                                   detach_boxes=box_grid_detach_boxes,
                                                   use_conf_weight=box_grid_use_conf_weight,
                                                   conf_power=box_grid_conf_power,
                                                   init_alpha_ratio=box_grid_init_alpha_ratio,
                                                   conf_is_logits=True)
            print(f"     --- Use GridBoxRefiner: "
                  f"grid_dim={box_grid_dim if box_grid_dim is not None else hidden_dim}, "
                  f"roi_size={box_grid_roi_size}, "
                  f"alpha_max={box_grid_alpha_max}, "
                  f"detach_grid={box_grid_detach_grid}, "
                  f"detach_boxes={box_grid_detach_boxes},"
                  f"use_conf_weight={box_grid_use_conf_weight}, "
                  f"conf_power={box_grid_conf_power}, "
                  f"init_alpha_ratio={box_grid_init_alpha_ratio} ---")
        else:
            self.grid_box_refiner = None

        # print the parameters
        print(f"     --- Use Gateway@{use_gateway} ---")
        print(f"     --- Use Share Bbox Head@{share_bbox_head} ---")
        print(f"     --- Use Share Score Head@{share_score_head} ---")

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            use_gateway=use_gateway,
        )

        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
            use_gateway=use_gateway,
        )

        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=activation,
        )

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1,
                hidden_dim,
                padding_idx=num_classes,
            )
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=mlp_act)

        # decoder head
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.integral = Integral(self.reg_max)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

        dec_score_head = nn.Linear(hidden_dim, num_classes)
        self.dec_score_head = nn.ModuleList(
            [
                dec_score_head if share_score_head else copy.deepcopy(dec_score_head)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                copy.deepcopy(dec_score_head)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )

        # Share the same bbox head for all layers
        dec_bbox_head = MLP(
            hidden_dim,
            hidden_dim,
            4 * (self.reg_max + 1),
            3,
            act=mlp_act,
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(
                    scaled_dim,
                    scaled_dim,
                    4 * (self.reg_max + 1),
                    3,
                    act=mlp_act,
                )
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList([
            self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
            for i in range(len(self.dec_bbox_head))
        ])

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)

        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)

        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()

        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict([
                            ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                            ('norm', nn.BatchNorm2d(self.hidden_dim)),
                        ])
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict([
                            ('conv', nn.Conv2d(
                                in_channels,
                                self.hidden_dim,
                                3,
                                2,
                                padding=1,
                                bias=False,
                            )),
                            ('norm', nn.BatchNorm2d(self.hidden_dim)),
                        ])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(
        self,
        feats: List[torch.Tensor],
        store_proj_feats: bool = True,
    ):
        """Convert multi-scale feature maps into flattened transformer memory."""

        # get projection features
        proj_feats = [
            self.input_proj[i](feat)
            for i, feat in enumerate(feats)
        ]

        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)

            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # Store projected features for potential feedback to grid head.
        # Important:
        #   In grid-enhanced value-only mode, do NOT overwrite this with
        #   enhanced features.
        if store_proj_feats:
            self._proj_feats = proj_feats

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []

        for feat in proj_feats:
            _, _, h, w = feat.shape

            # [B, C, H, W] -> [B, H*W, C]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))

            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [B, L, C]
        feat_flatten = torch.concat(feat_flatten, 1)

        return feat_flatten, spatial_shapes

    def get_intermediate_features(self):
        """
        Get intermediate detection features for feedback to grid head.
        Returns the projected features from the last forward pass.
        """
        if hasattr(self, '_proj_feats'):
            # Return stride-16 feature, middle scale, good balance.
            return self._proj_feats[1] if len(self._proj_feats) > 1 else self._proj_feats[0]

        return None

    def _generate_anchors(
        self,
        spatial_shapes=None,
        grid_size=0.05,
        dtype=torch.float32,
        device='cpu',
    ):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size

            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []

        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h),
                torch.arange(w),
                indexing='ij',
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (
                grid_xy.unsqueeze(0) + 0.5
            ) / torch.tensor([w, h], dtype=dtype)

            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)

        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(
            -1,
            keepdim=True,
        )

        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        denoising_logits=None,
        denoising_bbox_unact=None,
    ):
        """Prepare decoder initial queries."""

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes,
                device=memory.device,
            )
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory

        enc_outputs_logits: torch.Tensor = self.enc_score_head(memory)

        # select top-k queries
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            memory,
            enc_outputs_logits,
            anchors,
            self.num_queries,
        )

        enc_topk_bbox_unact: torch.Tensor = (
            self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors
        )

        enc_topk_bboxes_list, enc_topk_logits_list = [], []

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, enc_topk_bbox_unact],
                dim=1,
            )
            content = torch.concat([denoising_logits, content], dim=1)

        return (
            content,
            enc_topk_bbox_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
        )

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_anchors_unact: torch.Tensor,
        topk: int,
    ):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(
                outputs_logits.max(-1).values,
                topk,
                dim=-1,
            )

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(
                outputs_logits.flatten(1),
                topk,
                dim=-1,
            )
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(
                outputs_logits.squeeze(-1),
                topk,
                dim=-1,
            )

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(
                1,
                1,
                outputs_anchors_unact.shape[-1],
            ),
        )

        topk_logits = (
            outputs_logits.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(
                    1,
                    1,
                    outputs_logits.shape[-1],
                ),
            )
            if self.training
            else None
        )

        topk_memory = memory.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(
                1,
                1,
                memory.shape[-1],
            ),
        )

        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None, value_feats=None, grid_feat=None, grid_logits=None):
        """Args:
            feats:
                Original encoder features."""


        memory_select, spatial_shapes = self._get_encoder_input(
            feats,
            store_proj_feats=True,
        )


        if value_feats is None:
            memory_value = memory_select
        else:
            memory_value, value_spatial_shapes = self._get_encoder_input(
                value_feats,
                store_proj_feats=False,
            )

            if value_spatial_shapes != spatial_shapes:
                raise ValueError(
                    f"value_feats spatial shapes {value_spatial_shapes} must match "
                    f"original feats spatial shapes {spatial_shapes}"
                )


        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                None,
                None,
                None,
                None,
            )


        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
        ) = self._get_decoder_input(
            memory_select,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact,
        )


        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = (
            self.decoder(
                init_ref_contents,
                init_ref_points_unact,
                memory_value,
                spatial_shapes,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                self.pre_bbox_head,
                self.integral,
                self.up,
                self.reg_scale,
                attn_mask=attn_mask,
                dn_meta=dn_meta,
                grid_feat=grid_feat,
                grid_logits=grid_logits,
                grid_interaction=self.box_grid_interaction if self.use_box_grid_interaction else None,
                grid_box_refiner=self.grid_box_refiner if self.use_grid_box_refiner else None
            )
        )

        if self.training and dn_meta is not None:
            # the output from the first decoder layer, only one
            dn_pre_logits, pre_logits = torch.split(
                pre_logits,
                dn_meta['dn_num_split'],
                dim=1,
            )
            dn_pre_bboxes, pre_bboxes = torch.split(
                pre_bboxes,
                dn_meta['dn_num_split'],
                dim=1,
            )

            dn_out_logits, out_logits = torch.split(
                out_logits,
                dn_meta['dn_num_split'],
                dim=2,
            )
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes,
                dn_meta['dn_num_split'],
                dim=2,
            )

            dn_out_corners, out_corners = torch.split(
                out_corners,
                dn_meta['dn_num_split'],
                dim=2,
            )
            dn_out_refs, out_refs = torch.split(
                out_refs,
                dn_meta['dn_num_split'],
                dim=2,
            )

        if self.training:
            out = {
                'pred_logits': out_logits[-1],
                'pred_boxes': out_bboxes[-1],
                'pred_corners': out_corners[-1],
                'ref_points': out_refs[-1],
                'up': self.up,
                'reg_scale': self.reg_scale,
            }
        else:
            out = {
                'pred_logits': out_logits[-1],
                'pred_boxes': out_bboxes[-1],
            }

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )

            # Important:
            # Encoder auxiliary outputs are still based on original memory_select.
            out['enc_aux_outputs'] = self._set_aux_loss(
                enc_topk_logits_list,
                enc_topk_bboxes_list,
            )

            out['pre_outputs'] = {
                'pred_logits': pre_logits,
                'pred_boxes': pre_bboxes,
            }

            out['enc_meta'] = {
                'class_agnostic': self.query_select_method == 'agnostic',
            }

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out['dn_pre_outputs'] = {
                    'pred_logits': dn_pre_logits,
                    'pred_boxes': dn_pre_bboxes,
                }
                out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values,
        # such as a dict having both a Tensor and a list.
        return [
            {
                'pred_logits': a,
                'pred_boxes': b,
            }
            for a, b in zip(outputs_class, outputs_coord)
        ]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values,
        # such as a dict having both a Tensor and a list.
        return [
            {
                'pred_logits': a,
                'pred_boxes': b,
                'pred_corners': c,
                'ref_points': d,
                'teacher_corners': teacher_corners,
                'teacher_logits': teacher_logits,
            }
            for a, b, c, d in zip(
                outputs_class,
                outputs_coord,
                outputs_corners,
                outputs_ref,
            )
        ]
