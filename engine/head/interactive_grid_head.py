"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Interactive Grid Classification Head with Detection-Grid Bidirectional Interaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register

from .direction_aware_grid_head import (
    get_num_groups,
    MultiScaleFusion,
    DirectionAwareBlock,
    AdaptiveDirectionFusion,
)


class GridToDetectionBridge(nn.Module):
    """
    Converts grid classification predictions to spatial attention bias for detection.
    High grid confidence regions receive more attention in detection.
    """
    def __init__(self, num_grid_classes=4, hidden_dim=256, temperature=1.0):
        super().__init__()
        self.temperature = temperature

        self.grid_to_attn = nn.Sequential(
            nn.Conv2d(num_grid_classes, hidden_dim // 4, 1, bias=False),
            nn.GroupNorm(get_num_groups(hidden_dim // 4), hidden_dim // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, 1, bias=True),
        )

        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, grid_logits, target_size=None):
        grid_probs = torch.sigmoid(grid_logits)
        attn_bias = self.grid_to_attn(grid_probs)

        if target_size is not None and attn_bias.shape[2:] != target_size:
            attn_bias = F.interpolate(attn_bias, size=target_size, mode='bilinear', align_corners=False)

        attn_bias = self.scale * torch.sigmoid(attn_bias / self.temperature)
        return attn_bias


class DetectionToGridBridge(nn.Module):
    """
    Uses detection features to enhance grid classification via cross-attention.
    """
    def __init__(self, det_channels=256, grid_channels=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = grid_channels // num_heads

        self.det_proj = nn.Sequential(
            nn.Conv2d(det_channels, grid_channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(grid_channels), grid_channels),
            nn.SiLU(inplace=True),
        )

        self.q_proj = nn.Conv2d(grid_channels, grid_channels, 1, bias=False)
        self.k_proj = nn.Conv2d(grid_channels, grid_channels, 1, bias=False)
        self.v_proj = nn.Conv2d(grid_channels, grid_channels, 1, bias=False)

        self.out_proj = nn.Sequential(
            nn.Conv2d(grid_channels, grid_channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(grid_channels), grid_channels),
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(grid_channels * 2, grid_channels // 4, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(grid_channels // 4, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, grid_feat, det_feat):
        B, C, H, W = grid_feat.shape

        det_proj = self.det_proj(det_feat)
        if det_proj.shape[2:] != (H, W):
            det_proj = F.interpolate(det_proj, size=(H, W), mode='bilinear', align_corners=False)

        q = self.q_proj(grid_feat)
        k = self.k_proj(det_proj)
        v = self.v_proj(det_proj)

        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.out_proj(out)

        gate_input = torch.cat([grid_feat, out], dim=1)
        gate_weight = self.gate(gate_input)

        return grid_feat + gate_weight * out


@register()
class InteractiveGridHead(nn.Module):
    """Interactive Grid Classification Head with Multi-Scale Fusion and Bidirectional Interaction."""
    def __init__(
        self,
        in_channels=[224, 224, 224],
        hidden_dim=128,
        num_classes=4,
        det_channels=256,
        num_blocks=2,
        dropout=0.1,
        use_adf=True,
        use_spatial_attention=False,
        use_interaction=True,
        prior_probs=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_adf = use_adf
        self.use_spatial_attention = use_spatial_attention
        self.use_interaction = use_interaction
        self.prior_probs = prior_probs
        self.hidden_dim = hidden_dim

        # Multi-scale fusion
        if isinstance(in_channels, int):
            in_channels_list = [in_channels] * 3
        else:
            in_channels_list = in_channels

        self.multi_scale_fusion = MultiScaleFusion(
            in_channels_list, hidden_dim, target_size=32
        )

        # Direction-Aware Blocks
        self.blocks = nn.Sequential(*[
            DirectionAwareBlock(hidden_dim, use_dsfd=True, use_spatial_attention=use_spatial_attention)
            for _ in range(num_blocks)
        ])

        # Bidirectional interaction bridges
        if use_interaction:
            self.det_to_grid = DetectionToGridBridge(
                det_channels=det_channels,
                grid_channels=hidden_dim,
                num_heads=4
            )
            self.grid_to_det = GridToDetectionBridge(
                num_grid_classes=num_classes,
                hidden_dim=det_channels,
                temperature=1.0
            )

        # Adaptive Direction Fusion
        self.adf = AdaptiveDirectionFusion(hidden_dim) if use_adf else nn.Identity()

        # Dropout and classification
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Prior probability initialization for classification head
        if hasattr(self.cls_head, 'bias') and self.cls_head.bias is not None:
            if self.prior_probs is not None:
                prior_probs = torch.tensor(self.prior_probs, dtype=torch.float32)
                assert len(prior_probs) == self.num_classes

                eps = 1e-8
                prior_probs = torch.clamp(prior_probs, eps, 1 - eps)
                bias_init = -torch.log((1 - prior_probs) / prior_probs)

                with torch.no_grad():
                    self.cls_head.bias.copy_(bias_init)
                print(f"[InteractiveGridHead] Bias initialized with prior_probs: {self.prior_probs}")
            else:
                default_prior = 0.01
                bias_init = -torch.log(torch.tensor((1 - default_prior) / default_prior))
                nn.init.constant_(self.cls_head.bias, bias_init)

    def forward(self, features, det_features=None):
        """
        Args:
            features: List of encoder features [stride-8, stride-16, stride-32]
            det_features: Optional detection features for interaction (B, C, H, W)

        Returns:
            dict: {grid_logits, grid_features, grid_attn_bias} if use_interaction
            tensor: grid_logits if not use_interaction
        """
        # Multi-scale fusion
        x = self.multi_scale_fusion(features)

        # Direction-aware processing
        x = self.blocks(x)

        # Detection → Grid interaction
        if self.use_interaction and det_features is not None:
            x = self.det_to_grid(x, det_features)

        grid_features = x

        # ADF + classification
        x = self.adf(x)
        x = self.dropout(x)
        grid_logits = self.cls_head(x)

        if self.use_interaction:
            # Grid → Detection: generate attention bias
            target_size = features[1].shape[2:]  # stride-16 size
            attn_bias = self.grid_to_det(grid_logits, target_size=target_size)

            return {
                'grid_logits': grid_logits,
                'grid_features': grid_features,
                'grid_attn_bias': attn_bias,
            }
        else:
            return grid_logits

    def get_direction_attention(self, features):
        """Visualize direction attention maps."""
        x = self.multi_scale_fusion(features)
        if hasattr(self.blocks[0], 'dsfd') and hasattr(self.blocks[0].dsfd, 'direction_gate'):
            gate = self.blocks[0].dsfd.direction_gate(x)
            return gate[:, 0], gate[:, 1]
        return None, None
