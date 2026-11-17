"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Frequency-Guided Decoupled Grid Head (FGDG)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from ..core import register


class HighFrequencyPath(nn.Module):
    """Extract local detail with a Laplacian filter and convolutions."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()


        self.edge_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self._init_laplacian_kernel()


        self.horizontal_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(1, 7), padding=(0, 3), bias=False)

        self.vertical_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(7, 1), padding=(3, 0), bias=False)


        self.detail_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )


        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def _init_laplacian_kernel(self):
        """Initialize the depthwise Laplacian kernel."""
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32) / 8.0

        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(self.edge_conv.weight.shape[0], 1, 3, 3).contiguous()

        with torch.no_grad():
            self.edge_conv.weight.copy_(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        edge = self.edge_conv(x)
        x_enhanced = x + 0.5 * edge


        h_feat = self.horizontal_conv(x_enhanced)
        v_feat = self.vertical_conv(x_enhanced)
        direction_feat = torch.cat([h_feat, v_feat], dim=1)


        detail_feat = self.detail_conv(x_enhanced)


        out = self.fusion(torch.cat([direction_feat, detail_feat], dim=1))

        return out


class LowFrequencyPath(nn.Module):
    """Aggregate smoothed regional features."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()


        self.smooth = nn.AvgPool2d(3, stride=1, padding=1)


        self.region_conv = nn.Sequential(

            nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )


        self.context_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )


        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_smooth = self.smooth(x)


        region_feat = self.region_conv(x_smooth)


        context_feat = self.context_conv(x)


        out = self.fusion(torch.cat([region_feat, context_feat], dim=1))

        return out


class AdaptiveFusion(nn.Module):
    """Fuse frequency paths with learned channel weights."""
    def __init__(self, channels: int, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes


        self.global_pool = nn.AdaptiveAvgPool2d(1)


        self.weight_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, num_classes, 1),
            nn.Sigmoid(),
        )


        self.freq_prior = nn.Parameter(torch.tensor([0.7, 0.7, 0.3, 0.3]))

    def forward(
        self,
        high_freq_feat: torch.Tensor,
        low_freq_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the module to the input tensors."""
        B, C, H, W = high_freq_feat.shape


        concat_feat = torch.cat([high_freq_feat, low_freq_feat], dim=1)


        global_feat = self.global_pool(concat_feat)
        class_weights = self.weight_predictor(global_feat)  # (B, num_classes, 1, 1)


        prior = self.freq_prior.view(1, -1, 1, 1).to(class_weights.device)
        class_weights = 0.5 * class_weights + 0.5 * prior


        avg_weight = class_weights.mean(dim=1, keepdim=True)  # (B, 1, 1, 1)

        fused_feat = avg_weight * high_freq_feat + (1 - avg_weight) * low_freq_feat

        return fused_feat, class_weights


class DecoupledGridBranch(nn.Module):
    """Refine grid features independently from the detection branch."""
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)


        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(inplace=True),
            )
            for in_ch in in_channels_list
        ])


        self.downsample_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(self.num_levels - 1)
        ])


        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(self.num_levels - 1)
        ])

    def forward(self, backbone_feats: List[torch.Tensor]) -> torch.Tensor:
        """Apply the module to the input tensors."""

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, backbone_feats)]


        # stride8 -> stride16 -> stride32
        for i in range(self.num_levels - 1):
            down = self.downsample_convs[i](laterals[i])
            laterals[i + 1] = self.fusion_convs[i](laterals[i + 1] + down)


        return laterals[-1]


@register()
class FrequencyGuidedGridHead(nn.Module):
    """Classify grid occupancy using separate detail and regional feature paths."""
    def __init__(
        self,
        in_channels: int = 224,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.1,
        prior_probs: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.prior_probs = prior_probs


        if isinstance(in_channels, int):
            in_channels_list = [in_channels] * 3
        else:
            in_channels_list = list(in_channels)


        self.grid_branch = DecoupledGridBranch(
            in_channels_list=in_channels_list,
            out_channels=hidden_dim,
        )


        self.high_freq_path = HighFrequencyPath(hidden_dim, hidden_dim)


        self.low_freq_path = LowFrequencyPath(hidden_dim, hidden_dim)


        self.adaptive_fusion = AdaptiveFusion(hidden_dim, num_classes)


        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, 1, bias=True)

        self._init_weights()

        print(f"[FrequencyGuidedGridHead] Decoupled branch + High/Low freq paths")
        print(f"  hidden_dim={hidden_dim}, num_classes={num_classes}")

    def _init_weights(self):
        """Initialize layer weights and optional prior-probability biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if self.prior_probs is not None and self.cls_head.bias is not None:
            prior_probs = torch.tensor(self.prior_probs, dtype=torch.float32)
            assert len(prior_probs) == self.num_classes

            eps = 1e-8
            prior_probs = torch.clamp(prior_probs, eps, 1 - eps)
            bias_init = -torch.log((1 - prior_probs) / prior_probs)

            with torch.no_grad():
                self.cls_head.bias.copy_(bias_init)
            print(f"  Bias initialized with prior_probs: {self.prior_probs}")

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        encoder_feats: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply the module to the input tensors."""

        grid_feat = self.grid_branch(backbone_feats)


        high_freq_feat = self.high_freq_path(grid_feat)


        low_freq_feat = self.low_freq_path(grid_feat)


        fused_feat, class_weights = self.adaptive_fusion(high_freq_feat, low_freq_feat)


        x = self.dropout(fused_feat)
        logits = self.cls_head(x)


        # logits = logits * class_weights + logits * (1 - class_weights)

        return logits

    def get_freq_weights(self, backbone_feats: List[torch.Tensor]) -> torch.Tensor:
        """Return the learned frequency weights."""
        grid_feat = self.grid_branch(backbone_feats)
        high_freq_feat = self.high_freq_path(grid_feat)
        low_freq_feat = self.low_freq_path(grid_feat)
        _, class_weights = self.adaptive_fusion(high_freq_feat, low_freq_feat)
        return class_weights
