"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Implementation utilities.
"""

import torch
import torch.nn as nn
from ..core import register

class SEGate(nn.Module):
    """
    Squeeze-and-Excitation Gating Module (Channel Attention)
    Acts as a gate to emphasize important channels (cracks) and suppress noise (background texture).
    """
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Emphasizes important spatial locations (where cracks are likely to be).
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.conv(spatial)
        attention = self.sigmoid(spatial)

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Combines channel attention (SE) and spatial attention.
    """
    def __init__(self, channel, reduction=4, kernel_size=7):
        super().__init__()
        self.channel_attention = SEGate(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class GatedBottleneck(nn.Module):
    """
    Architecture:
    Input -> 1x1 Expand -> 3x3 DW-Conv -> GN -> ReLU -> Attention(CBAM) -> 1x1 Project -> GN -> Output
    """
    def __init__(self, in_ch, out_ch, expansion=2, stride=1, use_spatial_attention=False):
        super().__init__()
        hidden_dim = int(in_ch * expansion)
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_ch == out_ch

        def get_num_groups(channels):
            if channels >= 256:
                return 32
            elif channels >= 128:
                return 16
            elif channels >= 64:
                return 8
            else:
                return max(1, channels // 8)

        layers = []

        # Expand Phase (1x1 Conv)
        if expansion != 1:
            layers.append(nn.Conv2d(in_ch, hidden_dim, 1, bias=False))
            layers.append(nn.GroupNorm(get_num_groups(hidden_dim), hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Depth-wise Conv (3x3)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.GroupNorm(get_num_groups(hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
        ])

        # Attention Mechanism (SE or CBAM)
        if use_spatial_attention:
            layers.append(CBAM(hidden_dim))  # Channel + Spatial Attention
        else:
            layers.append(SEGate(hidden_dim))  # Channel Attention only

        # Project Phase (1x1 Conv) - Linear Bottleneck
        layers.extend([
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.GroupNorm(get_num_groups(out_ch), out_ch),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion module.
    Fuses features from different scales (stride 8, 16, 32) to capture both fine and coarse details.
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.num_scales = len(in_channels_list)

        # 1x1 convs to unify channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in in_channels_list
        ])

        # Fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * self.num_scales, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32 if out_channels >= 256 else 16, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        """
        Args:
            features: List of feature maps [feat_stride8, feat_stride16, feat_stride32]
        Returns:
            Fused feature map at stride 32 resolution
        """
        target_size = features[2].shape[2:]

        # Process each scale
        multi_scale_feats = []
        for i, (feat, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            # Project to unified channels
            feat = lateral_conv(feat)

            # Upsample to target size if needed
            if feat.shape[2:] != target_size:
                feat = nn.functional.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )

            multi_scale_feats.append(feat)

        # Concatenate and fuse
        fused = torch.cat(multi_scale_feats, dim=1)
        fused = self.fusion_conv(fused)

        return fused


@register()
class GatedGridHead(nn.Module):
    """A lightweight, efficient grid classification head using Gated Bottlenecks."""
    def __init__(self, in_channels=256, hidden_dim=384, num_classes=4, num_layers=3,
                 dropout=0.0, prior_probs=None, use_multi_scale=False, use_spatial_attention=False):
        super().__init__()
        self.dropout_rate = dropout
        self.num_classes = num_classes
        self.prior_probs = prior_probs
        self.use_multi_scale = use_multi_scale
        self.use_spatial_attention = use_spatial_attention

        def get_num_groups(channels):
            if channels >= 256:
                return 32
            elif channels >= 128:
                return 16
            elif channels >= 64:
                return 8
            else:
                return max(1, channels // 8)

        # Multi-scale fusion (optional)
        if use_multi_scale:
            # Expect in_channels to be a list [ch_stride8, ch_stride16, ch_stride32]
            if isinstance(in_channels, int):
                # If single int provided, assume all scales have same channels
                in_channels_list = [in_channels] * 3
            else:
                in_channels_list = in_channels

            self.multi_scale_fusion = MultiScaleFusion(in_channels_list, hidden_dim)
            proj_in_channels = hidden_dim
        else:
            self.multi_scale_fusion = None
            proj_in_channels = in_channels if isinstance(in_channels, int) else in_channels[0]

        # Initial Projection to hidden_dim
        if not use_multi_scale:
            self.input_proj = nn.Sequential(
                nn.Conv2d(proj_in_channels, hidden_dim, 1, bias=False),
                nn.GroupNorm(get_num_groups(hidden_dim), hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.input_proj = nn.Identity()  # Fusion already projects to hidden_dim

        # Stack Gated Bottlenecks
        blocks = []
        for _ in range(num_layers):
            # expansion=2 keeps it lightweight while allowing feature mixing
            blocks.append(GatedBottleneck(
                hidden_dim, hidden_dim, expansion=2,
                use_spatial_attention=use_spatial_attention
            ))
        self.blocks = nn.Sequential(*blocks)

        # Optional Dropout before classification
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Final Classification Layer
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if hasattr(self.cls_head, 'bias') and self.cls_head.bias is not None:
            if self.prior_probs is not None:
                if isinstance(self.prior_probs, (list, tuple)):
                    prior_probs = torch.tensor(self.prior_probs, dtype=torch.float32)
                else:
                    prior_probs = self.prior_probs

                assert len(prior_probs) == self.num_classes, \
                    f"prior_probs length ({len(prior_probs)}) must match num_classes ({self.num_classes})"

                eps = 1e-8
                prior_probs = torch.clamp(prior_probs, eps, 1 - eps)
                bias_init = -torch.log((1 - prior_probs) / prior_probs)

                with torch.no_grad():
                    self.cls_head.bias.copy_(bias_init)

                print(f"[GatedGridHead] Initialized bias with prior probabilities: {prior_probs.tolist()}")
                print(f"[GatedGridHead] Bias values: {bias_init.tolist()}")
            else:
                default_prior = 0.01
                bias_init = -torch.log(torch.tensor((1 - default_prior) / default_prior))
                nn.init.constant_(self.cls_head.bias, bias_init)
                print(f"[GatedGridHead] No prior_probs provided, using default bias={bias_init:.2f}")

    def forward(self, features):
        """
        Args:
            features (list[torch.Tensor]): Feature maps from encoder.
                - features[0]: stride=8 (highest resolution)
                - features[1]: stride=16
                - features[2]: stride=32

        Returns:
            logits (torch.Tensor): Classification logits of shape (B, num_classes, H, W)
        """

        if self.use_multi_scale and self.multi_scale_fusion is not None:
            x = self.multi_scale_fusion(features)
        else:
            x = features[2]
            x = self.input_proj(x)

        x = self.blocks(x)
        x = self.dropout(x)
        logits = self.cls_head(x)
        return logits
