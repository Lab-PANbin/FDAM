"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Direction-Aware Grid Classification Head for Road Distress Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register


def get_num_groups(channels):
    """Dynamic GroupNorm group calculation based on channel count."""
    if channels >= 256:
        return 32
    elif channels >= 128:
        return 16
    elif channels >= 64:
        return 8
    else:
        return max(1, channels // 8)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (from CBAM)
    Emphasizes important spatial locations using max+avg pooling.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.conv(spatial)
        return x * self.sigmoid(spatial)


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion module.
    Fuses features from different scales (stride 8, 16, 32) and outputs fixed 32x32 resolution.

    Args:
        in_channels_list: List of input channels [ch_stride8, ch_stride16, ch_stride32]
        out_channels: Output channel dimension
        target_size: Target spatial size (default: 32 for 32x32 grid)
    """
    def __init__(self, in_channels_list, out_channels, target_size=32):
        super().__init__()
        self.num_scales = len(in_channels_list)
        self.target_size = target_size

        # 1x1 convs to unify channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in in_channels_list
        ])

        # Fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * self.num_scales, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, features):
        """Apply the module to the input tensors."""

        # target_size = features[0].shape[2:]
        target_size = features[-1].shape[2:]

        multi_scale_feats = []
        for feat, lateral_conv in zip(features, self.lateral_convs):
            # Project to unified channels
            feat = lateral_conv(feat)

            # Resize to target size
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)

            multi_scale_feats.append(feat)

        # Concatenate and fuse
        fused = torch.cat(multi_scale_feats, dim=1)
        fused = self.fusion_conv(fused)

        return fused


class AsymmetricStripConv(nn.Module):
    """Asymmetric Strip Convolution (ASC) Module"""
    def __init__(self, in_channels, out_channels, kernel_lengths=[3, 7, 11]):
        super().__init__()
        self.num_scales = len(kernel_lengths)
        num_branches = 2 * self.num_scales  # horizontal + vertical


        base_ch = out_channels // num_branches
        remainder = out_channels % num_branches


        h_channels = []
        v_channels = []
        for i in range(self.num_scales):
            # horizontal branches
            h_ch = base_ch + (1 if i < remainder else 0)
            h_channels.append(h_ch)
            # vertical branches
            v_idx = self.num_scales + i
            v_ch = base_ch + (1 if v_idx < remainder else 0)
            v_channels.append(v_ch)

        # Multi-scale horizontal strips (1×k) - for transverse crack detection
        self.h_convs = nn.ModuleList([
            nn.Conv2d(in_channels, h_channels[i],
                     kernel_size=(1, k), padding=(0, k//2), bias=False)
            for i, k in enumerate(kernel_lengths)
        ])

        # Multi-scale vertical strips (k×1) - for longitudinal crack detection
        self.v_convs = nn.ModuleList([
            nn.Conv2d(in_channels, v_channels[i],
                     kernel_size=(k, 1), padding=(k//2, 0), bias=False)
            for i, k in enumerate(kernel_lengths)
        ])

        self.norm = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # Extract multi-scale horizontal features
        h_feats = [conv(x) for conv in self.h_convs]
        # Extract multi-scale vertical features
        v_feats = [conv(x) for conv in self.v_convs]

        # Concatenate all directional features
        out = torch.cat(h_feats + v_feats, dim=1)
        return self.act(self.norm(out))


class DirectionSensitiveDecoupling(nn.Module):
    """Direction-Sensitive Feature Decoupling (DSFD) Module"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_ch = max(channels // reduction, 32)

        # Horizontal pathway - processes row-wise patterns
        self.h_pathway = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(channels, reduced_ch, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, 1, bias=False),
        )

        # Vertical pathway - processes column-wise patterns
        self.v_pathway = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(channels, reduced_ch, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, 1, bias=False),
        )

        # Direction confidence predictor
        self.direction_gate = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(channels, 2, 1, bias=True),  # 2 = horizontal + vertical
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Compute directional attention maps
        h_attn = self.h_pathway(x.mean(dim=2, keepdim=True))  # (B, C, 1, W)
        v_attn = self.v_pathway(x.mean(dim=3, keepdim=True))  # (B, C, H, 1)

        # Expand to full spatial size
        h_attn = h_attn.expand(-1, -1, H, -1)  # (B, C, H, W)
        v_attn = v_attn.expand(-1, -1, -1, W)  # (B, C, H, W)

        # Predict direction confidence (which direction is more important)
        gate = self.direction_gate(x.mean(dim=(2, 3), keepdim=True))  # (B, 2, 1, 1)
        h_weight = gate[:, 0:1, :, :]  # (B, 1, 1, 1)
        v_weight = gate[:, 1:2, :, :]  # (B, 1, 1, 1)

        # Weighted combination
        out = x * (h_weight * torch.sigmoid(h_attn) + v_weight * torch.sigmoid(v_attn))

        return out


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution

    Reduces parameters from C_in × C_out × K × K to C_in × K × K + C_in × C_out
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AdaptiveDirectionFusion(nn.Module):
    """Adaptive Direction Fusion (ADF) Module"""
    def __init__(self, channels):
        super().__init__()

        # Spatial direction predictor with Depthwise Separable Conv
        self.spatial_gate = nn.Sequential(
            DepthwiseSeparableConv(channels, channels // 4, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1, bias=True),  # 2 directions
        )

        # Channel recalibration (already lightweight with 1x1 convs)
        self.channel_recalib = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Predict per-pixel direction importance
        spatial_weights = torch.softmax(self.spatial_gate(x), dim=1)  # (B, 2, H, W)

        # Split features into two halves (assumed: first half=horizontal, second=vertical)
        C = x.shape[1]
        x_h, x_v = x[:, :C//2], x[:, C//2:]

        # Apply spatial direction weighting
        w_h = spatial_weights[:, 0:1]  # (B, 1, H, W)
        w_v = spatial_weights[:, 1:2]  # (B, 1, H, W)

        x_fused = torch.cat([x_h * w_h, x_v * w_v], dim=1)

        # Channel recalibration
        x_fused = x_fused * self.channel_recalib(x_fused.mean(dim=(2, 3), keepdim=True))

        return x_fused


class DirectionAwareBlock(nn.Module):
    """
    Direction-Aware Processing Block

    Combines ASC, DSFD, optional Spatial Attention, and residual connection.

    Args:
        channels: Number of channels
        use_dsfd: Whether to use Direction-Sensitive Feature Decoupling
        use_spatial_attention: Whether to use CBAM-style spatial attention
    """
    def __init__(self, channels, use_dsfd=True, use_spatial_attention=False):
        super().__init__()
        self.asc = AsymmetricStripConv(channels, channels, kernel_lengths=[3, 7, 11])
        self.dsfd = DirectionSensitiveDecoupling(channels) if use_dsfd else nn.Identity()
        self.spatial_attn = SpatialAttention(kernel_size=7) if use_spatial_attention else nn.Identity()

        # Residual projection
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(channels), channels)
        )

    def forward(self, x):
        identity = x
        out = self.asc(x)
        out = self.dsfd(out)
        out = self.spatial_attn(out)
        out = self.residual(out)
        return out + identity


@register()
class DirectionAwareGridHead(nn.Module):
    """Direction-Aware Grid Classification Head (DAG-Head)"""
    def __init__(self, in_channels=224, hidden_dim=128, num_classes=4,
                 num_blocks=2, dropout=0.1, feature_idx=2, use_adf=True,
                 use_multi_scale=False, use_spatial_attention=False,
                 prior_probs=None, grid_size=32):
        super().__init__()
        self.feature_idx = feature_idx
        self.num_classes = num_classes
        self.use_adf = use_adf
        self.use_multi_scale = use_multi_scale
        self.use_spatial_attention = use_spatial_attention
        self.prior_probs = prior_probs
        self.grid_size = grid_size

        # Multi-scale fusion (optional)
        if use_multi_scale:
            if isinstance(in_channels, int):
                in_channels_list = [in_channels] * 3
            else:
                in_channels_list = in_channels

            self.multi_scale_fusion = MultiScaleFusion(
                in_channels_list, hidden_dim, target_size=grid_size
            )
            self.input_proj = nn.Identity()  # Fusion already projects to hidden_dim
        else:
            self.multi_scale_fusion = None
            proj_in_channels = in_channels if isinstance(in_channels, int) else in_channels[feature_idx]
            self.input_proj = nn.Sequential(
                nn.Conv2d(proj_in_channels, hidden_dim, 1, bias=False),
                nn.GroupNorm(get_num_groups(hidden_dim), hidden_dim),
                nn.SiLU(inplace=True)
            )

        # Stack of Direction-Aware Blocks
        self.blocks = nn.Sequential(*[
            DirectionAwareBlock(hidden_dim, use_dsfd=True, use_spatial_attention=use_spatial_attention)
            for _ in range(num_blocks)
        ])

        # Adaptive Direction Fusion
        self.adf = AdaptiveDirectionFusion(hidden_dim) if use_adf else nn.Identity()

        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Final classification head
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with careful initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize classification head bias
        if hasattr(self.cls_head, 'bias') and self.cls_head.bias is not None:
            if self.prior_probs is not None:
                # Per-class prior probability initialization
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

                print(f"[DirectionAwareGridHead] Initialized bias with prior probabilities: {prior_probs.tolist()}")
                print(f"[DirectionAwareGridHead] Bias values: {bias_init.tolist()}")
            else:
                # Default prior probability
                default_prior = 0.01
                bias_init = -torch.log(torch.tensor((1 - default_prior) / default_prior))
                nn.init.constant_(self.cls_head.bias, bias_init)
                print(f"[DirectionAwareGridHead] No prior_probs provided, using default bias={bias_init:.2f}")

    def forward(self, features):
        """Apply the module to the input tensors."""
        # Multi-scale fusion or single feature selection
        if self.use_multi_scale and self.multi_scale_fusion is not None:
            x = self.multi_scale_fusion(features)
        else:
            x = features[self.feature_idx]
            x = self.input_proj(x)


        # Apply direction-aware processing blocks
        x = self.blocks(x)

        # Adaptive direction fusion
        x = self.adf(x)

        # Dropout and classification
        x = self.dropout(x)
        logits = self.cls_head(x)

        return logits

    def get_direction_attention(self, features):
        """
        Utility function to visualize direction attention maps.
        Useful for interpretability and debugging.

        Returns:
            h_attention: Horizontal attention map
            v_attention: Vertical attention map
        """
        x = features[self.feature_idx]
        x = self.input_proj(x)

        # Get attention from first block's DSFD
        if hasattr(self.blocks[0], 'dsfd') and hasattr(self.blocks[0].dsfd, 'direction_gate'):
            gate = self.blocks[0].dsfd.direction_gate(x)
            return gate[:, 0], gate[:, 1]  # h_weight, v_weight

        return None, None
