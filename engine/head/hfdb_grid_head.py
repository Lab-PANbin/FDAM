"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

HFDB: High-Frequency Decoupled Branch for Grid Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from ..core import register


def get_num_groups(channels: int) -> int:
    """Dynamic GroupNorm group calculation."""
    if channels >= 256:
        return 32
    elif channels >= 128:
        return 16
    elif channels >= 64:
        return 8
    else:
        return max(1, channels // 8)


class HFEA(nn.Module):
    """High-Frequency Enhancement Attention (HFEA)"""
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        use_dct: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.use_dct = use_dct
        reduced_ch = max(channels // reduction, 16)

        # High-pass filter (learnable Laplacian-like kernel)
        self.high_pass = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self._init_high_pass_kernel()

        # Spatial attention from high-frequency response
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, reduced_ch, 1, bias=False),
            nn.GroupNorm(get_num_groups(reduced_ch), reduced_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # Channel attention from high-frequency statistics
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_ch, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, 1, bias=True),
            nn.Sigmoid()
        )

        # Optional: DCT-based frequency decomposition
        if use_dct:
            self.dct_weight = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)

        # Learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.5)

    def _init_high_pass_kernel(self):
        """Initialize high-pass filter as Laplacian kernel."""
        # Laplacian kernel: center=1, neighbors=-1/8
        kernel = torch.tensor([
            [-1/8, -1/8, -1/8],
            [-1/8,  1.0, -1/8],
            [-1/8, -1/8, -1/8]
        ], dtype=torch.float32)

        # Expand to all channels
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        kernel = kernel.expand(self.channels, 1, 3, 3).contiguous()

        with torch.no_grad():
            self.high_pass.weight.copy_(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            High-frequency enhanced features (B, C, H, W)
        """

        hf = self.high_pass(x)  # (B, C, H, W)


        if self.use_dct:
            hf = self._dct_enhance(hf)


        spatial_w = self.spatial_attn(hf)  # (B, 1, H, W)


        channel_w = self.channel_attn(torch.abs(hf))  # (B, C, 1, 1)


        alpha = torch.sigmoid(self.fusion_weight)
        enhanced = x * (1 + alpha * spatial_w * channel_w)

        return enhanced

    def _dct_enhance(self, hf: torch.Tensor) -> torch.Tensor:
        """
        DCT-based high-frequency enhancement.
        Approximates DCT filtering using spatial operations.
        """
        # Use absolute value to get frequency magnitude
        hf_mag = torch.abs(hf)

        # Learnable weighting
        hf_weighted = hf * torch.sigmoid(self.dct_weight * hf_mag)

        return hf_weighted

    def get_attention_maps(self, x: torch.Tensor) -> dict:
        """Get attention maps for visualization."""
        hf = self.high_pass(x)
        if self.use_dct:
            hf = self._dct_enhance(hf)

        spatial_w = self.spatial_attn(hf)
        channel_w = self.channel_attn(torch.abs(hf))

        return {
            'high_freq_response': hf,
            'spatial_attention': spatial_w,
            'channel_attention': channel_w,
        }


class CSA(nn.Module):
    """Cross-Scale Alignment (CSA)"""
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 7,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)

        # Relative position bias
        self.relative_position_bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size, window_size * window_size)
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

        # Layer norm
        self.norm = nn.GroupNorm(get_num_groups(channels), channels)

    def forward(
        self,
        low_feat: torch.Tensor,
        high_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            low_feat: Low-level features (high resolution) (B, C, H, W)
            high_feat: High-level features (upsampled to same size) (B, C, H, W)
        Returns:
            Aligned features (B, C, H, W)
        """
        B, C, H, W = low_feat.shape

        # Ensure high_feat matches low_feat size
        if high_feat.shape[2:] != low_feat.shape[2:]:
            high_feat = F.interpolate(
                high_feat, size=(H, W),
                mode='bilinear', align_corners=False
            )

        # Project to Q, K, V
        q = self.q_proj(low_feat)   # Query from low-level (detail)
        k = self.k_proj(high_feat)  # Key from high-level (semantic)
        v = self.v_proj(high_feat)  # Value from high-level

        # Window-based attention for efficiency
        out = self._window_attention(q, k, v, H, W)

        # Output projection and residual
        out = self.out_proj(out)
        out = self.norm(out + low_feat)

        return out

    def _window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """Efficient window-based cross-attention."""
        B, C, _, _ = q.shape
        ws = self.window_size

        # Pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w

        # Reshape to windows: (B, C, H, W) -> (B*nH*nW, heads, ws*ws, head_dim)
        nH, nW = Hp // ws, Wp // ws

        q = q.view(B, self.num_heads, self.head_dim, Hp, Wp)
        k = k.view(B, self.num_heads, self.head_dim, Hp, Wp)
        v = v.view(B, self.num_heads, self.head_dim, Hp, Wp)

        # Rearrange to windows
        q = q.view(B, self.num_heads, self.head_dim, nH, ws, nW, ws)
        q = q.permute(0, 3, 5, 1, 4, 6, 2).contiguous()  # (B, nH, nW, heads, ws, ws, head_dim)
        q = q.view(B * nH * nW, self.num_heads, ws * ws, self.head_dim)

        k = k.view(B, self.num_heads, self.head_dim, nH, ws, nW, ws)
        k = k.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
        k = k.view(B * nH * nW, self.num_heads, ws * ws, self.head_dim)

        v = v.view(B, self.num_heads, self.head_dim, nH, ws, nW, ws)
        v = v.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
        v = v.view(B * nH * nW, self.num_heads, ws * ws, self.head_dim)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*nH*nW, heads, ws*ws, ws*ws)
        attn = attn + self.relative_position_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B*nH*nW, heads, ws*ws, head_dim)

        # Reshape back
        out = out.view(B, nH, nW, self.num_heads, ws, ws, self.head_dim)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).contiguous()  # (B, heads, head_dim, nH, ws, nW, ws)
        out = out.view(B, C, Hp, Wp)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


class D2GAttention(nn.Module):
    """Detection-to-Grid Attention (D2G)"""
    def __init__(
        self,
        det_channels: int,
        grid_channels: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = grid_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Grid as Query
        self.q_proj = nn.Conv2d(grid_channels, grid_channels, 1, bias=False)

        # Detection as Key/Value
        self.k_proj = nn.Conv2d(det_channels, grid_channels, 1, bias=False)
        self.v_proj = nn.Conv2d(det_channels, grid_channels, 1, bias=False)

        # Output
        self.out_proj = nn.Sequential(
            nn.Conv2d(grid_channels, grid_channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(grid_channels), grid_channels),
        )

        # Learnable gate for controlling semantic injection
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(grid_channels * 2, grid_channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(grid_channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        grid_feat: torch.Tensor,
        det_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            grid_feat: Grid branch features (B, C_grid, H, W)
            det_feat: Detection branch features (B, C_det, H', W')
        Returns:
            Semantically enhanced grid features (B, C_grid, H, W)
        """
        B, C, H, W = grid_feat.shape

        # Align detection features to grid feature size
        if det_feat.shape[2:] != grid_feat.shape[2:]:
            det_feat = F.interpolate(
                det_feat, size=(H, W),
                mode='bilinear', align_corners=False
            )

        # Project
        q = self.q_proj(grid_feat)  # (B, C, H, W)
        k = self.k_proj(det_feat)   # (B, C, H, W)
        v = self.v_proj(det_feat)   # (B, C, H, W)

        # Reshape for attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.out_proj(out)

        # Gated fusion
        gate = self.gate(torch.cat([grid_feat, out], dim=1))
        out = grid_feat + gate * out

        return out


class HFDBranch(nn.Module):
    """High-Frequency Decoupled Branch (HFDB)"""
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 128,
        use_hfea: bool = True,
        use_csa: bool = True,
        use_d2g: bool = True,
        det_channels: int = 256,
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.use_hfea = use_hfea
        self.use_csa = use_csa
        self.use_d2g = use_d2g

        # Lateral connections: 1x1 conv to unify channels
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.GroupNorm(get_num_groups(out_channels), out_channels),
                nn.SiLU(inplace=True)
            )
            for in_ch in in_channels_list
        ])

        # HFEA modules for each level
        if use_hfea:
            self.hfea_modules = nn.ModuleList([
                HFEA(out_channels, reduction=4, use_dct=True)
                for _ in range(self.num_levels)
            ])

        # Bottom-up pathway with CSA
        # stride8 -> stride16 -> stride32
        if use_csa:
            self.csa_modules = nn.ModuleList([
                CSA(out_channels, num_heads=4, window_size=7)
                for _ in range(self.num_levels - 1)
            ])

        # Downsample convs for bottom-up
        self.downsample_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(get_num_groups(out_channels), out_channels),
                nn.SiLU(inplace=True)
            )
            for _ in range(self.num_levels - 1)
        ])

        # D2G attention
        if use_d2g:
            self.d2g_attn = D2GAttention(
                det_channels=det_channels,
                grid_channels=out_channels,
                num_heads=4
            )

        # Final enhancement
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True)
        )

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

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        det_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            backbone_feats: List of backbone features [stride8, stride16, stride32]
            det_feat: Optional detection features for D2G attention
        Returns:
            Grid features (B, out_channels, H, W) at stride-32 resolution
        """

        laterals = []
        for i, (feat, lateral_conv) in enumerate(zip(backbone_feats, self.lateral_convs)):
            lat = lateral_conv(feat)
            if self.use_hfea:
                lat = self.hfea_modules[i](lat)
            laterals.append(lat)


        # Process from high-res to low-res: stride8 -> stride16 -> stride32
        for i in range(self.num_levels - 1):
            # Downsample current level
            down = self.downsample_convs[i](laterals[i])

            # Cross-scale alignment
            if self.use_csa:
                # CSA: align downsampled features with next level
                laterals[i + 1] = self.csa_modules[i](laterals[i + 1], down)
            else:
                laterals[i + 1] = laterals[i + 1] + down


        out = laterals[-1]


        if self.use_d2g and det_feat is not None:
            out = self.d2g_attn(out, det_feat)


        out = self.output_conv(out)

        return out

    def get_multi_scale_outputs(
        self,
        backbone_feats: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Get outputs at all scales (for auxiliary supervision if needed)."""
        laterals = []
        for i, (feat, lateral_conv) in enumerate(zip(backbone_feats, self.lateral_convs)):
            lat = lateral_conv(feat)
            if self.use_hfea:
                lat = self.hfea_modules[i](lat)
            laterals.append(lat)

        return laterals


@register()
class HFDBGridHead(nn.Module):
    """HFDB Grid Classification Head"""
    def __init__(
        self,
        in_channels: int = 224,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.1,
        use_hfea: bool = True,
        use_csa: bool = True,
        use_d2g: bool = True,
        use_direction: bool = True,
        det_channels: int = 256,
        prior_probs: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_d2g = use_d2g
        self.use_direction = use_direction
        self.prior_probs = prior_probs

        # Handle input channels
        if isinstance(in_channels, int):
            in_channels_list = [in_channels] * 3
        else:
            in_channels_list = list(in_channels)

        # HFDB Branch
        self.hfdb = HFDBranch(
            in_channels_list=in_channels_list,
            out_channels=hidden_dim,
            use_hfea=use_hfea,
            use_csa=use_csa,
            use_d2g=use_d2g,
            det_channels=det_channels,
        )

        # Optional direction-aware processing
        if use_direction:
            from .direction_aware_grid_head import AsymmetricStripConv
            self.direction_conv = AsymmetricStripConv(
                hidden_dim, hidden_dim, kernel_lengths=[3, 7, 11]
            )

        # Classification head
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, 1, bias=True)

        self._init_weights()

        # Print configuration
        print(f"[HFDBGridHead] HFEA={use_hfea}, CSA={use_csa}, D2G={use_d2g}, Direction={use_direction}")

    def _init_weights(self):
        """Initialize classification head with prior probabilities."""
        if hasattr(self.cls_head, 'bias') and self.cls_head.bias is not None:
            if self.prior_probs is not None:
                prior_probs = torch.tensor(self.prior_probs, dtype=torch.float32)
                assert len(prior_probs) == self.num_classes

                eps = 1e-8
                prior_probs = torch.clamp(prior_probs, eps, 1 - eps)
                bias_init = -torch.log((1 - prior_probs) / prior_probs)

                with torch.no_grad():
                    self.cls_head.bias.copy_(bias_init)
                print(f"[HFDBGridHead] Bias initialized with prior_probs: {self.prior_probs}")
            else:
                default_prior = 0.01
                bias_init = -torch.log(torch.tensor((1 - default_prior) / default_prior))
                nn.init.constant_(self.cls_head.bias, bias_init)

    def forward(
        self,
        features: List[torch.Tensor],
        det_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: List of backbone/encoder features [stride8, stride16, stride32]
            det_feat: Optional detection features for D2G attention
        Returns:
            Grid classification logits (B, num_classes, H, W)
        """

        x = self.hfdb(features, det_feat)


        if self.use_direction:
            x = self.direction_conv(x)


        x = self.dropout(x)
        logits = self.cls_head(x)

        return logits

    def get_attention_maps(self, features: List[torch.Tensor]) -> dict:
        """Get attention maps for visualization."""
        attention_maps = {}

        # Get HFEA attention maps from each level
        if self.hfdb.use_hfea:
            for i, (feat, lateral_conv, hfea) in enumerate(zip(
                features,
                self.hfdb.lateral_convs,
                self.hfdb.hfea_modules
            )):
                lat = lateral_conv(feat)
                maps = hfea.get_attention_maps(lat)
                attention_maps[f'level_{i}'] = maps

        return attention_maps
