"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

HS-FPN Inspired Grid Classification Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
from ..core import register


def get_num_groups(channels: int, min_channels_per_group: int = 16) -> int:
    """Calculate number of groups for GroupNorm."""
    for num_groups in [32, 16, 8, 4, 2, 1]:
        if channels % num_groups == 0 and channels // num_groups >= min_channels_per_group:
            return num_groups
    return 1


class DCTLayer(nn.Module):
    """
    Learnable DCT-based frequency decomposition.

    Decomposes features into low-frequency and high-frequency components
    using DCT (Discrete Cosine Transform).
    """
    def __init__(self, channels: int, size: int = 8):
        super().__init__()
        self.size = size
        self.channels = channels

        # Pre-compute DCT basis
        self.register_buffer('dct_basis', self._create_dct_basis(size))

        # Learnable frequency threshold (which frequencies are "high")
        self.freq_threshold = nn.Parameter(torch.tensor(0.25))  # Top 25% frequencies

    def _create_dct_basis(self, N: int) -> torch.Tensor:
        """Create 2D DCT basis matrix."""
        basis = torch.zeros(N, N, N, N)
        for u in range(N):
            for v in range(N):
                for x in range(N):
                    for y in range(N):
                        cu = 1.0 / math.sqrt(N) if u == 0 else math.sqrt(2.0 / N)
                        cv = 1.0 / math.sqrt(N) if v == 0 else math.sqrt(2.0 / N)
                        basis[u, v, x, y] = cu * cv * math.cos((2*x+1)*u*math.pi/(2*N)) * \
                                            math.cos((2*y+1)*v*math.pi/(2*N))
        return basis.view(N*N, N*N)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input features

        Returns:
            low_freq: Low frequency components
            high_freq: High frequency components (edges, details)
        """
        B, C, H, W = x.shape

        # Use unfold to extract patches
        patch_size = min(self.size, H, W)
        if H < self.size or W < self.size:
            # Fallback: use simple high-pass filter
            return self._simple_freq_decompose(x)

        # Pad to make divisible
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        B, C, H_pad, W_pad = x.shape

        # Extract patches: (B, C, num_h, num_w, patch_size, patch_size)
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        num_h, num_w = patches.shape[2], patches.shape[3]

        # Reshape for DCT: (B*C*num_h*num_w, patch_size*patch_size)
        patches_flat = patches.permute(0, 1, 2, 3, 4, 5).reshape(-1, patch_size * patch_size)

        # Apply DCT
        dct_basis = self.dct_basis[:patch_size*patch_size, :patch_size*patch_size]
        dct_coeffs = torch.matmul(patches_flat, dct_basis.T.to(x.device))

        # Create frequency mask (high frequencies = far from DC)
        freq_mask = self._create_freq_mask(patch_size, x.device)
        threshold = torch.sigmoid(self.freq_threshold)

        # Separate low and high frequencies
        high_mask = (freq_mask > threshold).float()
        low_mask = 1 - high_mask

        # Apply masks
        low_coeffs = dct_coeffs * low_mask.view(1, -1)
        high_coeffs = dct_coeffs * high_mask.view(1, -1)

        # Inverse DCT
        low_patches = torch.matmul(low_coeffs, dct_basis.to(x.device))
        high_patches = torch.matmul(high_coeffs, dct_basis.to(x.device))

        # Reshape back
        low_patches = low_patches.view(B, C, num_h, num_w, patch_size, patch_size)
        high_patches = high_patches.view(B, C, num_h, num_w, patch_size, patch_size)

        # Fold back to image
        low_freq = low_patches.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H_pad, W_pad)
        high_freq = high_patches.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H_pad, W_pad)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            low_freq = low_freq[:, :, :H_pad-pad_h if pad_h > 0 else H_pad,
                                      :W_pad-pad_w if pad_w > 0 else W_pad]
            high_freq = high_freq[:, :, :H_pad-pad_h if pad_h > 0 else H_pad,
                                        :W_pad-pad_w if pad_w > 0 else W_pad]

        return low_freq, high_freq

    def _create_freq_mask(self, N: int, device) -> torch.Tensor:
        """Create frequency distance mask (normalized 0-1)."""
        u = torch.arange(N, device=device).float()
        v = torch.arange(N, device=device).float()
        u, v = torch.meshgrid(u, v, indexing='ij')
        # Distance from DC (top-left corner in DCT)
        dist = torch.sqrt(u**2 + v**2)
        dist = dist / dist.max()
        return dist.flatten()

    def _simple_freq_decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple frequency decomposition using Gaussian blur."""
        # Low-pass: Gaussian blur
        kernel_size = 5
        sigma = 1.0
        channels = x.shape[1]

        # Create Gaussian kernel
        coords = torch.arange(kernel_size, device=x.device).float() - kernel_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.view(1, -1) * g.view(-1, 1)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

        # Apply low-pass
        padding = kernel_size // 2
        low_freq = F.conv2d(x, kernel, padding=padding, groups=channels)

        # High-pass = original - low-pass
        high_freq = x - low_freq

        return low_freq, high_freq


class HighFrequencyPerception(nn.Module):
    """
    High Frequency Perception (HFP) Module

    Extracts high-frequency information and uses it as attention weights
    for spatial and channel reweighting.

    Key insight: High-frequency = edges, cracks, fine details
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels

        # DCT-based frequency decomposition
        self.dct = DCTLayer(channels, size=8)

        # Spatial attention from high-freq
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GroupNorm(get_num_groups(channels // reduction), channels // reduction),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # Channel attention from high-freq
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid()
        )

        # Learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features

        Returns:
            Enhanced features with high-frequency attention
        """
        # Decompose into low and high frequency
        low_freq, high_freq = self.dct(x)

        # Compute attention weights from high-frequency
        spatial_weight = self.spatial_attn(high_freq)  # (B, 1, H, W)
        channel_weight = self.channel_attn(high_freq)  # (B, C, 1, 1)

        # Apply dual attention
        alpha = torch.sigmoid(self.alpha)
        enhanced = x * (1 + alpha * spatial_weight * channel_weight)

        return enhanced


class SpatialDependencyPerception(nn.Module):
    """
    Spatial Dependency Perception (SDP) Module

    Cross-scale pixel-level attention for spatial alignment.
    Addresses the misalignment issue when upsampling low-resolution features.
    """
    def __init__(self, high_channels: int, low_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Project high-res features (query)
        self.q_proj = nn.Conv2d(high_channels, out_channels, 1, bias=False)
        # Project low-res features (key, value)
        self.k_proj = nn.Conv2d(low_channels, out_channels, 1, bias=False)
        self.v_proj = nn.Conv2d(low_channels, out_channels, 1, bias=False)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
        )

        # Residual connection weight
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, high_res: torch.Tensor, low_res: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_res: (B, C_high, H, W) higher resolution features
            low_res: (B, C_low, H_low, W_low) lower resolution features

        Returns:
            Aligned and fused features (B, C_out, H, W)
        """
        B, _, H, W = high_res.shape
        _, _, H_low, W_low = low_res.shape

        # Upsample low-res to match high-res spatial size
        low_res_up = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=False)

        # Project to Q, K, V
        Q = self.q_proj(high_res)  # (B, C, H, W)
        K = self.k_proj(low_res_up)  # (B, C, H, W)
        V = self.v_proj(low_res_up)  # (B, C, H, W)

        # Reshape for attention
        B, C, H, W = Q.shape
        Q = Q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        K = K.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        V = V.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)

        # Compute attention (local window to reduce computation)
        # For efficiency, we use a simplified version
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, heads, HW, HW)

        # Apply softmax along key dimension
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = (attn @ V).permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.out_proj(out)

        # Residual connection
        return high_res + self.gamma * out


class HSFPNMultiScaleFusion(nn.Module):
    """
    HS-FPN style Multi-Scale Fusion for Grid Classification

    Combines:
    1. High Frequency Perception (HFP) at each scale
    2. Spatial Dependency Perception (SDP) for cross-scale fusion
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        target_size: int = 32,
        use_hfp: bool = True,
        use_sdp: bool = True,
    ):
        super().__init__()
        self.target_size = target_size
        self.use_hfp = use_hfp
        self.use_sdp = use_sdp
        self.num_levels = len(in_channels_list)

        # Lateral convolutions (1x1 to unify channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in in_channels_list
        ])

        # HFP modules for each level
        if use_hfp:
            self.hfp_modules = nn.ModuleList([
                HighFrequencyPerception(out_channels, reduction=4)
                for _ in range(self.num_levels)
            ])

        # SDP modules for cross-scale fusion (top-down)
        if use_sdp:
            self.sdp_modules = nn.ModuleList([
                SpatialDependencyPerception(out_channels, out_channels, out_channels, num_heads=4)
                for _ in range(self.num_levels - 1)
            ])

        # Final fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
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

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [P3, P4, P5] features from encoder
                     P3: stride-8, P4: stride-16, P5: stride-32

        Returns:
            Fused features (B, out_channels, target_size, target_size)
        """
        assert len(features) == self.num_levels


        laterals = []
        for i, (feat, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            lat = lateral_conv(feat)
            if self.use_hfp:
                lat = self.hfp_modules[i](lat)
            laterals.append(lat)


        # Start from coarsest (P5) and go to finest (P3)
        if self.use_sdp:
            for i in range(self.num_levels - 1, 0, -1):
                # SDP: align low-res (laterals[i]) to high-res (laterals[i-1])
                laterals[i-1] = self.sdp_modules[i-1](laterals[i-1], laterals[i])
        else:
            # Simple top-down addition
            for i in range(self.num_levels - 1, 0, -1):
                laterals[i-1] = laterals[i-1] + F.interpolate(
                    laterals[i], size=laterals[i-1].shape[2:],
                    mode='bilinear', align_corners=False
                )


        out = laterals[0]
        if out.shape[2] != self.target_size or out.shape[3] != self.target_size:
            out = F.interpolate(out, size=(self.target_size, self.target_size),
                               mode='bilinear', align_corners=False)


        out = self.fusion_conv(out)

        return out


class DirectionAwareConv(nn.Module):
    """
    Direction-Aware Convolution for crack detection.
    Uses asymmetric kernels to capture horizontal and vertical patterns.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Horizontal pattern (for TC - transverse cracks)
        self.h_conv = nn.Conv2d(channels, channels // 2, (1, 7), padding=(0, 3),
                                groups=channels // 2, bias=False)
        # Vertical pattern (for LC - longitudinal cracks)
        self.v_conv = nn.Conv2d(channels, channels // 2, (7, 1), padding=(3, 0),
                                groups=channels // 2, bias=False)
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_feat = self.h_conv(x)
        v_feat = self.v_conv(x)
        out = torch.cat([h_feat, v_feat], dim=1)
        return self.fusion(out) + x


@register()
class HSFPNGridHead(nn.Module):
    """HS-FPN Inspired Grid Classification Head"""
    def __init__(
        self,
        in_channels: int = 224,
        hidden_dim: int = 224,
        num_classes: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
        use_hfp: bool = True,
        use_sdp: bool = True,
        target_size: int = 32,
        prior_probs: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.prior_probs = prior_probs

        # Multi-scale fusion with HFP and SDP
        if isinstance(in_channels, int):
            in_channels_list = [in_channels] * 3
        else:
            in_channels_list = list(in_channels)

        self.multi_scale_fusion = HSFPNMultiScaleFusion(
            in_channels_list,
            hidden_dim,
            target_size=target_size,
            use_hfp=use_hfp,
            use_sdp=use_sdp,
        )

        # Direction-aware blocks
        self.blocks = nn.Sequential(*[
            DirectionAwareConv(hidden_dim)
            for _ in range(num_blocks)
        ])

        # Classification head
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

        # Initialize classification head bias with prior probabilities
        if hasattr(self.cls_head, 'bias') and self.cls_head.bias is not None:
            if self.prior_probs is not None:
                prior_probs = torch.tensor(self.prior_probs, dtype=torch.float32)
                eps = 1e-8
                prior_probs = torch.clamp(prior_probs, eps, 1 - eps)
                bias_init = -torch.log((1 - prior_probs) / prior_probs)
                with torch.no_grad():
                    self.cls_head.bias.copy_(bias_init)
                print(f"[HSFPNGridHead] Bias initialized with prior_probs: {self.prior_probs}")
            else:
                default_prior = 0.01
                bias_init = -torch.log(torch.tensor((1 - default_prior) / default_prior))
                nn.init.constant_(self.cls_head.bias, bias_init)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of encoder features [stride-8, stride-16, stride-32]

        Returns:
            Grid classification logits (B, num_classes, H, W)
        """


        x = self.multi_scale_fusion(features)


        x = self.blocks(x)


        x = self.dropout(x)
        logits = self.cls_head(x)

        return logits
