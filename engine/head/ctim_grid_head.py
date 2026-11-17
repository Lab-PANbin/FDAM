"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

CTIM-Enhanced Grid Classification Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from ..core import register

from .direction_aware_grid_head import (
    get_num_groups,
    MultiScaleFusion,
    DirectionAwareBlock,
    AdaptiveDirectionFusion,
)
from .frequency_attention_grid_head import FrequencyDynamicAttention
from .cross_task_interaction import CrossTaskInteractionModule


@register()
class CTIMGridHead(nn.Module):
    """CTIM-Enhanced Grid Classification Head"""
    def __init__(
        self,
        in_channels: int = 224,
        hidden_dim: int = 224,
        num_classes: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
        # Feature options
        use_adf: bool = True,
        use_multi_scale: bool = True,
        use_frequency_attention: bool = True,
        # CTIM options
        use_ctim: bool = True,
        det_dim: int = 256,
        ctim_num_heads: int = 8,
        # Frequency attention options
        freq_num_bands: int = 4,
        freq_use_direction: bool = True,
        prior_probs: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_adf = use_adf
        self.use_multi_scale = use_multi_scale
        self.use_frequency_attention = use_frequency_attention
        self.use_ctim = use_ctim
        self.prior_probs = prior_probs

        # Multi-scale fusion
        if use_multi_scale:
            if isinstance(in_channels, int):
                in_channels_list = [in_channels] * 3
            else:
                in_channels_list = list(in_channels)
            self.multi_scale_fusion = MultiScaleFusion(
                in_channels_list, hidden_dim, target_size=32
            )
        else:
            proj_in = in_channels if isinstance(in_channels, int) else in_channels[-1]
            self.multi_scale_fusion = None
            self.input_proj = nn.Sequential(
                nn.Conv2d(proj_in, hidden_dim, 1, bias=False),
                nn.GroupNorm(get_num_groups(hidden_dim), hidden_dim),
                nn.SiLU(inplace=True)
            )

        # Frequency-Dynamic Attention
        if use_frequency_attention:
            self.freq_attention = FrequencyDynamicAttention(
                channels=hidden_dim,
                num_bands=freq_num_bands,
                use_direction_mask=freq_use_direction,
                reduction=4,
            )

        # Cross-Task Interaction Module
        if use_ctim:
            self.ctim = CrossTaskInteractionModule(
                det_dim=det_dim,
                grid_dim=hidden_dim,
                num_heads=ctim_num_heads,
                dropout=dropout,
                use_det_to_grid=True,
                use_grid_to_det=True,
            )
            # Projection for detection features if dimensions don't match
            if det_dim != hidden_dim:
                self.det_proj = nn.Linear(det_dim, hidden_dim)
            else:
                self.det_proj = nn.Identity()

        # Direction-Aware Blocks
        self.blocks = nn.Sequential(*[
            DirectionAwareBlock(hidden_dim, use_dsfd=True, use_spatial_attention=False)
            for _ in range(num_blocks)
        ])

        # Adaptive Direction Fusion
        self.adf = AdaptiveDirectionFusion(hidden_dim) if use_adf else nn.Identity()

        # Dropout and classification
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
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
                print(f"[CTIMGridHead] Bias initialized with prior_probs: {self.prior_probs}")
            else:
                default_prior = 0.01
                bias_init = -torch.log(torch.tensor((1 - default_prior) / default_prior))
                nn.init.constant_(self.cls_head.bias, bias_init)

    def forward(
        self,
        features: List[torch.Tensor],
        det_features: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass."""

        if self.use_multi_scale and self.multi_scale_fusion is not None:
            x = self.multi_scale_fusion(features)
        else:
            x = features[-1]
            x = self.input_proj(x)


        if self.use_frequency_attention:
            x = self.freq_attention(x)

        # Store intermediate features for CTIM
        grid_feat = x


        det_enhanced = None
        if self.use_ctim and det_features is not None:

            # Project detection features if needed
            det_feat_proj = self.det_proj(det_features)

            # Apply CTIM
            det_enhanced, grid_enhanced = self.ctim(
                det_feat=det_feat_proj,
                grid_feat=grid_feat,
            )

            # Use enhanced grid features
            x = grid_enhanced


        x = self.blocks(x)


        x = self.adf(x)


        x = self.dropout(x)
        logits = self.cls_head(x)

        # Return format depends on whether CTIM was used
        if det_features is not None and self.use_ctim:
            return {
                'grid_logits': logits,
                'grid_features': grid_feat,
                'det_enhanced': det_enhanced,
            }
        else:
            return logits

    def get_intermediate_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Get intermediate grid features (before classification).
        Useful for CTIM when called from DEIM model.

        Args:
            features: List of encoder features

        Returns:
            Intermediate features (B, hidden_dim, H, W)
        """
        if self.use_multi_scale and self.multi_scale_fusion is not None:
            x = self.multi_scale_fusion(features)
        else:
            x = features[-1]
            x = self.input_proj(x)

        if self.use_frequency_attention:
            x = self.freq_attention(x)

        return x
