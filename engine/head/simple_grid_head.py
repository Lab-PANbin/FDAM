"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Simple Grid Classification Head - Inspired by GCBD (YOLOv5 Patch Classify)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register


@register()
class SimpleGridHead(nn.Module):
    """A simple grid classification head using 1x1 convolution."""

    def __init__(
        self,
        in_channels=256,
        num_classes: int = 4,
        use_bias: bool = True,
        feature_idx: int = 2,
        prior_probs=None,
        target_size: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_idx = feature_idx
        self.prior_probs = prior_probs
        self.target_size = target_size

        # Handle list input for in_channels
        if isinstance(in_channels, (list, tuple)):
            in_ch = in_channels[feature_idx]
        else:
            in_ch = in_channels

        self.grid_conv = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=use_bias)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.grid_conv.weight, std=0.01)
        if self.grid_conv.bias is not None:
            if self.prior_probs is not None:
                # Per-class prior probability initialization
                prior_probs = torch.tensor(self.prior_probs, dtype=torch.float32)
                assert len(prior_probs) == self.num_classes
                eps = 1e-8
                prior_probs = torch.clamp(prior_probs, eps, 1 - eps)
                bias_init = -torch.log((1 - prior_probs) / prior_probs)
                with torch.no_grad():
                    self.grid_conv.bias.copy_(bias_init)
                print(f"[SimpleGridHead] Initialized bias with prior_probs: {self.prior_probs}")
            else:
                nn.init.constant_(self.grid_conv.bias, -4.6)  # sigmoid(-4.6) ≈ 0.01

    def forward(self, features):
        """
        Args:
            features: List of feature maps from encoder, or single feature map.
                     Uses feature at index `feature_idx` if list.

        Returns:
            Grid logits of shape (B, num_classes, target_size, target_size)
        """
        if isinstance(features, (list, tuple)):
            x = features[self.feature_idx]
        else:
            x = features

        logits = self.grid_conv(x)

        # Resize to target size if needed (GCBD uses 32x32)
        if self.target_size is not None and (logits.shape[2] != self.target_size or logits.shape[3] != self.target_size):
            logits = F.interpolate(logits, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)

        return logits


@register()
class LightGridHead(nn.Module):
    """A slightly more capable grid head with one hidden layer.
    Still much simpler than GatedGridHead."""

    def __init__(
        self,
        in_channels = 256,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.0,
        feature_idx: int = 0,
        target_size: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_idx = feature_idx
        self.target_size = target_size

        # Handle list input for in_channels
        if isinstance(in_channels, (list, tuple)):
            in_ch = in_channels[feature_idx]
        else:
            in_ch = in_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -4.6)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        if isinstance(features, (list, tuple)):
            x = features[self.feature_idx]
        else:
            x = features

        logits = self.layers(x)

        # Resize to target size if needed (GCBD uses 32x32)
        if self.target_size is not None and (logits.shape[2] != self.target_size or logits.shape[3] != self.target_size):
            logits = F.interpolate(logits, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)

        return logits
