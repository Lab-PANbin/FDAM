"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Mixture of Experts (MoE) Grid Classification Head for Road Distress Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from ..core import register


def get_num_groups(channels: int, min_channels_per_group: int = 16) -> int:
    """Calculate number of groups for GroupNorm."""
    for num_groups in [32, 16, 8, 4, 2, 1]:
        if channels % num_groups == 0 and channels // num_groups >= min_channels_per_group:
            return num_groups
    return 1


class CrackExpert(nn.Module):
    """Extract directional crack features."""
    def __init__(self, channels: int):
        super().__init__()
        half_ch = channels // 2


        self.h_conv = nn.Sequential(
            nn.Conv2d(channels, half_ch, (1, 7), padding=(0, 3), bias=False),
            nn.GroupNorm(get_num_groups(half_ch), half_ch),
            nn.SiLU(inplace=True),
        )


        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, half_ch, (7, 1), padding=(3, 0), bias=False),
            nn.GroupNorm(get_num_groups(half_ch), half_ch),
            nn.SiLU(inplace=True),
        )


        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_feat = self.h_conv(x)
        v_feat = self.v_conv(x)
        out = torch.cat([h_feat, v_feat], dim=1)
        return self.fusion(out)


class StripExpert(nn.Module):
    """Extract elongated regional features."""
    def __init__(self, channels: int):
        super().__init__()


        self.conv3 = nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(channels, channels // 2, 5, padding=2, bias=False)

        self.norm = nn.GroupNorm(get_num_groups(channels), channels)
        self.act = nn.SiLU(inplace=True)


        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        out = torch.cat([feat3, feat5], dim=1)
        out = self.act(self.norm(out))


        attn = self.channel_attn(out)
        return out * attn


class RegionExpert(nn.Module):
    """Extract broad regional features."""
    def __init__(self, channels: int):
        super().__init__()


        self.conv1 = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.conv_d2 = nn.Conv2d(channels, channels // 4, 3, padding=2, dilation=2, bias=False)
        self.conv_d4 = nn.Conv2d(channels, channels // 4, 3, padding=4, dilation=4, bias=False)


        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
        )

        self.fusion = nn.Sequential(
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        feat1 = self.conv1(x)
        feat_d2 = self.conv_d2(x)
        feat_d4 = self.conv_d4(x)
        feat_global = self.global_pool(x).expand(-1, -1, H, W)

        out = torch.cat([feat1, feat_d2, feat_d4, feat_global], dim=1)
        return self.fusion(out)


class SpatialRouter(nn.Module):
    """Predict expert weights at each spatial location."""
    def __init__(self, channels: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k


        self.router = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(channels // 2), channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, num_experts, 1, bias=True),
        )


        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the module to the input tensors."""
        logits = self.router(x)  # (B, num_experts, H, W)

        # Top-K selection
        B, E, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(B * H * W, E)  # (BHW, E)

        top_k_logits, top_k_indices = torch.topk(logits_flat, self.top_k, dim=-1)  # (BHW, top_k)

        # Softmax over top-k only
        temperature = torch.clamp(self.temperature, min=0.1)
        top_k_weights = F.softmax(top_k_logits / temperature, dim=-1)  # (BHW, top_k)

        # Reshape back
        top_k_indices = top_k_indices.reshape(B, H, W, self.top_k).permute(0, 3, 1, 2)  # (B, top_k, H, W)
        top_k_weights = top_k_weights.reshape(B, H, W, self.top_k).permute(0, 3, 1, 2)  # (B, top_k, H, W)

        # Full weights for auxiliary loss
        full_weights = F.softmax(logits / temperature, dim=1)  # (B, E, H, W)

        return top_k_weights, top_k_indices, full_weights


class MoELayer(nn.Module):
    """Combine expert features using spatial routing weights."""
    def __init__(
        self,
        channels: int,
        num_experts: int = 3,
        top_k: int = 2,
        use_load_balance: bool = True,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_load_balance = use_load_balance
        self.load_balance_weight = load_balance_weight


        self.experts = nn.ModuleList([
            CrackExpert(channels),
            StripExpert(channels),
            RegionExpert(channels),
        ])


        for _ in range(num_experts - 3):
            self.experts.append(self._create_generic_expert(channels))


        self.router = SpatialRouter(channels, num_experts, top_k)


        self.aux_loss = 0.0

    def _create_generic_expert(self, channels: int) -> nn.Module:
        """Create the generic convolutional expert."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features

        Returns:
            output: (B, C, H, W) MoE processed features
        """
        B, C, H, W = x.shape


        top_k_weights, top_k_indices, full_weights = self.router(x)
        # top_k_weights: (B, top_k, H, W)
        # top_k_indices: (B, top_k, H, W)
        # full_weights: (B, num_experts, H, W)


        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # expert_outputs: (B, num_experts, C, H, W)


        output = torch.zeros_like(x)
        for k in range(self.top_k):

            indices_k = top_k_indices[:, k, :, :]  # (B, H, W)
            weights_k = top_k_weights[:, k, :, :]  # (B, H, W)


            indices_expanded = indices_k.unsqueeze(1).unsqueeze(2).expand(-1, 1, C, -1, -1)
            # indices_expanded: (B, 1, C, H, W)

            selected_output = torch.gather(expert_outputs, 1, indices_expanded).squeeze(1)
            # selected_output: (B, C, H, W)

            output = output + weights_k.unsqueeze(1) * selected_output


        if self.training and self.use_load_balance:
            self.aux_loss = self._compute_load_balance_loss(full_weights)
        else:
            self.aux_loss = 0.0

        return output

    def _compute_load_balance_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Penalize imbalanced expert utilization."""

        # weights: (B, E, H, W) -> mean over B, H, W -> (E,)
        expert_usage = weights.mean(dim=(0, 2, 3))  # (num_experts,)


        target_usage = 1.0 / self.num_experts


        load_balance_loss = ((expert_usage - target_usage) ** 2).sum() * self.num_experts

        return load_balance_loss * self.load_balance_weight

    def get_expert_usage(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return mean expert utilization."""
        with torch.no_grad():
            _, top_k_indices, full_weights = self.router(x)

            expert_names = ['Crack', 'Strip', 'Region'] + [f'Generic_{i}' for i in range(self.num_experts - 3)]

            usage = {}
            for i, name in enumerate(expert_names[:self.num_experts]):
                usage[name] = full_weights[:, i].mean().item()

            return usage


class MultiScaleFusion(nn.Module):
    """Fuse multi-scale features at the coarsest spatial resolution."""
    def __init__(self, in_channels_list: List[int], out_channels: int, target_size: int = 32):
        super().__init__()
        self.target_size = target_size

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.GroupNorm(get_num_groups(out_channels), out_channels),
                nn.SiLU(inplace=True),
            ) for in_ch in in_channels_list
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:

        aligned = []
        for feat, lateral in zip(features, self.lateral_convs):
            x = lateral(feat)
            if x.shape[2] != self.target_size or x.shape[3] != self.target_size:
                x = F.interpolate(x, size=(self.target_size, self.target_size),
                                 mode='bilinear', align_corners=False)
            aligned.append(x)


        out = torch.cat(aligned, dim=1)
        return self.fusion(out)


@register()
class MoEGridHead(nn.Module):
    """Classify grid occupancy with spatially routed experts."""
    def __init__(
        self,
        in_channels: int = 224,
        hidden_dim: int = 224,
        num_classes: int = 4,
        num_experts: int = 3,
        top_k: int = 2,
        num_moe_layers: int = 2,
        dropout: float = 0.1,
        use_load_balance: bool = True,
        load_balance_weight: float = 0.01,
        prior_probs: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.prior_probs = prior_probs
        self.num_moe_layers = num_moe_layers


        if isinstance(in_channels, int):
            in_channels_list = [in_channels] * 3
        else:
            in_channels_list = list(in_channels)

        self.multi_scale_fusion = MultiScaleFusion(
            in_channels_list, hidden_dim, target_size=32
        )


        self.moe_layers = nn.ModuleList([
            MoELayer(
                channels=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                use_load_balance=use_load_balance,
                load_balance_weight=load_balance_weight,
            ) for _ in range(num_moe_layers)
        ])


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


        if hasattr(self.cls_head, 'bias') and self.cls_head.bias is not None:
            if self.prior_probs is not None:
                prior_probs = torch.tensor(self.prior_probs, dtype=torch.float32)
                eps = 1e-8
                prior_probs = torch.clamp(prior_probs, eps, 1 - eps)
                bias_init = -torch.log((1 - prior_probs) / prior_probs)
                with torch.no_grad():
                    self.cls_head.bias.copy_(bias_init)
                print(f"[MoEGridHead] Bias initialized with prior_probs: {self.prior_probs}")
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


        for moe_layer in self.moe_layers:
            x = moe_layer(x) + x


        x = self.dropout(x)
        logits = self.cls_head(x)

        return logits

    def get_aux_loss(self) -> torch.Tensor:
        """Return the expert load-balancing loss."""
        total_aux_loss = 0.0
        for moe_layer in self.moe_layers:
            total_aux_loss = total_aux_loss + moe_layer.aux_loss
        return total_aux_loss

    def get_expert_usage(self, features: List[torch.Tensor]) -> Dict[str, float]:
        """Return mean expert utilization."""
        x = self.multi_scale_fusion(features)

        all_usage = {}
        for i, moe_layer in enumerate(self.moe_layers):
            usage = moe_layer.get_expert_usage(x)
            for name, value in usage.items():
                all_usage[f'layer{i}_{name}'] = value
            x = moe_layer(x) + x

        return all_usage
