"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Grid-Guided Feature Enhancer
"""

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register


__all__ = ['GridGuidedFeatureEnhancer']


def _choose_gn_groups(channels: int, max_groups: int = 32) -> int:
    """
    Choose a valid GroupNorm group number.
    For channels=192, this returns 32.
    """
    groups = min(max_groups, channels)
    while groups > 1:
        if channels % groups == 0:
            return groups
        groups -= 1
    return 1


def _make_norm(channels: int, norm_type: str = 'gn', gn_groups: int = 32):
    norm_type = norm_type.lower()

    if norm_type == 'bn':
        return nn.BatchNorm2d(channels)

    if norm_type == 'gn':
        groups = _choose_gn_groups(channels, gn_groups)
        return nn.GroupNorm(groups, channels)

    if norm_type in ('none', 'identity'):
        return nn.Identity()

    raise ValueError(f"Unsupported norm_type: {norm_type}")


class _EnhanceBlock(nn.Module):
    """
    Lightweight residual transform block.

    Important:
        If init_zero=True, the last norm layer is zero-initialized,
        so this block outputs zero at initialization.
        Therefore:
            enhanced = feat + gain * conf * 0 = feat
        at the beginning of training.
    """

    def __init__(
        self,
        channels: int,
        norm_type: str = 'gn',
        gn_groups: int = 32,
        init_zero: bool = True,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _make_norm(channels, norm_type=norm_type, gn_groups=gn_groups),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            _make_norm(channels, norm_type=norm_type, gn_groups=gn_groups),
            nn.SiLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _make_norm(channels, norm_type=norm_type, gn_groups=gn_groups),
        )

        if init_zero:
            last_norm = self.block[-1]
            if hasattr(last_norm, 'weight') and last_norm.weight is not None:
                nn.init.zeros_(last_norm.weight)
            if hasattr(last_norm, 'bias') and last_norm.bias is not None:
                nn.init.zeros_(last_norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register()
class GridGuidedFeatureEnhancer(nn.Module):
    """Grid-guided detection feature enhancement."""

    def __init__(
        self,
        num_scales: int = 3,
        channels: int = 256,
        num_classes: int = 1,
        detach_grid: bool = True,
        max_gain: float = 0.03,
        use_smooth: bool = False,
        smooth_kernel_size: Union[int, Sequence[int]] = 3,
        min_gate: float = 0.0,
        init_identity: bool = True,
        norm_type: str = 'gn',
        gn_groups: int = 32,
        conf_is_logits: bool = True,
        gate_power: float = 1.0,
        enhance_levels: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        self.num_scales = num_scales
        self.channels = channels
        self.num_classes = num_classes
        self.detach_grid = detach_grid
        self.max_gain = float(max_gain)
        self.use_smooth = use_smooth
        self.min_gate = float(min_gate)
        self.init_identity = init_identity
        self.norm_type = norm_type
        self.gn_groups = gn_groups
        self.conf_is_logits = conf_is_logits
        self.gate_power = float(gate_power)

        self.smooth_kernel_size = self._parse_level_values(
            smooth_kernel_size,
            num_scales,
            name='smooth_kernel_size',
        )

        for k in self.smooth_kernel_size:
            if k < 1:
                raise ValueError(
                    f"smooth_kernel_size must be >= 1, got {k}"
                )
            if k % 2 == 0:
                raise ValueError(
                    f"smooth_kernel_size must be odd, got {k}"
                )

        self.feat_proj = nn.ModuleList([
            _EnhanceBlock(
                channels=channels,
                norm_type=norm_type,
                gn_groups=gn_groups,
                init_zero=init_identity,
            )
            for _ in range(num_scales)
        ])

        # Non-negative gain.
        # Initial sigmoid(0)=0.5, but residual branch is zero-initialized when
        # init_identity=True, so the whole module is still exact identity at init.
        self.level_gain = nn.Parameter(torch.zeros(num_scales))

        if enhance_levels is None:
            level_mask = torch.ones(num_scales, dtype=torch.float32)
        else:
            level_mask = torch.zeros(num_scales, dtype=torch.float32)
            for idx in enhance_levels:
                if idx < 0 or idx >= num_scales:
                    raise ValueError(
                        f"enhance_levels contains invalid index {idx}, "
                        f"but num_scales={num_scales}"
                    )
                level_mask[idx] = 1.0

        # Use buffer mask instead of if-continue skip.
        # This avoids DDP unused-parameter issues.
        self.register_buffer('level_mask', level_mask, persistent=False)

    @staticmethod
    def _parse_level_values(
        value: Union[int, Sequence[int]],
        num_scales: int,
        name: str,
    ):
        if isinstance(value, int):
            return [value for _ in range(num_scales)]

        if isinstance(value, (list, tuple)):
            if len(value) != num_scales:
                raise ValueError(
                    f"{name} length={len(value)} must equal num_scales={num_scales}"
                )
            return [int(v) for v in value]

        raise TypeError(
            f"{name} must be int/list/tuple, got {type(value)}"
        )

    def _to_objectness(self, conf: torch.Tensor) -> torch.Tensor:
        if conf.dim() != 4:
            raise ValueError(
                f"grid_conf_map must be 4D Tensor[B, C, H, W], got {tuple(conf.shape)}"
            )

        if conf.shape[1] > 1:
            conf = conf.max(dim=1, keepdim=True).values
        elif conf.shape[1] == 1:
            pass
        else:
            raise ValueError(
                f"grid_conf_map channel must be >= 1, got {conf.shape[1]}"
            )

        return conf

    def _smooth_conf(self, conf: torch.Tensor, level_idx: int) -> torch.Tensor:
        if not self.use_smooth:
            return conf

        k = self.smooth_kernel_size[level_idx]

        # k=1 means no smoothing for this level.
        if k <= 1:
            return conf

        pad = k // 2

        # Be careful:
        # For stride 8 / 16 / 32, k=5 corresponds to large image regions.
        # So level-wise k is safer.
        conf = F.max_pool2d(conf, kernel_size=k, stride=1, padding=pad)
        conf = F.avg_pool2d(conf, kernel_size=k, stride=1, padding=pad)

        return conf

    def forward(self, encoder_feats: list, grid_conf_maps: list) -> list:
        """Args:
            encoder_feats:
                List[Tensor[B, C, H_i, W_i]]"""

        if not isinstance(encoder_feats, (list, tuple)):
            raise TypeError(
                f"encoder_feats must be list/tuple, got {type(encoder_feats)}"
            )

        if not isinstance(grid_conf_maps, (list, tuple)):
            raise TypeError(
                f"grid_conf_maps must be list/tuple, got {type(grid_conf_maps)}"
            )

        if len(encoder_feats) != self.num_scales:
            raise ValueError(
                f"len(encoder_feats)={len(encoder_feats)} must equal "
                f"num_scales={self.num_scales}"
            )

        if len(grid_conf_maps) != self.num_scales:
            raise ValueError(
                f"len(grid_conf_maps)={len(grid_conf_maps)} must equal "
                f"num_scales={self.num_scales}"
            )

        enhanced_feats = []

        for i, (feat, conf) in enumerate(zip(encoder_feats, grid_conf_maps)):
            if feat.dim() != 4:
                raise ValueError(
                    f"Each encoder_feat must be 4D Tensor[B, C, H, W], "
                    f"got shape {tuple(feat.shape)}"
                )

            if feat.shape[1] != self.channels:
                raise ValueError(
                    f"encoder_feat channel={feat.shape[1]} must equal "
                    f"channels={self.channels}"
                )

            if conf.shape[0] != feat.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: grid_conf_map B={conf.shape[0]}, "
                    f"encoder_feat B={feat.shape[0]}"
                )

            if self.detach_grid:
                conf = conf.detach()

            conf = conf.to(device=feat.device, dtype=feat.dtype)
            conf = self._to_objectness(conf)

            if self.conf_is_logits:
                conf = conf.sigmoid()

            if conf.shape[-2:] != feat.shape[-2:]:
                conf = F.interpolate(
                    conf,
                    size=feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )

            conf = torch.nan_to_num(
                conf,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )

            conf = conf.clamp(min=0.0, max=1.0)
            conf = self._smooth_conf(conf, level_idx=i)
            conf = conf.clamp(min=self.min_gate, max=1.0)

            if self.gate_power != 1.0:
                conf = conf.pow(self.gate_power)

            residual = self.feat_proj[i](feat)
            residual = torch.nan_to_num(
                residual,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # Non-negative gain.
            gain = self.max_gain * torch.sigmoid(self.level_gain[i])

            # Multiplicative level mask.
            # Do not use if-continue to skip levels, otherwise DDP may have
            # unused parameters and training may hang or error.
            gain = gain * self.level_mask[i].to(device=feat.device, dtype=feat.dtype)

            enhanced = feat + gain.view(1, 1, 1, 1) * conf * residual
            enhanced_feats.append(enhanced)

        return enhanced_feats
