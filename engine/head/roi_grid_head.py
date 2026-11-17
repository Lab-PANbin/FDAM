"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

RoI-based Grid Classification Head for Road Distress Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from typing import Dict, List, Optional, Tuple
from ..core import register


class RoIFeatureExtractor(nn.Module):
    """
    Extract and enhance features from RoI regions.

    Similar to Mask R-CNN's mask head but adapted for grid classification.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_conv_layers: int = 4,
        use_deformable: bool = False,
    ):
        super().__init__()

        # Feature extraction layers (similar to Mask R-CNN mask head)
        layers = []
        for i in range(num_conv_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            layers.extend([
                nn.Conv2d(in_ch, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
            ])

        self.feature_extractor = nn.Sequential(*layers)

        # Optional: Deformable convolution for better alignment
        if use_deformable:
            try:
                from torchvision.ops import DeformConv2d
                self.deform_conv = DeformConv2d(hidden_dim, hidden_dim, 3, padding=1)
                self.offset_conv = nn.Conv2d(hidden_dim, 18, 3, padding=1)  # 2*3*3 offsets
            except ImportError:
                print("[RoIFeatureExtractor] DeformConv2d not available, using standard conv")
                self.deform_conv = None
        else:
            self.deform_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RoI features (N, C, H, W)
        Returns:
            Enhanced features (N, hidden_dim, H, W)
        """
        x = self.feature_extractor(x)

        if self.deform_conv is not None:
            offset = self.offset_conv(x)
            x = self.deform_conv(x, offset)

        return x


@register()
class RoIGridClassificationHead(nn.Module):
    """RoI-based Grid Classification Head"""
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 4,
        roi_size: int = 14,
        grid_size: int = 32,
        num_conv_layers: int = 4,
        use_deformable: bool = False,
        score_threshold: float = 0.05,
        use_gt_boxes_train: bool = True,
        spatial_scale: float = 1/16,  # stride-16 feature map
    ):
        super().__init__()
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.grid_size = grid_size
        self.score_threshold = score_threshold
        self.use_gt_boxes_train = use_gt_boxes_train
        self.spatial_scale = spatial_scale

        # RoI Align layer
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )

        # Feature extractor (Mask R-CNN style)
        self.feature_extractor = RoIFeatureExtractor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            use_deformable=use_deformable,
        )

        # Grid classification head
        # Upsample to roi_size if needed, then classify
        self.grid_classifier = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2),  # 14→28 or 7→14
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

        # Fallback: Dense grid head for images without detections
        # Handle list input for in_channels
        fallback_in_channels = in_channels
        if isinstance(in_channels, list):
            fallback_in_channels = in_channels[1] if len(in_channels) > 1 else in_channels[0]

        self.fallback_head = nn.Sequential(
            nn.Conv2d(fallback_in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        features: List[torch.Tensor],
        det_outputs: Optional[Dict] = None,
        targets: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: List of backbone/encoder features [stride-8, stride-16, stride-32]
            det_outputs: Detection outputs containing boxes and scores
            targets: Ground truth targets (used in training if use_gt_boxes_train=True)

        Returns:
            Grid classification logits (B, num_classes, grid_size, grid_size)
        """
        # Handle both list and single tensor input
        if isinstance(features, list):
            # Use stride-16 feature (middle scale, good balance)
            feat_map = features[1] if len(features) > 1 else features[-1]
        else:
            feat_map = features

        B, C, H, W = feat_map.shape

        # Initialize output grid
        device = feat_map.device
        grid_logits = torch.zeros(
            B, self.num_classes, self.grid_size, self.grid_size,
            device=device, dtype=feat_map.dtype
        )

        # Get boxes for RoI Align
        if self.training and self.use_gt_boxes_train and targets is not None:
            # Use GT boxes during training for better supervision
            boxes_list = self._get_gt_boxes(targets, B)
        elif det_outputs is not None:
            # Use predicted boxes
            boxes_list = self._get_pred_boxes(det_outputs, B)
        else:
            # No boxes available, use fallback dense prediction
            return self._fallback_prediction(feat_map)

        # Process each image in batch
        for batch_idx in range(B):
            boxes = boxes_list[batch_idx]  # (N, 4) in [x1, y1, x2, y2] format

            if boxes is None or len(boxes) == 0:
                # No boxes, use fallback for this image
                grid_logits[batch_idx] = self._fallback_prediction(feat_map[batch_idx:batch_idx+1])[0]
                continue

            # Prepare boxes for RoI Align: add batch index
            # RoIAlign expects (N, 5) with [batch_idx, x1, y1, x2, y2]
            batch_indices = torch.full((len(boxes), 1), batch_idx, device=device, dtype=boxes.dtype)
            roi_boxes = torch.cat([batch_indices, boxes], dim=1)

            # Extract RoI features
            roi_features = self.roi_align(feat_map, roi_boxes)  # (N, C, roi_size, roi_size)

            # Enhance features
            roi_features = self.feature_extractor(roi_features)  # (N, hidden_dim, roi_size, roi_size)

            # Classify grid
            roi_grids = self.grid_classifier(roi_features)  # (N, num_classes, roi_size*2, roi_size*2)

            # Map RoI grids back to image-level grid
            grid_logits[batch_idx] = self._aggregate_roi_grids(
                roi_grids, boxes, self.grid_size, device
            )

        return grid_logits

    def _get_gt_boxes(self, targets: Dict, batch_size: int) -> List[Optional[torch.Tensor]]:
        """Extract GT boxes from targets."""
        boxes_list = []

        # Assuming targets format: {'boxes': (N, 4), 'batch_idx': (N,), ...}
        if 'boxes' in targets and 'batch_idx' in targets:
            all_boxes = targets['boxes']  # (N, 4) in normalized [cx, cy, w, h]
            batch_idx = targets['batch_idx']  # (N,)

            for b in range(batch_size):
                mask = batch_idx == b
                if mask.any():
                    boxes = all_boxes[mask]
                    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
                    boxes = self._cxcywh_to_xyxy(boxes)
                    # Scale to feature map size (assuming input is 1024, feature is 64 for stride-16)
                    boxes = boxes * (1024 * self.spatial_scale)
                    boxes_list.append(boxes)
                else:
                    boxes_list.append(None)
        else:
            boxes_list = [None] * batch_size

        return boxes_list

    def _get_pred_boxes(self, det_outputs: Dict, batch_size: int) -> List[Optional[torch.Tensor]]:
        """Extract predicted boxes from detection outputs."""
        boxes_list = []

        # Assuming det_outputs format from DEIM decoder
        if 'pred_boxes' in det_outputs and 'pred_logits' in det_outputs:
            pred_boxes = det_outputs['pred_boxes']  # (B, num_queries, 4)
            pred_logits = det_outputs['pred_logits']  # (B, num_queries, num_classes)

            # Get scores
            pred_scores = pred_logits.sigmoid().max(dim=-1)[0]  # (B, num_queries)

            for b in range(batch_size):
                scores = pred_scores[b]
                boxes = pred_boxes[b]

                # Filter by score threshold
                mask = scores > self.score_threshold
                if mask.any():
                    boxes = boxes[mask]
                    # Convert from normalized [cx, cy, w, h] to feature map [x1, y1, x2, y2]
                    boxes = self._cxcywh_to_xyxy(boxes)
                    boxes = boxes * (1024 * self.spatial_scale)
                    boxes_list.append(boxes)
                else:
                    boxes_list.append(None)
        else:
            boxes_list = [None] * batch_size

        return boxes_list

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _aggregate_roi_grids(
        self,
        roi_grids: torch.Tensor,
        boxes: torch.Tensor,
        grid_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Aggregate RoI-level grids to image-level grid."""
        N, C, H_roi, W_roi = roi_grids.shape

        # Initialize output grid and count map
        output_grid = torch.zeros(C, grid_size, grid_size, device=device, dtype=roi_grids.dtype)
        count_map = torch.zeros(1, grid_size, grid_size, device=device, dtype=roi_grids.dtype)

        # Assuming feature map size is 64×64 for stride-16 on 1024×1024 input
        feat_size = 1024 * self.spatial_scale

        for i in range(N):
            x1, y1, x2, y2 = boxes[i]

            # Map box coordinates to grid coordinates
            gx1 = int((x1 / feat_size) * grid_size)
            gy1 = int((y1 / feat_size) * grid_size)
            gx2 = int((x2 / feat_size) * grid_size)
            gy2 = int((y2 / feat_size) * grid_size)

            # Clamp to valid range
            gx1, gy1 = max(0, gx1), max(0, gy1)
            gx2, gy2 = min(grid_size, gx2), min(grid_size, gy2)

            if gx2 <= gx1 or gy2 <= gy1:
                continue

            # Resize RoI grid to match target region
            roi_grid_resized = F.interpolate(
                roi_grids[i:i+1],
                size=(gy2 - gy1, gx2 - gx1),
                mode='bilinear',
                align_corners=False
            )[0]  # (C, gy2-gy1, gx2-gx1)

            # Accumulate
            output_grid[:, gy1:gy2, gx1:gx2] += roi_grid_resized
            count_map[:, gy1:gy2, gx1:gx2] += 1

        # Average overlapping regions
        count_map = torch.clamp(count_map, min=1)
        output_grid = output_grid / count_map

        return output_grid

    def _fallback_prediction(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Fallback to dense prediction when no boxes are available.

        Args:
            feat_map: (B, C, H, W)

        Returns:
            Grid logits (B, num_classes, grid_size, grid_size)
        """
        x = self.fallback_head(feat_map)
        x = F.interpolate(x, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=False)
        return x


@register()
class HybridGridHead(nn.Module):
    """Hybrid Grid Classification Head"""
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 4,
        roi_size: int = 14,
        grid_size: int = 32,
        fusion_mode: str = 'learned',
        **kwargs
    ):
        super().__init__()

        # Handle list input for in_channels (multi-scale)
        if isinstance(in_channels, list):
            # Use the middle scale (stride-16) as default
            actual_in_channels = in_channels[1] if len(in_channels) > 1 else in_channels[0]
        else:
            actual_in_channels = in_channels

        # RoI-based head
        self.roi_head = RoIGridClassificationHead(
            in_channels=actual_in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            roi_size=roi_size,
            grid_size=grid_size,
            **kwargs
        )

        # Dense head (from your FrequencyAttentionGridHead)
        from .frequency_attention_grid_head import FrequencyAttentionGridHead
        self.dense_head = FrequencyAttentionGridHead(
            in_channels=in_channels,  # Pass original (can be list)
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_multi_scale=isinstance(in_channels, list),
        )

        # Fusion module
        self.fusion_mode = fusion_mode
        if fusion_mode == 'learned':
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(num_classes * 2, hidden_dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, 1, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        features: List[torch.Tensor],
        det_outputs: Optional[Dict] = None,
        targets: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: List of features
            det_outputs: Detection outputs
            targets: Ground truth targets

        Returns:
            Fused grid logits (B, num_classes, grid_size, grid_size)
        """
        # Get predictions from both heads
        roi_logits = self.roi_head(features, det_outputs, targets)
        dense_logits = self.dense_head(features)

        # Ensure both outputs have the same spatial size
        target_size = roi_logits.shape[-2:]  # (H, W)
        if dense_logits.shape[-2:] != target_size:
            dense_logits = F.interpolate(
                dense_logits,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

        # Fuse predictions
        if self.fusion_mode == 'weighted':
            # Simple weighted average (favor RoI where confident)
            roi_conf = roi_logits.sigmoid().max(dim=1, keepdim=True)[0]
            fused = roi_conf * roi_logits + (1 - roi_conf) * dense_logits
        elif self.fusion_mode == 'max':
            # Take maximum (most confident)
            fused = torch.maximum(roi_logits, dense_logits)
        elif self.fusion_mode == 'learned':
            # Learned fusion gate
            gate = self.fusion_gate(torch.cat([roi_logits, dense_logits], dim=1))
            fused = gate * roi_logits + (1 - gate) * dense_logits
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        return fused
