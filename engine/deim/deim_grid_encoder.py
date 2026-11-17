"""
FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""
import torch.nn as nn
import torch.nn.functional as F
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'grid_head', 'cross_task_interaction']
    __inject_type__ = {'grid_head': 'optional', 'cross_task_interaction': 'optional'}

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        grid_head: nn.Module = None,
        cross_task_interaction: nn.Module = None,
        use_grid_interaction: bool = False,
        use_ctim: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.grid_head = grid_head
        self.cross_task_interaction = cross_task_interaction
        self.use_grid_interaction = use_grid_interaction
        self.use_ctim = use_ctim and (cross_task_interaction is not None)

        # Print CTIM status
        if self.use_ctim:
            print("[DEIM] Cross-Task Interaction Module (CTIM) enabled")

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)

        # Check if grid head supports interaction
        grid_attn_bias = None
        is_interactive = (self.grid_head is not None and
                         self.use_grid_interaction and
                         hasattr(self.grid_head, 'use_interaction') and
                         self.grid_head.use_interaction)

        if is_interactive:
            # First pass: get initial grid predictions and attention bias (no det feedback yet)
            grid_out = self.grid_head(x, det_features=None)
            if isinstance(grid_out, dict):
                grid_attn_bias = grid_out.get('grid_attn_bias', None)

        # Run decoder (with grid attention bias if available)
        outputs = self.decoder(x, targets, grid_attn_bias=grid_attn_bias)

        if self.grid_head is not None:
            if is_interactive:
                # Get detection features for feedback
                det_features = None
                if hasattr(self.decoder, 'get_intermediate_features'):
                    det_features = self.decoder.get_intermediate_features()

                # Second pass: refine grid predictions with detection feedback
                grid_out = self.grid_head(x, det_features=det_features)
                if isinstance(grid_out, dict):
                    grid_outputs = grid_out['grid_logits']
                    grid_features = grid_out.get('grid_features')
                else:
                    grid_outputs = grid_out
                    grid_features = None
            else:
                # Non-interactive mode: single forward pass
                grid_outputs = self.grid_head(x)
                grid_features = None

            # ============ Cross-Task Interaction Module (CTIM) ============
            if self.use_ctim and self.training:
                grid_outputs, outputs = self._apply_ctim(x, grid_outputs, outputs)

            # Add grid outputs to decoder outputs
            if self.training:
                outputs['pred_grid_logits'] = grid_outputs
                if grid_features is not None:
                    outputs['grid_features'] = grid_features
            else:
                if not isinstance(outputs, dict):
                    outputs = {'pred_logits': outputs}
                outputs['pred_grid_logits'] = grid_outputs

        return outputs

    def _apply_ctim(self, encoder_feats, grid_outputs, det_outputs):
        """Apply Cross-Task Interaction Module."""
        # Get detection features from decoder's projected features
        if not hasattr(self.decoder, '_proj_feats'):
            return grid_outputs, det_outputs

        # Use stride-16 feature as detection representation
        det_feat_map = self.decoder._proj_feats[1]  # (B, C_det, H, W)
        B, C_det, H_det, W_det = det_feat_map.shape

        # Flatten to sequence: (B, H*W, C)
        det_feat = det_feat_map.flatten(2).permute(0, 2, 1)  # (B, H*W, C_det)

        # Prepare grid features
        # Resize grid outputs to match detection feature spatial size
        _, C_grid, H_g, W_g = grid_outputs.shape

        # We need intermediate grid features, not logits
        # For now, use encoder features at similar scale
        grid_feat_map = encoder_feats[1]  # stride-16 feature
        if grid_feat_map.shape[1] != self.cross_task_interaction.grid_dim:
            # Project to grid_dim if needed
            grid_feat_map = F.interpolate(
                grid_outputs.detach(),  # Use grid logits as proxy
                size=(H_det, W_det),
                mode='bilinear',
                align_corners=False
            )
            # This is a simplified version - ideally we'd have proper grid features
            return grid_outputs, det_outputs

        # Apply CTIM
        det_enhanced, grid_enhanced = self.cross_task_interaction(
            det_feat=det_feat,
            grid_feat=grid_feat_map,
        )

        # Store enhanced features for potential use in loss
        det_outputs['ctim_det_feat'] = det_enhanced
        det_outputs['ctim_grid_feat'] = grid_enhanced

        return grid_outputs, det_outputs

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
