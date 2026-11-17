"""
FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = [
        'backbone', 'encoder', 'decoder',
        'grid_head', 'cross_task_interaction', 'grid_enhancer',
    ]
    __inject_type__ = {
        'decoder': 'optional',
        'grid_head': 'optional',
        'cross_task_interaction': 'optional',
        'grid_enhancer': 'optional',
    }

    def __init__(self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        grid_head: nn.Module = None,
        cross_task_interaction: nn.Module = None,
        grid_enhancer: nn.Module = None,
        use_ctim: bool = False,
        use_decoupled_grid: bool = False,
        use_grid_guided_enhancement: bool = False,
        use_det_feedback: bool = False,
        det_input_dim: int = 256,
        use_box_grid_interaction: bool = False,
    ):
        super().__init__()
        if decoder is None:
            if grid_head is None:
                raise ValueError('DEIM requires a detection decoder or a grid head')
            if use_ctim or use_box_grid_interaction or use_grid_guided_enhancement or use_det_feedback:
                raise ValueError('Grid-only DEIM cannot use detection/grid interaction')
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.grid_head = grid_head
        self.cross_task_interaction = cross_task_interaction
        self.grid_enhancer = grid_enhancer
        self.use_ctim = use_ctim and (cross_task_interaction is not None)
        self.use_decoupled_grid = use_decoupled_grid
        self.use_box_grid_interaction = (use_box_grid_interaction and (grid_head is not None))
        if self.use_box_grid_interaction:
            print("[DEIM] Box-conditioned Grid → Box Query Interaction enabled")

        self.use_grid_guided_enhancement = (
            use_grid_guided_enhancement
            and (grid_enhancer is not None)
            and (grid_head is not None)
        )

        self.use_det_feedback = (
            use_det_feedback
            and (grid_head is not None)
            and self.use_grid_guided_enhancement
        )
        self.det_context_proj = None
        if self.use_det_feedback:
            grid_hidden = getattr(grid_head, 'hidden_dim', 128)
            self.det_context_proj = nn.Linear(det_input_dim, grid_hidden)
            nn.init.normal_(self.det_context_proj.weight, std=0.01)
            nn.init.zeros_(self.det_context_proj.bias)

        if self.use_ctim:
            print("[DEIM] Cross-Task Interaction Module (CTIM) enabled")
        if self.use_decoupled_grid:
            print("[DEIM] Decoupled Grid Branch enabled - Grid uses backbone features")
        if self.use_grid_guided_enhancement:
            print(f"[DEIM]  Grid-Guided Enhancement")
        if self.use_det_feedback:
            grid_hidden = getattr(grid_head, 'hidden_dim', 128)
            print(f"[DEIM] Detection→Grid feedback enabled: "
                  f"det_input_dim={det_input_dim} → grid_hidden={grid_hidden}")

    def forward(self, x, targets=None):
        backbone_feats = self.backbone(x)
        if isinstance(backbone_feats, tuple):
            backbone_feats = list(backbone_feats)

        encoder_feats = self.encoder(backbone_feats)

        if self.decoder is None:
            grid_feats = backbone_feats if self.use_decoupled_grid else encoder_feats
            return {'pred_grid_logits': self.grid_head(grid_feats)}

        if self.use_grid_guided_enhancement:
            return self._forward_with_grid_enhancement(
                backbone_feats, encoder_feats, targets
            )
        else:
            return self._forward_original(
                backbone_feats, encoder_feats, targets
            )

    def _forward_with_grid_enhancement(self, backbone_feats, encoder_feats, targets):
        """Refine encoder features with grid confidence before decoding boxes."""

        grid_input_feats = backbone_feats if self.use_decoupled_grid else encoder_feats

        grid_logits = self.grid_head(grid_input_feats)

        scale_logits = None
        if hasattr(self.grid_head, 'get_scale_logits'):
            scale_logits = self.grid_head.get_scale_logits()

        if scale_logits is not None:
            grid_conf_maps = self._normalize_to_conf_map_list(scale_logits, encoder_feats)
        else:
            grid_conf_maps = self._normalize_to_conf_map_list(grid_logits, encoder_feats)

        enhanced_encoder_feats = self.grid_enhancer(encoder_feats, grid_conf_maps)

        outputs = self.decoder(encoder_feats, targets, value_feats=enhanced_encoder_feats)

        final_grid_logits = grid_logits

        if (self.use_det_feedback
                and self.det_context_proj is not None
                and hasattr(self.grid_head, 'forward_with_detection_context')):

            det_context = self._extract_detection_context(outputs)

            if det_context is not None:
                grid_logits_v2 = self.grid_head.forward_with_detection_context(
                    det_context
                )
                final_grid_logits = grid_logits_v2

        if self.training:
            outputs['pred_grid_logits'] = final_grid_logits
            if self.use_det_feedback and final_grid_logits is not grid_logits:
                outputs['pred_grid_logits_aux'] = grid_logits
        else:
            if not isinstance(outputs, dict):
                outputs = {'pred_logits': outputs}
            outputs['pred_grid_logits'] = final_grid_logits

        return outputs

    def _extract_detection_context(
        self, outputs: dict
    ) -> Optional[torch.Tensor]:
        """Pool decoder context with detached confidence weights, then project channels."""
        if not isinstance(outputs, dict):
            return None

        context_raw = None

        hs = outputs.get('hs', None)
        if hs is not None:
            if hs.dim() == 4:
                hs = hs[-1]

            pred_logits = outputs.get('pred_logits', None)
            if pred_logits is not None:
                scores = pred_logits.detach().sigmoid().max(dim=-1).values  # (B, N_q)
                weights = F.softmax(scores * 5.0, dim=-1)                  # (B, N_q)
                context_raw = (weights.unsqueeze(-1) * hs).sum(dim=1)      # (B, C_dec)
            else:
                context_raw = hs.mean(dim=1)                               # (B, C_dec)

        if context_raw is None:
            pred_logits = outputs.get('pred_logits', None)
            if pred_logits is None:
                return None
            # pred_logits: (B, num_queries, num_classes)
            scores = pred_logits.detach().sigmoid().max(dim=-1).values  # (B, N_q)
            weights = F.softmax(scores * 5.0, dim=-1)                  # (B, N_q)
            context_raw = (weights.unsqueeze(-1) * pred_logits).sum(dim=1)  # (B, C_cls)

        return self.det_context_proj(context_raw)  # (B, grid_hidden_dim)

    def _normalize_to_conf_map_list(self, grid_outputs, encoder_feats) -> list:
        """Return one sigmoid confidence map per encoder scale."""

        num_scales = len(encoder_feats)
        if isinstance(grid_outputs, (list, tuple)):
            conf_maps = [g.sigmoid() for g in grid_outputs]

        elif isinstance(grid_outputs, dict):
            logits = grid_outputs.get(
            'grid_logits',
            grid_outputs.get('pred_grid_logits')
        )
            if logits is None:
                raise KeyError(
                "grid_outputs dict must contain 'grid_logits' or 'pred_grid_logits'"
                )

            if isinstance(logits, (list, tuple)):
                conf_maps = [g.sigmoid() for g in logits]
            else:
                conf = logits.sigmoid()
                conf_maps = [conf] * num_scales

        else:
            conf = grid_outputs.sigmoid()
            conf_maps = [conf] * num_scales

        if len(conf_maps) < num_scales:
            conf_maps = conf_maps + [conf_maps[-1]] * (num_scales - len(conf_maps))
        elif len(conf_maps) > num_scales:
            conf_maps = conf_maps[:num_scales]

        return conf_maps

    def _forward_original(self, backbone_feats, encoder_feats, targets):

        # Case 1: Grid → Box interaction enabled

        if self.grid_head is not None and self.use_box_grid_interaction:
            grid_input_feats = backbone_feats if self.use_decoupled_grid else encoder_feats

            is_roi_based = hasattr(self.grid_head, 'roi_align')
            if is_roi_based:
                # ROI-based grid head depends on detection outputs, so it cannot
                # provide grid feature before decoder.
                outputs = self.decoder(encoder_feats, targets)
                grid_outputs = self.grid_head(
                    grid_input_feats,
                    det_outputs=outputs if isinstance(outputs, dict) else None,
                    targets=targets if self.training else None,
                    )

                if self.training:
                    outputs['pred_grid_logits'] = grid_outputs
                else:
                    if not isinstance(outputs, dict):
                        outputs = {'pred_logits': outputs}
                    outputs['pred_grid_logits'] = grid_outputs

                return outputs


            grid_outputs = self.grid_head(grid_input_feats)


            grid_feat = None
            if hasattr(self.grid_head, 'get_pre_cls_features'):
                grid_feat = self.grid_head.get_pre_cls_features()
            elif hasattr(self.grid_head, '_last_grid_context_features'):
                grid_feat = self.grid_head._last_grid_context_features
            elif hasattr(self.grid_head, '_last_pre_cls_features'):
                grid_feat = self.grid_head._last_pre_cls_features

            # For FrequencyAttentionGridHead, grid_outputs is Tensor[B, 1, H, W].
            # Use it as box-internal confidence map.
            grid_logits_for_box = grid_outputs

            outputs = self.decoder(encoder_feats, targets, grid_feat=grid_feat,grid_logits=grid_logits_for_box)


            if self.training:
                outputs['pred_grid_logits'] = grid_outputs
            else:
                if not isinstance(outputs, dict):
                    outputs = {'pred_logits': outputs}
                outputs['pred_grid_logits'] = grid_outputs

            return outputs


        # Case 2: Original behavior

        outputs = self.decoder(encoder_feats, targets)

        if self.grid_head is not None:
            if self.use_decoupled_grid:
                grid_input_feats = backbone_feats
            else:
                grid_input_feats = encoder_feats

            is_roi_based = hasattr(self.grid_head, 'roi_align')

            if is_roi_based:
                det_outputs = outputs if isinstance(outputs, dict) else None
                grid_outputs = self.grid_head(
                    grid_input_feats,
                    det_outputs=det_outputs,
                    targets=targets if self.training else None
                )
                grid_features = None
            else:
                grid_outputs = self.grid_head(grid_input_feats)
                grid_features = None

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
        if not hasattr(self.decoder, '_proj_feats'):
            return grid_outputs, det_outputs

        det_feat_map = self.decoder._proj_feats[1]
        B, C_det, H_det, W_det = det_feat_map.shape
        det_feat = det_feat_map.flatten(2).permute(0, 2, 1)

        _, C_grid, H_g, W_g = grid_outputs.shape
        grid_feat_map = encoder_feats[1]

        if grid_feat_map.shape[1] != self.cross_task_interaction.grid_dim:
            grid_feat_map = F.interpolate(
                grid_outputs.detach(),
                size=(H_det, W_det),
                mode='bilinear',
                align_corners=False
            )
            return grid_outputs, det_outputs

        det_enhanced, grid_enhanced = self.cross_task_interaction(
            det_feat=det_feat,
            grid_feat=grid_feat_map,
        )

        det_outputs['ctim_det_feat'] = det_enhanced
        det_outputs['ctim_grid_feat'] = grid_enhanced

        return grid_outputs, det_outputs

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
