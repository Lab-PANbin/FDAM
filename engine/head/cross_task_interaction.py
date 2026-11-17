"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Cross-Task Interaction Module (CTIM) for Detection-Grid Bidirectional Feature Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from ..core import register


class CrossTaskAttention(nn.Module):
    """
    Cross-Task Attention for feature interaction between Detection and Grid.

    Uses cross-attention mechanism where one task's features serve as queries
    and the other task's features serve as keys/values.
    """
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query projection
        self.q_proj = nn.Linear(query_dim, query_dim)
        # Key and Value projections (from the other task)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)
        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.constant_(self.v_proj.bias, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C_q) query features
            key_value: (B, N_kv, C_kv) key/value features
            key_padding_mask: (B, N_kv) mask for key/value positions

        Returns:
            output: (B, N_q, C_q) attended features
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]

        # Project queries, keys, values
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N_q, N_kv)

        # Apply mask if provided
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N_q, C)
        out = self.out_proj(out)

        return out


@register()
class CrossTaskInteractionModule(nn.Module):
    """Cross-Task Interaction Module (CTIM)"""
    def __init__(
        self,
        det_dim: int = 256,
        grid_dim: int = 224,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_det_to_grid: bool = True,
        use_grid_to_det: bool = True,
    ):
        super().__init__()
        self.det_dim = det_dim
        self.grid_dim = grid_dim
        self.use_det_to_grid = use_det_to_grid
        self.use_grid_to_det = use_grid_to_det

        # Det → Grid: Grid queries attend to Detection features
        if use_det_to_grid:
            self.det_to_grid_attn = CrossTaskAttention(
                query_dim=grid_dim,
                kv_dim=det_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            # Gated fusion for Grid
            self.grid_gate = nn.Sequential(
                nn.Conv2d(grid_dim * 2, grid_dim // 4, 1, bias=False),
                nn.GroupNorm(8, grid_dim // 4),
                nn.SiLU(inplace=True),
                nn.Conv2d(grid_dim // 4, 1, 1, bias=True),
                nn.Sigmoid()
            )

        # Grid → Det: Detection queries attend to Grid features
        if use_grid_to_det:
            self.grid_to_det_attn = CrossTaskAttention(
                query_dim=det_dim,
                kv_dim=grid_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            # Gated fusion for Detection
            self.det_gate = nn.Sequential(
                nn.Linear(det_dim * 2, det_dim // 4),
                nn.SiLU(inplace=True),
                nn.Linear(det_dim // 4, 1),
                nn.Sigmoid()
            )

        # Layer norms for stability
        if use_det_to_grid:
            self.grid_norm = nn.LayerNorm(grid_dim)
        if use_grid_to_det:
            self.det_norm = nn.LayerNorm(det_dim)


    def forward(
        self,
        det_feat: torch.Tensor,
        grid_feat: torch.Tensor,
        det_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional feature interaction."""
        B, N, C_det = det_feat.shape
        _, C_grid, H, W = grid_feat.shape

        # Flatten grid features for attention: (B, C, H, W) -> (B, H*W, C)
        grid_flat = grid_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C_grid)

        det_out = det_feat
        grid_out = grid_feat


        if self.use_det_to_grid:
            # Grid positions attend to Detection queries
            grid_enhanced = self.det_to_grid_attn(
                query=grid_flat,      # (B, H*W, C_grid)
                key_value=det_feat,   # (B, N, C_det)
            )  # (B, H*W, C_grid)

            # Reshape back to spatial
            grid_enhanced = grid_enhanced.permute(0, 2, 1).reshape(B, C_grid, H, W)

            # Apply layer norm
            grid_enhanced = self.grid_norm(grid_enhanced.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            # Gated fusion
            gate = self.grid_gate(torch.cat([grid_feat, grid_enhanced], dim=1))
            grid_out = gate * grid_enhanced + (1 - gate) * grid_feat


        if self.use_grid_to_det:
            # Detection queries attend to Grid features
            det_enhanced = self.grid_to_det_attn(
                query=det_feat,    # (B, N, C_det)
                key_value=grid_flat,  # (B, H*W, C_grid)
            )  # (B, N, C_det)

            # Apply layer norm
            det_enhanced = self.det_norm(det_enhanced)

            # Gated fusion
            gate = self.det_gate(torch.cat([det_feat, det_enhanced], dim=-1))
            det_out = gate * det_enhanced + (1 - gate) * det_feat

        return det_out, grid_out

    def get_attention_weights(
        self,
        det_feat: torch.Tensor,
        grid_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for visualization.

        Returns:
            Dictionary with attention weight tensors
        """
        # For now, return empty dict
        return {}


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for Grid features.

    Adds learnable or sinusoidal positional information to grid features.
    """
    def __init__(self, channels: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, channels // 2)
        self.col_embed = nn.Embedding(max_w, channels // 2)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features

        Returns:
            (B, C, H, W) features with positional encoding added
        """
        B, C, H, W = x.shape
        device = x.device

        # Create position indices
        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)

        # Get embeddings
        row_emb = self.row_embed(rows)  # (H, C//2)
        col_emb = self.col_embed(cols)  # (W, C//2)

        # Combine: (H, W, C)
        pos = torch.cat([
            row_emb.unsqueeze(1).expand(-1, W, -1),
            col_emb.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1)

        # Add to input: (B, C, H, W)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        return x + pos
