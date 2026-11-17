"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Grid classification with spectral and directional feature enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from ..core import register

from .direction_aware_grid_head import (
    get_num_groups,
    MultiScaleFusion,
    DirectionAwareBlock,
    AdaptiveDirectionFusion,
)

class DetectionContextModulator(nn.Module):
    """Apply zero-initialized FiLM modulation from projected detection context."""
    def __init__(self, hidden_dim: int):
        super().__init__()


        self.film_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )


        self.scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):

        for i, m in enumerate(self.film_net):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last_linear = self.film_net[-1]
        nn.init.normal_(last_linear.weight, std=0.01)
        nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        grid_feat: torch.Tensor,    # (B, C, H, W)
        det_context: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the module to the input tensors."""
        B, C, H, W = grid_feat.shape

        film_params = self.film_net(det_context)           # (B, 2C)
        gamma, beta = film_params.chunk(2, dim=-1)         # (B, C), (B, C)


        gamma = gamma.tanh() * self.scale.abs()
        beta  = beta.tanh()  * self.scale.abs()

        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)

        return (1 + gamma) * grid_feat + beta

class FrequencyDynamicAttention(nn.Module):
    """Frequency-Dynamic Attention (FDA) Module"""
    def __init__(
        self,
        channels: int,
        num_bands: int = 4,
        use_direction_mask: bool = True,
        reduction: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands
        self.use_direction_mask = use_direction_mask

        # Band attention: learn importance of each frequency band
        reduced_ch = max(channels // reduction, 16)
        self.band_attention = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(channels, reduced_ch, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_ch, num_bands, 1, bias=True),
        )

        # Direction-aware frequency weights [horizontal, vertical]
        if use_direction_mask:
            self.direction_weights = nn.Parameter(torch.ones(2) * 0.5)

        # Spatial-frequency fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, reduced_ch, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # Learnable scale for frequency enhancement
        self.freq_scale = nn.Parameter(torch.ones(1) * 0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial features (B, C, H, W)
        Returns:
            Enhanced features (B, C, H, W)

        Uses hand-rolled DFT via cos/sin matrices — zero complex tensors,
        fully ONNX-compatible with ANY export method (jit.trace or Dynamo).
        """
        B, C, H, W = x.shape
        orig_dtype = x.dtype
        x = x.float()

        # ---- 2D Real FFT (hand-rolled) ----
        real_part, imag_part = self._rfft2_manual(x)  # both (B, C, H, W//2+1)
        freq_w = real_part.shape[-1]

        # Amplitude (all real-valued)
        amplitude = torch.sqrt(real_part ** 2 + imag_part ** 2 + 1e-8)

        # Band attention weights from spatial features
        band_weights = self.band_attention(x.mean(dim=(2, 3), keepdim=True))  # (B, num_bands, 1, 1)
        band_weights = F.softmax(band_weights, dim=1)

        # Frequency band masks
        band_masks = self._create_band_masks(H, freq_w, x.device)

        # Weighted mask: (B, 1, H, freq_w)
        weighted_mask = (band_weights.unsqueeze(-1) * band_masks.unsqueeze(0).unsqueeze(2)).sum(dim=1)

        # Direction-aware mask (optional)
        if self.use_direction_mask:
            direction_mask = self._create_direction_mask(H, freq_w, x.device)
            weighted_mask = weighted_mask * (1 + direction_mask.unsqueeze(0).unsqueeze(0))

        # Enhancement factor: (B, 1, H, freq_w)
        enhancement = 1 + self.freq_scale * weighted_mask

        # Scale real and imag parts (NO complex multiplication!)
        enhanced_real = enhancement * real_part
        enhanced_imag = enhancement * imag_part

        # ---- 2D Inverse Real FFT (hand-rolled) ----
        freq_enhanced = self._irfft2_manual(enhanced_real, enhanced_imag, H, W)
        freq_enhanced = freq_enhanced.to(orig_dtype)

        # Gated fusion
        gate = self.fusion_gate(torch.cat([x.to(orig_dtype), freq_enhanced], dim=1))
        out = gate * freq_enhanced + (1 - gate) * x.to(orig_dtype)

        return out


    #  Pure-real DFT helpers (ONNX-friendly, zero complex tensors)


    @staticmethod
    def _dft_matrices(N: int, device: torch.device):
        """
        Return (cos_N, sin_N) matrices for 1D DFT of length N.
          DFT[k, n] = cos(2π·n·k/N) - i·sin(2π·n·k/N)
        with  n ∈ [0, N),  k ∈ [0, N).
        Shape: (N_rows, N_cols) — rows=dst_freq, cols=src_spatial.
        """
        n = torch.arange(N, device=device, dtype=torch.float32)  # (N,)
        k = n.unsqueeze(1)                                       # (N, 1)
        theta = 2.0 * torch.pi * n * k / N                       # (N, N)
        return torch.cos(theta), torch.sin(theta)

    @staticmethod
    def _rfft_matrices(N: int, device: torch.device):
        """
        RFFT matrices for real input of length N.
        Returns (cos, sin) of shape (N//2+1, N).
          real[k] =  Σ x[n]· cos(2π·n·k/N)
          imag[k] = -Σ x[n]· sin(2π·n·k/N)
        """
        K = N // 2 + 1
        n = torch.arange(N, device=device, dtype=torch.float32)  # (N,)
        k = torch.arange(K, device=device, dtype=torch.float32).unsqueeze(1)  # (K,1)
        theta = 2.0 * torch.pi * n * k / N                       # (K, N)
        return torch.cos(theta), torch.sin(theta)

    def _rfft2_manual(self, x: torch.Tensor):
        """
        Orthonormal 2D RFFT using only matmul + cos/sin.
        x: (B, C, H, W) real
        Returns (real, imag) each (B, C, H, W//2+1).
        """
        B, C, H, W = x.shape
        device = x.device
        freq_w = W // 2 + 1

        # ---- Step 1: 1D RFFT along last axis (W) ----
        cos_w, sin_w = self._rfft_matrices(W, device)  # (freq_w, W)

        x_flat = x.reshape(B * C * H, W)               # (B*C*H, W)
        real_w = x_flat @ cos_w.T                       # (B*C*H, freq_w)
        imag_w = x_flat @ (-sin_w.T)                    # (B*C*H, freq_w)

        # ---- Step 2: 1D CFFT along H axis ----
        # Input: complex (real_w, imag_w) of shape (B*C*H, freq_w)
        # Reshape to (B*C, H, freq_w)
        real_w = real_w.reshape(B * C, H, freq_w)
        imag_w = imag_w.reshape(B * C, H, freq_w)
        # Transpose so H is last: (B*C, freq_w, H)
        real_w = real_w.permute(0, 2, 1)    # (B*C, freq_w, H)
        imag_w = imag_w.permute(0, 2, 1)

        cos_h, sin_h = self._dft_matrices(H, device)  # (H, H)


        # Actually let me just treat columns independently.
        # For each column (freq_w dim), we do a 1D FFT along H.
        # Input shape for matmul: (B*C*freq_w, H) @ (H, H) → (B*C*freq_w, H)
        real_flat = real_w.reshape(B * C * freq_w, H)  # (B*C*freq_w, H)
        imag_flat = imag_w.reshape(B * C * freq_w, H)

        # Y_real = real * cos - imag * (-sin)?? No.

        # DFT: for each output k:
        #   Y[k] = Σ_n X[n] * exp(-i * 2π * n * k / H)
        #        = Σ_n X[n] * (cos(θ) - i·sin(θ))
        #   where θ = 2π·n·k/H, cos_h[k,n] = cos(θ), sin_h[k,n] = sin(θ)
        #
        #   Y_real[k] = Σ_n (real[n]·cos(θ) + imag[n]·sin(θ))
        #   Y_imag[k] = Σ_n (imag[n]·cos(θ) - real[n]·sin(θ))
        #
        # In matrix form: cos_h[k,n], sin_h[k,n]
        #   Y_real = real_flat @ cos_h.T + imag_flat @ sin_h.T
        #   Y_imag = imag_flat @ cos_h.T - real_flat @ sin_h.T

        Y_real = real_flat @ cos_h.T + imag_flat @ sin_h.T    # (B*C*freq_w, H)
        Y_imag = imag_flat @ cos_h.T - real_flat @ sin_h.T

        # Reshape back: (B*C, freq_w, H) → (B, C, H, freq_w)
        Y_real = Y_real.reshape(B * C, freq_w, H).permute(0, 2, 1).reshape(B, C, H, freq_w)
        Y_imag = Y_imag.reshape(B * C, freq_w, H).permute(0, 2, 1).reshape(B, C, H, freq_w)

        # Match torch.fft.rfft2(..., norm='ortho') on both spatial axes.
        scale = (H * W) ** -0.5
        return Y_real * scale, Y_imag * scale

    def _irfft2_manual(self, real_part: torch.Tensor, imag_part: torch.Tensor,
                        H: int, W: int):
        """
        Orthonormal 2D inverse RFFT using only matmul + cos/sin.
        real_part, imag_part: (B, C, H, W//2+1)
        Returns: (B, C, H, W) real
        """
        B, C, _, freq_w = real_part.shape
        device = real_part.device

        # ---- Step 1: 1D ICFFT along H ----
        # IFFT uses +i instead of -i: exp(+i*2π*n*k/H)
        # Y_real[k] = Σ_n (real[n]·cos(θ) - imag[n]·sin(θ))
        # Y_imag[k] = Σ_n (real[n]·sin(θ) + imag[n]·cos(θ))
        cos_h, sin_h = self._dft_matrices(H, device)  # (H, H)

        # (B, C, H, freq_w) → (B*C, H, freq_w) → (B*C, freq_w, H)
        r = real_part.reshape(B * C, H, freq_w).permute(0, 2, 1)  # (B*C, freq_w, H)
        i = imag_part.reshape(B * C, H, freq_w).permute(0, 2, 1)

        # Flatten for matmul
        r_flat = r.reshape(B * C * freq_w, H)  # (B*C*freq_w, H)
        i_flat = i.reshape(B * C * freq_w, H)

        Y_real = r_flat @ cos_h.T - i_flat @ sin_h.T    # IFFT: sign flip on sin
        Y_imag = r_flat @ sin_h.T + i_flat @ cos_h.T

        Y_real = Y_real.reshape(B * C, freq_w, H).permute(0, 2, 1).reshape(B, C, H, freq_w)
        Y_imag = Y_imag.reshape(B * C, freq_w, H).permute(0, 2, 1).reshape(B, C, H, freq_w)

        # ---- Step 2: 1D IRFFT along W (frequency → spatial) ----
        # IRFFT of length freq_w (=W//2+1) back to length W
        # For real output, we use: x[n] = (1/W) * Σ_k X[k] * exp(+i*2π*k*n/W)
        # But need correct scaling + conjugate symmetry

        # IRFFT matrix: (W, freq_w)
        # x[n] = Σ_k (real[k]*cos(θ) - imag[k]*sin(θ))  for k=0..freq_w-1
        # with θ = 2π*k*n/W
        cos_w, sin_w = self._rfft_matrices(W, device)  # (freq_w, W)

        # We need cos_w[k,n] = cos(2π*k*n/W), shape (freq_w, W)
        # Y_irfft[n] = Σ_k (Y_real[k]*cos_w[k,n] - Y_imag[k]*sin_w[k,n])
        # Flatten: (B*C*H, freq_w) @ (freq_w, W)
        Yr_flat = Y_real.reshape(B * C * H, freq_w)
        Yi_flat = Y_imag.reshape(B * C * H, freq_w)

        # Restore the omitted conjugate frequencies. DC is counted once; the
        # Nyquist term is also counted once only when W is even. For odd W,
        # the last stored frequency still has a conjugate partner.
        k = torch.arange(freq_w, device=device)
        endpoint = (k == 0) | ((W % 2 == 0) & (k == W // 2))
        multiplicity = torch.where(endpoint, 1.0, 2.0).to(real_part.dtype)
        x_out = (Yr_flat * multiplicity) @ cos_w - (Yi_flat * multiplicity) @ sin_w

        # Normalize both axes, matching torch.fft.irfft2(..., norm='ortho').
        x_out = x_out * ((H * W) ** -0.5)

        return x_out.reshape(B, C, H, W)

    def _create_band_masks(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create octave-based frequency band masks."""
        # Create frequency coordinate grids
        # For rfft2, horizontal frequencies are [0, 1, ..., W-1] / (2*W-2)
        # Vertical frequencies are shifted: [0, 1, ..., H//2, -H//2+1, ..., -1] / H
        freq_h = torch.arange(H, device=device, dtype=torch.float32)
        freq_h = torch.where(freq_h <= H // 2, freq_h, freq_h - H) / float(H)
        freq_w = torch.arange(W, device=device, dtype=torch.float32) / float(max(W * 2 - 1, 1))

        # Compute distance from DC (center)
        dist_h = freq_h.unsqueeze(1)  # (H, 1)
        dist_w = freq_w.unsqueeze(0)  # (1, W)
        dist = torch.sqrt(dist_h ** 2 + dist_w ** 2)  # (H, W)

        # Normalize distance to [0, 1]
        max_dist = dist.max() + 1e-8
        dist_norm = dist / max_dist

        # Create band edges (octave-based: exponential spacing)
        # This gives more resolution to low frequencies
        band_edges = torch.linspace(0, 1, self.num_bands + 1, device=device)

        # Create masks for each band
        masks = []
        for i in range(self.num_bands):
            mask = ((dist_norm >= band_edges[i]) & (dist_norm < band_edges[i + 1])).float()
            masks.append(mask)

        return torch.stack(masks, dim=0)  # (num_bands, H, W)

    def _create_direction_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create direction-aware frequency mask."""
        freq_h = torch.arange(H, device=device, dtype=torch.float32)
        freq_h = torch.where(freq_h <= H // 2, freq_h, freq_h - H).abs() / float(H)
        freq_w = torch.arange(W, device=device, dtype=torch.float32).abs() / float(max(W * 2 - 1, 1))

        # Normalize direction weights
        weights = F.softmax(self.direction_weights, dim=0)

        # Horizontal frequency emphasis (for TC - transverse cracks)
        h_emphasis = freq_h.unsqueeze(1).expand(H, W)
        # Vertical frequency emphasis (for LC - longitudinal cracks)
        v_emphasis = freq_w.unsqueeze(0).expand(H, W)

        # Weighted combination
        direction_mask = weights[0] * h_emphasis + weights[1] * v_emphasis

        # Normalize to [0, 1]
        direction_mask = direction_mask / (direction_mask.max() + 1e-8)

        return direction_mask

    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention maps for visualization."""
        B, C, H, W = x.shape

        # Compute frequency spectrum
        freq = torch.fft.rfft2(x, norm='ortho')
        amplitude = torch.abs(freq).mean(dim=1)  # (B, H, W//2+1)

        # Get band attention weights
        band_weights = self.band_attention(x.mean(dim=(2, 3), keepdim=True))
        band_weights = F.softmax(band_weights, dim=1).squeeze(-1).squeeze(-1)  # (B, num_bands)

        result = {
            'amplitude_spectrum': amplitude,
            'band_weights': band_weights,
        }

        if self.use_direction_mask:
            result['direction_weights'] = F.softmax(self.direction_weights, dim=0)

        return result


@register()
class FrequencyAttentionGridHead(nn.Module):
    """Frequency-Attention Grid Classification Head"""
    def __init__(
        self,
        in_channels: int = 224,
        hidden_dim: int = 128,
        num_classes: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
        use_adf: bool = True,
        use_multi_scale: bool = True,
        use_frequency_attention: bool = True,
        freq_num_bands: int = 4,
        freq_use_direction: bool = True,
        prior_probs: Optional[List[float]] = None,
        use_multi_scale_output: bool = False,
        use_det_feedback: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_adf = use_adf
        self.use_multi_scale = use_multi_scale
        self.use_frequency_attention = use_frequency_attention
        self.prior_probs = prior_probs
        self.hidden_dim = hidden_dim
        self.use_multi_scale_output = use_multi_scale_output
        self.use_det_feedback = use_det_feedback

        self._last_scale_logits: Optional[List[torch.Tensor]] = None
        self._last_pre_cls_features: Optional[torch.Tensor] = None

        self._last_grid_context_features: Optional[torch.Tensor] = None


        if isinstance(in_channels, int):
            in_channels_list = [in_channels] * 3
        else:
            in_channels_list = list(in_channels)
        self._in_channels_list = in_channels_list

        # Multi-scale fusion
        if use_multi_scale:
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

        if use_multi_scale_output and use_multi_scale:
            scale_mid = max(hidden_dim // 2, 32)
            self.scale_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, scale_mid, 1, bias=False),
                    nn.GroupNorm(get_num_groups(scale_mid), scale_mid),
                    nn.SiLU(inplace=True),
                )
                for ch in in_channels_list
            ])
            self.scale_cls_heads = nn.ModuleList([
                nn.Conv2d(scale_mid, num_classes, 1, bias=True)
                for _ in range(len(in_channels_list))
            ])
            print(f"[FrequencyAttentionGridHead] Per-scale output enabled: "
                  f"{len(in_channels_list)} scales, "
                  f"scale_mid={scale_mid}, num_classes={num_classes}")
        else:
            self.scale_projs = None
            self.scale_cls_heads = None

        if use_det_feedback:
            self.det_context_modulator = DetectionContextModulator(hidden_dim)
            print(f"[FrequencyAttentionGridHead] Detection→Grid feedback enabled, "
                  f"hidden_dim={hidden_dim}")
        else:
            self.det_context_modulator = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with careful initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Classification outputs need a small initialization. Applying fan_out
        # Kaiming to a one-class 1x1 conv gives std=sqrt(2), which can saturate
        # the loss's [-20, 20] clamp and leave confidently wrong cells no gradient.
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
        if self.scale_cls_heads is not None:
            for head in self.scale_cls_heads:
                nn.init.normal_(head.weight, mean=0.0, std=0.01)

        # Initialize classification head bias with prior probabilities
        if hasattr(self.cls_head, 'bias') and self.cls_head.bias is not None:
            if self.prior_probs is not None:
                prior_probs = torch.tensor(self.prior_probs, dtype=torch.float32)
                assert len(prior_probs) == self.num_classes, \
                    f"prior_probs length ({len(prior_probs)}) must match num_classes ({self.num_classes})"
                eps = 1e-8
                prior_probs = torch.clamp(prior_probs, eps, 1 - eps)
                bias_init = -torch.log((1 - prior_probs) / prior_probs)
                with torch.no_grad():
                    self.cls_head.bias.copy_(bias_init)
                print(f"[FrequencyAttentionGridHead] Main head bias initialized "
                      f"with prior_probs: {self.prior_probs}")
            else:
                default_prior = 0.01
                bias_init = -torch.log(torch.tensor((1 - default_prior) / default_prior))
                nn.init.constant_(self.cls_head.bias, bias_init)

        if self.scale_cls_heads is not None:
            for head in self.scale_cls_heads:
                if self.prior_probs is not None:
                    prior_probs = torch.tensor(self.prior_probs, dtype=torch.float32)
                    eps = 1e-8
                    prior_probs = torch.clamp(prior_probs, eps, 1 - eps)
                    bias_init = -torch.log((1 - prior_probs) / prior_probs)
                    with torch.no_grad():
                        head.bias.copy_(bias_init)
                else:
                    default_prior = 0.01
                    bias_init = -torch.log(torch.tensor((1 - default_prior) / default_prior))
                    nn.init.constant_(head.bias, bias_init)

    def get_pre_cls_features(self) -> Optional[torch.Tensor]:
        """
        Return grid semantic feature for Grid → Box interaction.
        Prefer dropout-free feature because detection refinement should not depend
        on stochastic dropout noise.
        """
        if self._last_grid_context_features is not None:
            return self._last_grid_context_features
        return self._last_pre_cls_features

    def forward(self, features: List[torch.Tensor],
                det_context: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of encoder features [stride-8, stride-16, stride-32]

        Returns:
            Grid classification logits (B, num_classes, H, W)
        """
        # Multi-scale fusion or single feature projection
        if (self.use_multi_scale_output
                and self.scale_projs is not None
                and self.scale_cls_heads is not None
                and isinstance(features, (list, tuple))):
            scale_logits = []
            for feat, proj, head in zip(features, self.scale_projs, self.scale_cls_heads):
                scale_logits.append(head(proj(feat)))
            self._last_scale_logits = scale_logits
        else:
            self._last_scale_logits = None

        if self.use_multi_scale and self.multi_scale_fusion is not None:
            x = self.multi_scale_fusion(features)
        else:
            x = features[-1]
            x = self.input_proj(x)

        # Frequency-Dynamic Attention (optional)
        if self.use_frequency_attention:
            x = self.freq_attention(x)

        # Direction-Aware processing
        x = self.blocks(x)

        # Adaptive Direction Fusion
        x = self.adf(x)

        # Cache clean grid context feature for Grid → Box interaction.
        # This is before dropout, more stable for detection query refinement.
        self._last_grid_context_features = x

        # Classification
        x = self.dropout(x)
        self._last_pre_cls_features = x
        if det_context is not None and self.det_context_modulator is not None:
            x = self.det_context_modulator(x, det_context)

        return self.cls_head(x)

    def forward_with_detection_context(
        self, det_context: torch.Tensor
    ) -> torch.Tensor:
        """Recompute grid logits from cached features and detection context."""
        if self._last_pre_cls_features is None:
            raise RuntimeError(
                "forward_with_detection_context() 必须在 forward() 之后调用"
            )
        if self.det_context_modulator is None:
            raise RuntimeError(
                "use_det_feedback=False，无法使用检测上下文调制"
            )

        x = self.det_context_modulator(self._last_pre_cls_features, det_context)
        return self.cls_head(x)

    def get_scale_logits(self) -> Optional[List[torch.Tensor]]:
        return self._last_scale_logits

    def get_frequency_attention_maps(self, features: List[torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get frequency attention maps for visualization.

        Args:
            features: List of encoder features

        Returns:
            Dictionary of attention maps, or None if frequency attention is disabled
        """
        if not self.use_frequency_attention:
            return None

        # Get fused features
        if self.use_multi_scale and self.multi_scale_fusion is not None:
            x = self.multi_scale_fusion(features)
        else:
            x = features[-1]
            x = self.input_proj(x)

        return self.freq_attention.get_attention_maps(x)

    def get_direction_attention(self, features: List[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get direction attention maps from DSFD module.

        Args:
            features: List of encoder features

        Returns:
            Tuple of (h_attention, v_attention) or (None, None)
        """
        # Get fused features
        if self.use_multi_scale and self.multi_scale_fusion is not None:
            x = self.multi_scale_fusion(features)
        else:
            x = features[-1]
            x = self.input_proj(x)

        # Apply frequency attention if enabled
        if self.use_frequency_attention:
            x = self.freq_attention(x)

        # Get direction attention from first block
        if hasattr(self.blocks[0], 'dsfd') and hasattr(self.blocks[0].dsfd, 'direction_gate'):
            gate = self.blocks[0].dsfd.direction_gate(x)
            return gate[:, 0], gate[:, 1]

        return None, None
