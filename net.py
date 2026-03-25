"""
net.py — SM-HAD Network Architecture

Implements the three core modules of Spectrum Mamba for Hyperspectral Anomaly Detection:

    EinFFT      : Ortho Spectrum Fourier Block (OSFB)
    Attention   : Masked Vanilla Attention Block (MVAB)
    SMHADBlock  : Hybrid Transformer Block (OSFB + MVAB + RMB + MLP)
    Net         : Full encoder-decoder network

Reference:
    Rachamalla et al., "SM-HAD: Spectrum Mamba for Hyperspectral Anomaly Detection",
    IEEE Transactions on Geoscience and Remote Sensing, 2026.
    DOI: 10.1109/TGRS.2026.3676658
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# ============================================================================
# Ortho Spectrum Fourier Block (OSFB)
# ============================================================================

class EinFFT(nn.Module):
    """
    Ortho Spectrum Fourier Block (OSFB).

    Projects patch features into the 2-D frequency domain via ortho-normalised
    FFT, applies two-stage learnable complex filtering (nonlinear then linear),
    sparsifies coefficients with soft-shrinkage, and reconstructs via IFFT.

    Args:
        dim (int): feature dimension (must be divisible by num_blocks=4)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.hidden_size  = dim
        self.num_blocks   = 4
        self.block_size   = dim // self.num_blocks
        assert dim % self.num_blocks == 0, "dim must be divisible by 4"

        self.sparsity_threshold = 0.01
        scale = 0.02

        # Two-stage learnable complex weights and biases
        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size) * scale
        )
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size) * scale
        )
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size) * scale
        )
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size) * scale
        )

    def _multiply(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Block-wise matrix multiply: (..., num_blocks, block_size) × (num_blocks, block_size, block_size)."""
        return torch.einsum('...bd,bdk->...bk', x, w)

    def forward(self, x: torch.Tensor, H=None, W=None) -> torch.Tensor:
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)

        # 2-D FFT across (N, num_blocks) — ortho normalisation
        x = torch.fft.fft2(x, dim=(1, 2), norm='ortho')

        # Stage 1: nonlinear complex filtering (complex multiplication + ReLU)
        x_r1 = F.relu(
            self._multiply(x.real, self.complex_weight_1[0])
            - self._multiply(x.imag, self.complex_weight_1[1])
            + self.complex_bias_1[0]
        )
        x_i1 = F.relu(
            self._multiply(x.real, self.complex_weight_1[1])
            + self._multiply(x.imag, self.complex_weight_1[0])
            + self.complex_bias_1[1]
        )

        # Stage 2: linear complex refinement
        x_r2 = (
            self._multiply(x_r1, self.complex_weight_2[0])
            - self._multiply(x_i1, self.complex_weight_2[1])
            + self.complex_bias_2[0]
        )
        x_i2 = (
            self._multiply(x_r1, self.complex_weight_2[1])
            + self._multiply(x_i1, self.complex_weight_2[0])
            + self.complex_bias_2[1]
        )

        # Merge, soft-shrink, inverse FFT
        x = torch.stack([x_r2, x_i2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = torch.fft.ifft2(x, dim=(1, 2), norm='ortho').real.float()
        return x.reshape(B, N, C)


# ============================================================================
# Feedforward MLP
# ============================================================================

class Mlp(nn.Module):
    """
    Two-layer feedforward MLP with GELU activation and optional dropout.

    Args:
        in_features     (int): input feature dimension
        hidden_features (int): hidden layer width  (default = in_features)
        out_features    (int): output dimension     (default = in_features)
        drop           (float): dropout probability
    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ============================================================================
# Masked Vanilla Attention Block (MVAB)
# ============================================================================

class Attention(nn.Module):
    """
    Masked Vanilla Attention Block (MVAB).

    Partitions the spatial feature map into overlapping patches and computes
    self-attention between them. A diagonal self-suppression mask (M_{a,a} = -inf)
    prevents any patch from attending to itself, preserving fine-grained locality
    and avoiding over-smoothing of anomaly boundaries.

    Args:
        dim          (int)  : feature channels
        patch_size   (int)  : inner sub-patch size (default 3)
        patch_stride (int)  : sub-patch stride     (default 3)
        attn_drop   (float) : attention dropout
    """

    def __init__(self, dim: int, patch_size: int = 3, patch_stride: int = 3,
                 attn_drop: float = 0.0):
        super().__init__()
        self.psize   = patch_size
        self.pstride = patch_stride
        self.N       = patch_size * patch_stride

        self.attn_drop  = nn.Dropout(attn_drop)
        self.softmax    = nn.Softmax(dim=-1)
        self.hidden_dim = dim // 2
        self.fc         = nn.Linear(dim, self.hidden_dim, bias=True)
        self.scale      = (self.hidden_dim * patch_size ** 2) ** -0.5

        # Diagonal self-suppression mask (fixed, not learned)
        mask = torch.eye(self.N)
        mask[mask == 1] = -100.0
        self.register_buffer('mask', mask)

    def _calculate_mask(self, block_idx, match_vec):
        B = block_idx.size(0) if isinstance(block_idx, torch.Tensor) else 1
        if match_vec is None or (isinstance(match_vec, torch.Tensor) and match_vec.sum() == 0):
            return self.mask
        cur_match = torch.index_select(match_vec, 0, block_idx).squeeze()
        mask = self.mask.unsqueeze(0).repeat(B, 1, 1)
        mask[cur_match == 1] = 0.0
        return mask

    def _attn_cal(self, attn: torch.Tensor, mask, v: torch.Tensor, shape):
        B, H, W, C = shape
        attn = attn + mask
        # Second-order feedback: self-attention on attention matrix
        attn2 = self.softmax(torch.bmm(attn, attn.transpose(2, 1)) / (attn.size(-1) ** 0.5))
        attn  = self.softmax(attn + attn2)
        x_attn = attn @ v
        x = x_attn.view(B, self.pstride, self.pstride, self.psize, self.psize, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

    def forward(self, x: torch.Tensor, block_idx=0, match_vec=None) -> torch.Tensor:
        B, H, W, C = x.shape
        P = self.psize ** 2

        x_view = x.view(B, self.pstride, self.psize, self.pstride, self.psize, C)
        x_fc   = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, P, C)
        x_q    = self.fc(x_fc).view(B, self.N, -1)

        attn = (x_q @ x_q.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn)

        v    = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, self.N, P * C)
        mask = self._calculate_mask(block_idx, match_vec)
        return self._attn_cal(attn, mask, v, [B, H, W, C])


# ============================================================================
# SM-HAD Hybrid Transformer Block  (OSFB + MVAB + RMB + MLP)
# ============================================================================

class SMHADBlock(nn.Module):
    """
    Core building block of SM-HAD.

    Sequential composition:
        1. OSFB  — global spectral processing in frequency domain
        2. MVAB  — locality-aware masked spatial attention
        3. RMB   — linear-time state-space long-range modeling (Mamba)
        4. MLP   — pointwise nonlinear feature refinement

    Each sub-module uses a residual connection and layer normalisation.

    Args:
        dim          (int)  : feature channels
        patch_size   (int)  : MVAB inner patch size
        patch_stride (int)  : MVAB inner patch stride
        mlp_ratio   (float) : MLP hidden-dim expansion ratio
        attn_drop   (float) : MVAB attention dropout
        drop        (float) : MLP dropout
        d_state      (int)  : Mamba SSM state dimension
    """

    def __init__(self, dim: int, patch_size: int = 3, patch_stride: int = 3,
                 mlp_ratio: float = 4.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_drop: float = 0.0,
                 drop: float = 0.0, d_state: int = 32):
        super().__init__()

        # OSFB
        self.norm_osfb = norm_layer(dim)
        self.osfb      = EinFFT(dim)

        # MVAB
        self.norm_mvab = norm_layer(dim)
        self.mvab      = Attention(dim, patch_size=patch_size,
                                   patch_stride=patch_stride, attn_drop=attn_drop)

        # RMB (Residual Mamba Block)
        self.norm_rmb  = norm_layer(dim)
        self.rmb       = Mamba(d_model=dim, d_state=d_state)

        # MLP
        self.norm_mlp  = norm_layer(dim)
        self.mlp       = Mlp(in_features=dim,
                             hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, block_idx=0, match_vec=None) -> torch.Tensor:
        B, H, W, C = x.shape
        x_seq = x.view(B, H * W, C)

        # 1. OSFB
        x_seq = x_seq + self.osfb(self.norm_osfb(x_seq), H, W)

        # 2. MVAB
        x_spatial = self.norm_mvab(x_seq).view(B, H, W, C)
        x_spatial = self.mvab(x_spatial, block_idx=block_idx, match_vec=match_vec)
        x_seq     = x_spatial.view(B, H * W, C)

        # 3. RMB
        x_seq = x_seq + self.rmb(self.norm_rmb(x_seq))

        # 4. MLP
        x_seq = x_seq + self.mlp(self.norm_mlp(x_seq))

        return x_seq.view(B, H, W, C)


# ============================================================================
# SM-HAD: Full Encoder-Decoder Network
# ============================================================================

class Net(nn.Module):
    """
    SM-HAD: Spectrum Mamba for Hyperspectral Anomaly Detection.

    Encoder-decoder reconstruction network:
        conv_head   : 3×3 conv → embed_dim feature space
        attn_layer  : one SMHADBlock (OSFB + MVAB + RMB + MLP)
        conv_tail   : 3×3 conv → original spectral channels

    At inference the squared reconstruction residual is used as the anomaly score.

    Args:
        in_chans     (int)  : number of spectral bands (C)
        embed_dim    (int)  : embedding dimension D  (paper: 96)
        patch_size   (int)  : MVAB sub-patch size    (paper: 3)
        patch_stride (int)  : MVAB sub-patch stride  (paper: 3)
        mlp_ratio   (float) : MLP expansion ratio    (paper: 2.0)
        attn_drop   (float) : attention dropout
        drop        (float) : MLP dropout
        d_state      (int)  : Mamba state dimension  (paper: 64)
    """

    def __init__(self, in_chans: int = 3, embed_dim: int = 96,
                 patch_size: int = 3, patch_stride: int = 3,
                 mlp_ratio: float = 2.0, attn_drop: float = 0.0,
                 drop: float = 0.0, d_state: int = 64):
        super().__init__()

        self.conv_head = nn.Conv2d(in_chans, embed_dim,
                                   kernel_size=3, stride=1, padding=1)
        self.attn_layer = SMHADBlock(
            dim=embed_dim,
            patch_size=patch_size,
            patch_stride=patch_stride,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            drop=drop,
            d_state=d_state,
        )
        self.conv_tail = nn.Conv2d(embed_dim, in_chans,
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, block_idx=0, match_vec=None) -> torch.Tensor:
        x = self.conv_head(x)
        x = x.permute(0, 2, 3, 1).contiguous()          # (B, H, W, C)
        x = self.attn_layer(x, block_idx=block_idx, match_vec=match_vec)
        x = x.permute(0, 3, 1, 2).contiguous()          # (B, C, H, W)
        x = self.conv_tail(x)
        return x
