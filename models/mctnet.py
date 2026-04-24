"""
=======================================================================
MCTNet — Multi-scale CNN-Transformer Network
Paper : A lightweight CNN-Transformer network for pixel-based crop
        mapping using time-series Sentinel-2 imagery (Wang et al., 2024)

Architecture (3-stage hierarchical):
    Stage 1: (B,36,10) → ALPE → MSCNN + Transformer → CTFusion → (B,36,20)
             → MaxPool → (B,18,20)
    Stage 2: (B,18,20) → Positional Encoding(PE) → MSCNN + Transformer → CTFusion → (B,18,40)
             → MaxPool → (B,9,40)
    Stage 3: (B,9,40)  → PE   → MSCNN + Transformer → CTFusion → (B,9,80)
             → GMP (Max) → (B,80) → Linear → (B, n_classes)


             
Le Positional Encoding permet au Transformer de connaître la position temporelle de chaque observation. 
Sans cela, il ne pourrait pas comprendre l’ordre des données.

CNN → capture les motifs locaux dans le temps

Transformer → capture les dépendances à long terme

ALPE → gère intelligemment les dates manquantes (position+mark+eca(ychuf les donnees inportantes))
=======================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# ECA — Efficient Channel Attention => importance t3 les bande
# -----------------------------------------------------------------------
class ECAModule(nn.Module):
    """
    Efficient Channel Attention (ECA).
    Learns channel-wise attention weights via a lightweight 1D convolution
    on the global-average-pooled features.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, T, C) — channel-attention weighted
        """
        # Pool over time → (B, C, 1)
        y = self.avg_pool(x.transpose(1, 2))   # (B, C, 1)
        # Conv over channels → (B, 1, C)
        y = y.transpose(1, 2)                   # (B, 1, C)
        y = self.conv(y)                         # (B, 1, C)
        y = self.sigmoid(y)                      # (B, 1, C)
        return x * y                             # broadcast (B, T, C)


# -----------------------------------------------------------------------
# ALPE — Adaptive Learned Positional Encoding (Stage 1 only)  smart 3la pe ki myl9ach date y9ulu ychuf li 9blu w morah 
# -----------------------------------------------------------------------
class ALPE(nn.Module):
    """
    Adaptive Learned Positional Encoding.
    Handles missing data by masking the positional encoding and
    refining it with a Conv1D + ECA attention module.

    Pipeline: Sinusoidal PE → mask → Conv1D → ECA → add to input
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()

        # Pre-compute sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        # Learnable refinement
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.eca = ECAModule(kernel_size=3)

    def forward(self, x, mask):
        """
        Args:
            x    : (B, T, C)
            mask : (B, T) — 1 = present, 0 = missing
        Returns:
            (B, T, C) — input with adaptive positional encoding added
        """
        T = x.size(1)
        pe = self.pe[:, :T, :]                           # (1, T, C)
        pe_masked = pe * mask.unsqueeze(-1)               # (B, T, C)
        pe_refined = self.conv(
            pe_masked.transpose(1, 2)
        ).transpose(1, 2)                                 # (B, T, C)
        pe_out = self.eca(pe_refined)                     # (B, T, C)
        return x + pe_out


# -----------------------------------------------------------------------
# Standard Sinusoidal PE (Stages 2 and 3) ystfhum 3la 7ssab les dates 
# -----------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """x: (B, T, C) → (B, T, C) with PE added."""
        return x + self.pe[:, :x.size(1), :]


# -----------------------------------------------------------------------
# MSCNN — Multi-Scale CNN Branch
# -----------------------------------------------------------------------
class MSCNN(nn.Module):
    """
    Multi-Scale CNN branch.
    Two-layer Conv1D with BatchNorm and residual connection.
    Extracts local temporal features within each hierarchical stage.
    The multi-scale aspect is achieved through the 3-stage hierarchy
    operating at different temporal resolutions (36 → 18 → 9).
    """
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, T, C) — local temporal features + residual
        """
        residual = x
        out = x.transpose(1, 2)                  # (B, C, T)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out = out.transpose(1, 2)                 # (B, T, C)
        return F.relu(out + residual)


# -----------------------------------------------------------------------
# Transformer Branch
# -----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Transformer branch with pre-LayerNorm.
    Multi-Head Self-Attention + Feed-Forward Network.
    Captures global temporal dependencies across all timesteps.
    """
    def __init__(self, d_model, n_heads=5, ffn_factor=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_factor, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, T, C)
        """
        # Self-attention with pre-norm + residual
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + self.dropout1(attn_out)

        # FFN with pre-norm + residual
        x = x + self.ffn(self.norm2(x))
        return x


# -----------------------------------------------------------------------
# MCTBlock — Single stage block (fusionne CNN + Transformer)
# -----------------------------------------------------------------------
class MCTBlock(nn.Module):
    """
    One MCTNet stage: MSCNN + Transformer branches → CTFusion (concat).
    """
    def __init__(self, d_model, n_heads=5, ffn_factor=8,
                 dropout=0.1, use_alpe=False):
        super().__init__()
        self.use_alpe = use_alpe

        # Positional encoding
        self.pe = ALPE(d_model) if use_alpe else SinusoidalPE(d_model)

        # Dual branches
        self.mscnn = MSCNN(d_model)
        self.transformer = TransformerBlock(d_model, n_heads, ffn_factor, dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x    : (B, T, d_model)
            mask : (B, T) — only used when use_alpe=True
        Returns:
            (B, T, 2 * d_model) — fused features from both branches
        """
        # Apply positional encoding
        if self.use_alpe and mask is not None:
            x_pe = self.pe(x, mask)
        else:
            x_pe = self.pe(x)

        # Parallel branches
        cnn_out = self.mscnn(x_pe)            # (B, T, d_model)
        trans_out = self.transformer(x_pe)    # (B, T, d_model)

        # CTFusion: concatenation along feature dimension
        return torch.cat([cnn_out, trans_out], dim=-1)   # (B, T, 2*d_model)


# -----------------------------------------------------------------------
# MCTNet — Full model
# -----------------------------------------------------------------------
class MCTNet(nn.Module):
    """
    MCTNet: Multi-scale CNN-Transformer Network.

    3-stage hierarchical architecture for pixel-based crop classification
    from time-series Sentinel-2 data.

    Dimension flow:
        Input   : (B, 36, 10)
        Stage 1 : (B, 36, 10)  → (B, 36, 20)  → Pool → (B, 18, 20)
        Stage 2 : (B, 18, 20)  → (B, 18, 40)  → Pool → (B,  9, 40)
        Stage 3 : (B,  9, 40)  → (B,  9, 80)
        GAP     : (B, 80)
        Head    : (B, n_classes)

    Args:
        in_channels : int — number of spectral bands (default: 10)
        n_classes   : int — number of crop classes
        n_heads     : int — attention heads per transformer (default: 5)
        ffn_factor  : int — FFN expansion factor (default: 8)
        dropout     : float — dropout rate (default: 0.1)
    """
    def __init__(self, in_channels=10, n_classes=5, n_heads=5,
                 ffn_factor=8, dropout=0.1):
        super().__init__()

        d1 = in_channels        # 10
        d2 = in_channels * 2    # 20
        d3 = in_channels * 4    # 40
        d_out = in_channels * 8 # 80

        # Stage 1: ALPE-enabled (handles missing data)
        self.stage1 = MCTBlock(d1, n_heads, ffn_factor, dropout, use_alpe=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Stage 2: standard PE
        self.stage2 = MCTBlock(d2, n_heads, ffn_factor, dropout, use_alpe=False)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Stage 3: standard PE
        self.stage3 = MCTBlock(d3, n_heads, ffn_factor, dropout, use_alpe=False)

        # Classification head
        self.head = nn.Linear(d_out, n_classes)

    def forward(self, x, mask=None):
        """
        Args:
            x    : (B, T=36, C=10)  — normalised spectral time series
            mask : (B, T=36)        — binary missing-data mask
        Returns:
            logits : (B, n_classes)
        """
        # Stage 1: (B,36,10) → (B,36,20) → pool → (B,18,20)
        out = self.stage1(x, mask)
        out = self.pool1(out.transpose(1, 2)).transpose(1, 2)

        # Stage 2: (B,18,20) → (B,18,40) → pool → (B,9,40)
        out = self.stage2(out)
        out = self.pool2(out.transpose(1, 2)).transpose(1, 2)

        # Stage 3: (B,9,40) → (B,9,80)
        out = self.stage3(out)

        # Global Max Pooling (Paper Section 2.3.2) -> (B, 80)
        out = out.amax(dim=1)

        # Classification -> (B, n_classes)
        return self.head(out)


# -----------------------------------------------------------------------
# Utility: model summary
# -----------------------------------------------------------------------
def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick sanity check
    print("=" * 55)
    print("MCTNet — Architecture Verification")
    print("=" * 55)

    for n_cls, name in [(5, "Arkansas"), (6, "California")]:
        model = MCTNet(in_channels=10, n_classes=n_cls)
        x = torch.randn(4, 36, 10)
        mask = torch.ones(4, 36)
        mask[:, 5] = 0   # simulate one missing timestep

        logits = model(x, mask)
        total, trainable = count_parameters(model)

        print(f"\n  {name} (n_classes={n_cls}):")
        print(f"    Input  : x={tuple(x.shape)}, mask={tuple(mask.shape)}")
        print(f"    Output : logits={tuple(logits.shape)}")
        print(f"    Params : {total:,} total, {trainable:,} trainable")

    print(f"\n{'=' * 55}")
    print("✅ MCTNet architecture OK")
    print("=" * 55)
