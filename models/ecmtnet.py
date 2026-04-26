"""
=======================================================================
ECMTNet — Enhanced Crop Mapping Transformer Network
Part 3: Improved Architecture

Project: Deep Learning for Crop Classification Using Multi-Source
         Satellite Data (M1 SII 2025/2026 — USTHB)

Improvements over MCTNet (Wang et al., 2024):
    1. GatedFusion  : replaces naive concatenation with a learned gate
                      that dynamically balances CNN vs Transformer.
    2. PhenologyAttention : temporal attention head that explicitly
                      learns to upweight phenologically important dates.
    3. CrossScaleFusion : aggregates multi-scale representations from
                      ALL three stages instead of only the last one.
    4. Deeper MLP head with LayerNorm and Dropout for regularisation.

Architecture dimension flow (baseline in_channels=10):
    Input   : (B, 36, 10)
    Stage 1 : → GatedFusion → (B, 36, 20)  → Pool → (B, 18, 20)
    Stage 2 : → GatedFusion + PhenoAttn → (B, 18, 40) → Pool → (B, 9, 40)
    Stage 3 : → GatedFusion → (B,  9, 80)
    CrossScale: pool each stage → concat → (B, 140)
    Head    : → Linear(140, 64) → GELU → Linear(64, n_classes)

Inspired by:
    - MCTNet     (Wang et al., 2024)
    - GL-TAE     (Zhang et al., 2023)
    - ECA-Net    (Wang et al., 2020)
    - GFNet / Gated feature fusion (Chen et al., 2023)
=======================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# ECA — Efficient Channel Attention (same as MCTNet)
# -----------------------------------------------------------------------
class ECAModule(nn.Module):
    """Efficient Channel Attention via a lightweight 1D conv."""
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, T, C)"""
        y = self.avg_pool(x.transpose(1, 2))   # (B, C, 1)
        y = self.conv(y.transpose(1, 2))        # (B, 1, C)
        y = self.sigmoid(y)                     # (B, 1, C)
        return x * y


# -----------------------------------------------------------------------
# ALPE — Adaptive Learned Positional Encoding (Stage 1 only, from MCTNet)
# -----------------------------------------------------------------------
class ALPE(nn.Module):
    """Masks PE at missing timesteps then refines via Conv1D + ECA."""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.eca  = ECAModule(kernel_size=3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B,T,C), mask: (B,T)  →  (B,T,C)"""
        T = x.size(1)
        pe = self.pe[:, :T, :]
        pe_masked  = pe * mask.unsqueeze(-1)
        pe_refined = self.conv(pe_masked.transpose(1, 2)).transpose(1, 2)
        pe_out     = self.eca(pe_refined)
        return x + pe_out


# -----------------------------------------------------------------------
# Standard Sinusoidal PE (Stages 2 & 3)
# -----------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# -----------------------------------------------------------------------
# MSCNN — Multi-Scale CNN Branch (same as MCTNet)
# -----------------------------------------------------------------------
class MSCNN(nn.Module):
    """Two-layer Conv1D with BN and residual connection."""
    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = x.transpose(1, 2)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out)).transpose(1, 2)
        return F.relu(out + res)


# -----------------------------------------------------------------------
# Transformer Branch (same as MCTNet, pre-LayerNorm)
# -----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Multi-Head Self-Attention + FFN with pre-norm."""
    def __init__(self, d_model: int, n_heads: int = 5, ffn_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_factor, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        a, _ = self.attn(xn, xn, xn)
        x = x + self.drop1(a)
        x = x + self.ffn(self.norm2(x))
        return x


# -----------------------------------------------------------------------
# ★ NEW: GatedFusion — learned gate replaces naive concatenation
# -----------------------------------------------------------------------
class GatedFusion(nn.Module):
    """
    Gated feature fusion of CNN and Transformer outputs.

    Instead of concatenating [cnn_out, trans_out] (which weights both
    equally), we learn a sigmoid gate G ∈ (0,1)^(B,T,d) that controls
    how much of each branch to keep at each position.

        fused = G * cnn_out + (1 - G) * trans_out
        out   = LayerNorm(fused)  then project to 2*d_model

    This lets the model be selective: if temporal context matters more
    (growing-season boundaries), trans_out dominates; if local spectral
    texture matters (distinguishing cotton from soybean at peak NDVI),
    cnn_out dominates.

    Output dim: 2 * d_model  (same as MCTNet for fair comparison)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Gate network: takes concatenated features and produces gate
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        # Project fused (d_model) to 2*d_model to match MCTNet output size
        self.proj   = nn.Linear(d_model, 2 * d_model)
        self.norm   = nn.LayerNorm(2 * d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cnn_out: torch.Tensor, trans_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_out   : (B, T, d_model)
            trans_out : (B, T, d_model)
        Returns:
            (B, T, 2 * d_model)
        """
        combined = torch.cat([cnn_out, trans_out], dim=-1)  # (B,T,2d)
        g = self.gate(combined)                             # (B,T,d) in (0,1)
        fused = g * cnn_out + (1 - g) * trans_out           # (B,T,d)
        out   = self.proj(fused)                            # (B,T,2d)
        return self.norm(self.dropout(out))


# -----------------------------------------------------------------------
# ★ NEW: PhenologyAttention — highlights phenologically important dates
# -----------------------------------------------------------------------
class PhenologyAttention(nn.Module):
    """
    Temporal attention module that learns to upweight phenologically
    important timesteps (e.g., flowering, maturity, harvest).

    Mechanism:
        1. Linear projection from d_model → 1 (scalar score per timestep)
        2. Softmax over T → attention weights α ∈ Δ^T
        3. Weighted aggregation along time → context vector c ∈ R^d
        4. Broadcast c and add to x (residual-style enhancement)

    This is applied AFTER the Transformer block in Stage 2, where the
    sequence is at half the temporal resolution (T=18), making the
    attention weights interpretable as importance over ~20-day windows.

    The module does NOT reduce the time dimension — it produces an
    enhanced sequence of the same shape (B, T, d_model).
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Linear(d_model, 1)   # temporal importance scorer
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, d_model)
        Returns:
            (B, T, d_model) — attention-enhanced sequence
        """
        # Compute softmax attention weights over time
        scores = self.score(x)                    # (B, T, 1)
        weights = torch.softmax(scores, dim=1)    # (B, T, 1) — sum over T = 1

        # Weighted context vector (global phenological summary)
        context = (weights * x).sum(dim=1, keepdim=True)  # (B, 1, d)

        # Broadcast context to all timesteps and add (residual)
        out = x + self.drop(context.expand_as(x))
        return self.norm(out)


# -----------------------------------------------------------------------
# ECMTBlock — Single stage: PE → CNN + Transformer → GatedFusion [+ PhenoAttn]
# -----------------------------------------------------------------------
class ECMTBlock(nn.Module):
    """
    One ECMTNet stage.
    use_alpe=True  → Stage 1 (handles missing data)
    use_pheno=True → Stage 2 (phenology attention before fusion)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 5,
        ffn_factor: int = 4,
        dropout: float = 0.1,
        use_alpe: bool = False,
        use_pheno: bool = False,
    ):
        super().__init__()
        self.use_alpe  = use_alpe
        self.use_pheno = use_pheno

        self.pe          = ALPE(d_model) if use_alpe else SinusoidalPE(d_model)
        self.mscnn       = MSCNN(d_model)
        self.transformer = TransformerBlock(d_model, n_heads, ffn_factor, dropout)
        self.fusion      = GatedFusion(d_model, dropout)

        if use_pheno:
            self.pheno_attn = PhenologyAttention(d_model, dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x    : (B, T, d_model)
            mask : (B, T) — only used when use_alpe=True
        Returns:
            (B, T, 2 * d_model)
        """
        if self.use_alpe and mask is not None:
            x_pe = self.pe(x, mask)
        else:
            x_pe = self.pe(x)

        cnn_out   = self.mscnn(x_pe)         # (B, T, d_model)
        trans_out = self.transformer(x_pe)   # (B, T, d_model)

        # Optional phenology attention on transformer branch BEFORE fusion
        if self.use_pheno:
            trans_out = self.pheno_attn(trans_out)

        return self.fusion(cnn_out, trans_out)  # (B, T, 2*d_model)


# -----------------------------------------------------------------------
# ★ NEW: CrossScaleFusion — aggregate representations from all 3 stages
# -----------------------------------------------------------------------
class CrossScaleFusion(nn.Module):
    """
    Combines intermediate feature maps from all three stages.

    Each stage output is reduced to a global vector via Max Pooling,
    then all vectors are concatenated → fed to a linear mixer.

    Motivation: MCTNet discards information from Stages 1 & 2 (only
    the final Stage 3 output feeds the classifier). CrossScaleFusion
    preserves multi-resolution phenological signals:
        - Stage 1 (T=36, 20-ch): fine-grained 10-day patterns
        - Stage 2 (T=18, 40-ch): mid-season transitions
        - Stage 3 (T= 9, 80-ch): high-level crop identity
    """
    def __init__(self, dims: list, out_dim: int, dropout: float = 0.1):
        """
        Args:
            dims    : list of channel sizes from each stage, e.g. [20, 40, 80]
            out_dim : final output dimension
        """
        super().__init__()
        in_total = sum(dims)
        self.mixer = nn.Sequential(
            nn.Linear(in_total, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, feats: list) -> torch.Tensor:
        """
        Args:
            feats : list of (B, T_i, C_i) tensors from each stage
        Returns:
            (B, out_dim)
        """
        pooled = [f.amax(dim=1) for f in feats]   # list of (B, C_i)
        cat    = torch.cat(pooled, dim=-1)          # (B, sum_C_i)
        return self.mixer(cat)                      # (B, out_dim)


# -----------------------------------------------------------------------
# ECMTNet — Full Improved Model
# -----------------------------------------------------------------------
class ECMTNet(nn.Module):
    """
    ECMTNet: Enhanced Crop Mapping Transformer Network.

    Key improvements over MCTNet:
        1. GatedFusion        replaces concatenation in each stage
        2. PhenologyAttention applied in Stage 2 on Transformer branch
        3. CrossScaleFusion   aggregates all three stage representations

    Dimension flow (base_dim = 10):
        Input   : (B, 36, in_channels)
        Proj    : (B, 36, 10)          ← input_proj (fair ablation)
        Stage 1 : (B, 36, 20)          → pool → (B, 18, 20)
        Stage 2 : (B, 18, 40)          → pool → (B,  9, 40)
        Stage 3 : (B,  9, 80)
        Cross   : pool each → concat [20+40+80=140] → mixer → (B, out_dim)
        Head    : Linear(out_dim, 64) → GELU → Linear(64, n_classes)

    Args:
        in_channels : number of spectral/covariate bands (default: 10)
        n_classes   : number of crop types
        n_heads     : attention heads per transformer (default: 5)
        ffn_factor  : FFN expansion factor (default: 4)
        dropout     : dropout rate (default: 0.2)
        out_dim     : cross-scale fusion output dimension (default: 128)
    """

    def __init__(
        self,
        in_channels: int = 10,
        n_classes: int = 5,
        n_heads: int = 5,
        ffn_factor: int = 4,
        dropout: float = 0.2,
        out_dim: int = 128,
    ):
        super().__init__()

        # --- FAIR ABLATION: project all configs to same base_dim ---
        self.base_dim = 10
        self.input_proj = nn.Linear(in_channels, self.base_dim)

        d1 = self.base_dim      # 10
        d2 = d1 * 2             # 20  (output of stage 1 GatedFusion)
        d3 = d2 * 2             # 40  (output of stage 2 GatedFusion)
        d4 = d3 * 2             # 80  (output of stage 3 GatedFusion)

        # Stage 1: ALPE + GatedFusion (missing data handling)
        self.stage1 = ECMTBlock(d1, n_heads, ffn_factor, dropout,
                                use_alpe=True, use_pheno=False)
        self.pool1  = nn.MaxPool1d(kernel_size=2)

        # Stage 2: SinPE + GatedFusion + PhenologyAttention
        self.stage2 = ECMTBlock(d2, n_heads, ffn_factor, dropout,
                                use_alpe=False, use_pheno=True)
        self.pool2  = nn.MaxPool1d(kernel_size=2)

        # Stage 3: SinPE + GatedFusion
        self.stage3 = ECMTBlock(d3, n_heads, ffn_factor, dropout,
                                use_alpe=False, use_pheno=False)

        # Cross-scale fusion: aggregates [d2, d3, d4] = [20, 40, 80]
        self.cross_scale = CrossScaleFusion([d2, d3, d4], out_dim, dropout)

        # Deep classification head
        self.head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x    : (B, T=36, C)  — normalised time-series features
            mask : (B, T=36)     — binary missing-data mask (1=present)
        Returns:
            logits : (B, n_classes)
        """
        # Input projection (ensures fair parameter count across configs)
        x = self.input_proj(x)   # (B, 36, base_dim)

        # Stage 1: ALPE handles missing data
        # (B,36,10) → GatedFusion → (B,36,20) → pool → (B,18,20)
        out1 = self.stage1(x, mask)
        out1_pooled = self.pool1(out1.transpose(1, 2)).transpose(1, 2)

        # Stage 2: PhenologyAttention on Transformer branch
        # (B,18,20) → GatedFusion → (B,18,40) → pool → (B,9,40)
        out2 = self.stage2(out1_pooled)
        out2_pooled = self.pool2(out2.transpose(1, 2)).transpose(1, 2)

        # Stage 3: Final feature extraction
        # (B,9,40) → GatedFusion → (B,9,80)
        out3 = self.stage3(out2_pooled)

        # Cross-scale fusion: pools all stages and concatenates
        # [pool(out1)=20, pool(out2)=40, pool(out3)=80] → (B, out_dim)
        fused = self.cross_scale([out1, out2, out3])

        # Classification
        return self.head(fused)   # (B, n_classes)


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------
def count_parameters(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("=" * 60)
    print("ECMTNet — Architecture Verification")
    print("=" * 60)

    for n_cls, name, n_in in [(5, "Arkansas (baseline)", 10),
                               (6, "California (baseline)", 10),
                               (5, "Arkansas (all covariates)", 19)]:
        model = ECMTNet(in_channels=n_in, n_classes=n_cls)
        x     = torch.randn(4, 36, n_in)
        mask  = torch.ones(4, 36)
        mask[:, [5, 10, 20]] = 0   # simulate missing timesteps

        logits = model(x, mask)
        total, trainable = count_parameters(model)

        print(f"\n  {name} (n_classes={n_cls}, in_channels={n_in}):")
        print(f"    Input  : x={tuple(x.shape)}, mask={tuple(mask.shape)}")
        print(f"    Output : logits={tuple(logits.shape)}")
        print(f"    Params : {total:,} total, {trainable:,} trainable")

    # Compare with MCTNet parameter count (from paper: ~55K)
    from models.mctnet import MCTNet
    mct = MCTNet(in_channels=10, n_classes=5)
    mct_total, _ = count_parameters(mct)
    ecmt = ECMTNet(in_channels=10, n_classes=5)
    ecmt_total, _ = count_parameters(ecmt)

    print(f"\n  Parameter comparison:")
    print(f"    MCTNet  : {mct_total:,}")
    print(f"    ECMTNet : {ecmt_total:,}  ({(ecmt_total/mct_total - 1)*100:+.1f}%)")
    print(f"\n{'=' * 60}")
    print("ECMTNet architecture OK")
    print("=" * 60)
