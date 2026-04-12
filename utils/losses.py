"""
Focal Loss for handling class imbalance in crop classification.
Paper reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.
Used in MCTNet (Wang et al., 2024) for crop mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss = -α_t * (1 - p_t)^γ * log(p_t)

    Reduces the loss contribution from easy/well-classified examples
    and focuses training on hard, misclassified ones.

    Args:
        alpha   : Tensor of shape (n_classes,) — per-class weights.
                  If None, all classes are weighted equally.
        gamma   : float — focusing parameter (default: 2.0).
                  γ=0 is equivalent to standard cross-entropy.
        reduction : str — 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs  : (B, C) — raw logits
            targets : (B,)   — class indices (int64)
        Returns:
            Focal loss scalar (or per-sample if reduction='none')
        """
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(labels, n_classes, device="cpu"):
    """
    Compute inverse-frequency class weights for Focal Loss alpha.

    Args:
        labels    : Tensor of all training labels
        n_classes : int
        device    : str

    Returns:
        Tensor of shape (n_classes,) with normalised weights
    """
    counts = torch.bincount(labels, minlength=n_classes).float()
    counts = counts.clamp(min=1)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes  # normalise
    return weights.to(device)
