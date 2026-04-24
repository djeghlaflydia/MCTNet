"""
=======================================================================
PART 1 — STEP 7 : EVALUATION & VISUALIZATION
Project: Deep Learning for Crop Classification Using Multi-Source
         Satellite Data (M1 SII 2025/2026 — USTHB)
Paper  : MCTNet — Wang et al., 2024

Evaluates the trained MCTNet model and produces:
    - OA, AA, Kappa, F1 (Table comparison with paper)
    - Per-class F1 scores
    - Confusion matrix heatmap
    - Training curves (loss & accuracy)
=======================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mctnet import MCTNet
from utils.metrics import compute_metrics, print_metrics

# -----------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------
class CropDataset(torch.utils.data.Dataset):
    def __init__(self, state, split):
        path = os.path.join(PREPROCESSED_DIR, state, f"{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found: {path}. Run preprocessing first.")
        
        data = torch.load(path, weights_only=False)
        self.X = data["X"]
        self.mask = data["mask"]
        self.y = data["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
PREPROCESSED_DIR = "./preprocessed"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

AR_CLASSES = {0: "Others", 1: "Corn", 2: "Cotton", 3: "Rice", 4: "Soybeans"}
CA_CLASSES = {0: "Others", 1: "Grapes", 2: "Rice", 3: "Alfalfa",
              4: "Almonds", 5: "Pistachios"}

PAPER_BASELINES = {
    "Arkansas": {"OA": 0.968, "Kappa": 0.951, "F1_macro": 0.933},
    "California": {"OA": 0.852, "Kappa": 0.806, "F1_macro": 0.829},
}

# -----------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for X, mask, y in loader:
        X = X.to(device)
        mask = mask.to(device)

        logits = model(X, mask)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

    return np.array(all_preds), np.array(all_labels)

# -----------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalise for display
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        annot_kws={"size": 11},
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.75,
                f"({cm[i, j]})",
                ha="center", va="center",
                fontsize=7, color="gray",
            )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def plot_training_curves(history, save_path, title="Training Curves"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], label="Val", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=1.5)
    axes[1].plot(epochs, history["val_acc"], label="Val", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def plot_per_class_f1(f1_scores, class_names, save_path, paper_f1=None, title="Per-Class F1 Score"):
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(class_names))
    width = 0.35

    bars = ax.bar(x, f1_scores, width, label="Ours", color="#1f77b4", edgecolor="white", linewidth=0.8)

    if paper_f1 is not None and len(paper_f1) == len(class_names):
        ax.bar(x + width, paper_f1, width, label="Paper", color="#ff7f0e", edgecolor="white", linewidth=0.8, alpha=0.7)

    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Crop Type")
    ax.set_ylabel("F1 Score")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x + width / 2 if paper_f1 else x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def print_paper_comparison(metrics, state):
    baseline = PAPER_BASELINES.get(state, {})
    if not baseline:
        return

    print(f"\n{'=' * 55}")
    print(f"  COMPARISON WITH PAPER — {state}")
    print(f"{'=' * 55}")
    print(f"  {'Metric':<20} {'Ours':>10} {'Paper':>10} {'Δ':>10}")
    print(f"  {'─' * 50}")

    for key, paper_val in baseline.items():
        our_val = metrics.get(key, 0)
        delta = our_val - paper_val
        symbol = "✅" if abs(delta) < 0.02 else ("⚠️" if delta < 0 else "🎯")
        print(f"  {key:<20} {our_val:>10.4f} {paper_val:>10.4f} {delta:>+10.4f} {symbol}")
    print(f"{'=' * 55}")

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def evaluate(state, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = os.path.join(RESULTS_DIR, state)
    ckpt_path = os.path.join(CHECKPOINT_DIR, state, "best_model.pt")
    hist_path = os.path.join(CHECKPOINT_DIR, state, "history.pt")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📂 Loading model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

    # Reconstruct class mapping
    class_names_map = AR_CLASSES if state == "Arkansas" else CA_CLASSES
    class_names = [class_names_map.get(i, f"Class {i}") for i in range(len(class_names_map))]
    n_classes = len(class_names)

    model = MCTNet(
        in_channels=10,
        n_classes=n_classes,
        n_heads=5,
        ffn_factor=8,
        dropout=0.1
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  ✓ Loaded best model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    print(f"\n🔍 Evaluating...")
    test_ds = CropDataset(state, "test")
    if len(test_ds) == 0:
        print("  ⚠ Test set is empty, evaluating on Val set instead.")
        test_ds = CropDataset(state, "val")
        
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    y_pred, y_true = predict(model, test_loader, device)

    if len(y_true) > 0:
        metrics = compute_metrics(y_true, y_pred, class_names)
        print_metrics(metrics, title=f"{state} — Evaluation Results")
        print_paper_comparison(metrics, state)

        plot_confusion_matrix(
            metrics["confusion_matrix"], class_names,
            os.path.join(save_dir, "confusion_matrix.png"),
            title=f"Confusion Matrix — {state}",
        )

        plot_per_class_f1(
            metrics["per_class_f1"], class_names,
            os.path.join(save_dir, "per_class_f1.png"),
            title=f"Per-Class F1 — {state}",
        )
    else:
        print("  ⚠ Not enough data for plotting.")

    if os.path.exists(hist_path):
        history = torch.load(hist_path, weights_only=False)
        plot_training_curves(
            history,
            os.path.join(save_dir, "training_curves.png"),
            title=f"Training Curves — {state}",
        )

    print(f"\n📁 All results saved to: {save_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained MCTNet")
    parser.add_argument("--state", type=str, default="Arkansas", choices=["Arkansas", "California"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 7 — EVALUATION & VISUALIZATION")
    print("======================================================")

    evaluate(args.state, args.device)
