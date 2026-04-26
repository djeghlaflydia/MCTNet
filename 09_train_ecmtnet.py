"""
=======================================================================
PART 3 — TRAINING ECMTNet
Project: Deep Learning for Crop Classification Using Multi-Source
         Satellite Data (M1 SII 2025/2026 — USTHB)

Trains ECMTNet and compares it with MCTNet using the same
preprocessed data from Part 1 & 2.

Usage:
    python 09_train_ecmtnet.py --state Arkansas --config baseline
    python 09_train_ecmtnet.py --state all --config all
=======================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ecmtnet import ECMTNet
from models.mctnet import MCTNet
from utils.metrics import compute_metrics, print_metrics

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
PREPROCESSED_DIR = "./preprocessed_ablation"
CHECKPOINT_DIR   = "./checkpoints/part3"
RESULTS_DIR      = "./results/part3"

CONFIG_CHANNELS = {
    "baseline":   10,
    "s2_climate": 12,
    "s2_soil":    14,
    "s2_topo":    13,
    "all":        19,
}

AR_CLASSES = ["Others", "Corn", "Cotton", "Rice", "Soybeans"]
CA_CLASSES = ["Others", "Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios"]


# -----------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------
class CropDataset(Dataset):
    def __init__(self, state: str, config: str, split: str):
        path = os.path.join(PREPROCESSED_DIR, state, config, f"{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found: {path}. Run preprocessing first.")
        data = torch.load(path, weights_only=False)
        self.X    = data["X"]
        self.mask = data["mask"]
        self.y    = data["y"]

    def __len__(self):  return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.mask[idx], self.y[idx]


def get_dataloaders(state: str, config: str, batch_size: int = 32):
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = CropDataset(state, config, split)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
    return loaders


# -----------------------------------------------------------------------
# TRAIN / EVAL HELPERS
# -----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, mask, y in loader:
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        logits = model(X, mask)
        loss   = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += X.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_labels), np.array(all_preds))


def train_model(model, model_name, state, config, epochs, device):
    save_dir = os.path.join(CHECKPOINT_DIR, state, model_name)
    result_dir = os.path.join(RESULTS_DIR, state, model_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    loaders      = get_dataloaders(state, config)
    class_names  = AR_CLASSES if state == "Arkansas" else CA_CLASSES
    criterion    = nn.CrossEntropyLoss()
    optimizer    = Adam(model.parameters(), lr=0.001, weight_decay=5e-2)
    scheduler    = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc  = 0.0
    patience_cnt  = 0
    PATIENCE      = 15
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n  [{model_name}] Training {state} | config={config}")
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for X, mask, y in loaders["train"]:
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X, mask)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * X.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)

        train_loss = run_loss / total
        train_acc  = correct  / total
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            marker = " ★ best"
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        else:
            patience_cnt += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or marker:
            print(f"  Epoch {epoch+1:3d}/{epochs:3d} | "
                  f"loss {train_loss:.4f}/{val_loss:.4f} | "
                  f"acc {train_acc:.4f}/{val_acc:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.6f}{marker}")

        if patience_cnt >= PATIENCE:
            print(f"  [!] Early stopping at epoch {epoch+1}")
            break

    # Save history
    torch.save(history, os.path.join(save_dir, "history.pt"))

    # Final evaluation on test set
    best_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    _, test_acc, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    metrics = compute_metrics(y_true, y_pred, class_names)
    print_metrics(metrics, title=f"{model_name} | {state} | {config}")

    # Save results
    torch.save({
        "metrics":    {k: v for k, v in metrics.items() if k != "classification_report"},
        "y_true":     y_true,
        "y_pred":     y_pred,
        "class_names": class_names,
        "report":     metrics["classification_report"],
    }, os.path.join(result_dir, "test_metrics.pt"))

    return {
        "model":         model_name,
        "state":         state,
        "config":        config,
        "test_accuracy": test_acc,
        "macro_f1":      metrics["F1_macro"],
        "kappa":         metrics["Kappa"],
        "OA":            metrics["OA"],
        "AA":            metrics["AA"],
    }


# -----------------------------------------------------------------------
# COMPARISON RUNNER
# -----------------------------------------------------------------------
def run_comparison(state, config, epochs, device):
    """Trains both MCTNet and ECMTNet and prints a comparison table."""
    in_ch       = CONFIG_CHANNELS[config]
    class_names = AR_CLASSES if state == "Arkansas" else CA_CLASSES
    n_cls       = len(class_names)

    models = {
        "MCTNet":  MCTNet(in_channels=in_ch, n_classes=n_cls, n_heads=5, ffn_factor=4, dropout=0.2),
        "ECMTNet": ECMTNet(in_channels=in_ch, n_classes=n_cls, n_heads=5, ffn_factor=4, dropout=0.2),
    }

    results = []
    for name, model in models.items():
        model = model.to(device)
        res = train_model(model, name, state, config, epochs, device)
        results.append(res)

    # Print side-by-side comparison
    print("\n" + "=" * 65)
    print(f"  PART 3 COMPARISON — {state} | config={config}")
    print("=" * 65)
    print(f"  {'Model':<12} {'OA':>8} {'Kappa':>8} {'F1_macro':>10} {'AA':>8}")
    print(f"  {'-' * 50}")
    for r in results:
        print(f"  {r['model']:<12} {r['OA']:>8.4f} {r['kappa']:>8.4f} "
              f"{r['macro_f1']:>10.4f} {r['AA']:>8.4f}")

    # Delta
    if len(results) == 2:
        delta_f1  = results[1]["macro_f1"]      - results[0]["macro_f1"]
        delta_oa  = results[1]["test_accuracy"]  - results[0]["test_accuracy"]
        delta_kap = results[1]["kappa"]          - results[0]["kappa"]
        print(f"\n  Delta (ECMTNet - MCTNet):")
        print(f"    OA:    {delta_oa:+.4f}")
        print(f"    Kappa: {delta_kap:+.4f}")
        print(f"    F1:    {delta_f1:+.4f}")
    print("=" * 65)

    return results


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 3 — ECMTNet Training & Comparison")
    parser.add_argument("--state",  type=str, default="Both",
                        choices=["Arkansas", "California", "Both"])
    parser.add_argument("--config", type=str, default="baseline",
                        choices=list(CONFIG_CHANNELS.keys()))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print("=" * 65)
    print("PART 3 — ECMTNet vs MCTNet Comparison")
    print("USTHB M1 SII 2025/2026")
    print("=" * 65)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n  Device: {device}")

    states = ["Arkansas", "California"] if args.state == "Both" else [args.state]

    all_results = []
    for state in states:
        res = run_comparison(state, args.config, args.epochs, device)
        all_results.extend(res)

    # Save summary CSV
    import pandas as pd
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(all_results)
    out_csv = os.path.join(RESULTS_DIR, f"part3_comparison_{args.state.lower()}_{args.config}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Summary saved: {out_csv}")
    print(df.to_string(index=False))
