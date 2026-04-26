"""
=======================================================================
PART 1 — STEP 6 : TRAINING MCTNet
Project: Deep Learning for Crop Classification Using Multi-Source
         Satellite Data (M1 SII 2025/2026 — USTHB)
Paper  : MCTNet — Wang et al., 2024

Training configuration (faithful to paper):
    Optimizer  : Adam
    LR         : 0.001
    Batch size : 32
    Epochs     : 200
    Loss       : Cross Entropy
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
import importlib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mctnet import MCTNet
from utils.losses import FocalLoss, compute_class_weights
from utils.metrics import compute_metrics, print_metrics

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
PREPROCESSED_DIR = "./preprocessed_ablation"
CONFIG_NAME = "baseline"  # Default to baseline (10 channels)
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

# -----------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------
class CropDataset(Dataset):
    def __init__(self, state, split):
        path = os.path.join(PREPROCESSED_DIR, state, CONFIG_NAME, f"{split}.pt")
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

def get_dataloaders(state, batch_size):
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = CropDataset(state, split)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
    return loaders

AR_CLASSES = {0: "Others", 1: "Corn", 2: "Cotton", 3: "Rice", 4: "Soybeans"}
CA_CLASSES = {0: "Others", 1: "Grapes", 2: "Rice", 3: "Alfalfa",
              4: "Almonds", 5: "Pistachios"}

# -----------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate. Returns average loss, accuracy, all predictions & labels."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X, mask, y in loader:
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)

        logits = model(X, mask)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    if total == 0:
        return 0.0, 0.0, np.array([]), np.array([])

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels))


def train(
    model, train_loader, val_loader, test_loader, num_epochs, device,
    state, class_names, test_ds=None, val_ds=None
):
    save_dir = os.path.join(CHECKPOINT_DIR, state)
    result_dir = os.path.join(RESULTS_DIR, state)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    best_val_acc = 0.0
    patience = 15
    trigger_times = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Paramètres article (Table 3)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    print(f"\n  Optimizer: Adam (WD: 0.05), Loss: CrossEntropy")
    print(f"  Early Stopping: {patience} epochs, Scheduler: ReduceLROnPlateau")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, mask_batch, y_batch in train_loader:
            X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch, mask_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Scheduler step based on validation accuracy
        scheduler.step(val_acc)

        # Save checkpoint and Early Stopping logic
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
            marker = " ★ best"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, os.path.join(save_dir, "best_model.pt"))
        else:
            trigger_times += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or marker:
            print(f"  Epoch {epoch+1:3d}/{num_epochs:3d} │ "
                  f"loss: {train_loss:.4f}/{val_loss:.4f} │ "
                  f"acc: {train_acc:.4f}/{val_acc:.4f} │ "
                  f"lr: {optimizer.param_groups[0]['lr']:.6f} {marker}")

        if trigger_times >= patience:
            print(f"\n  [!] Early stopping at epoch {epoch+1}")
            break

    print(f"\n  ✅ Best val accuracy: {best_val_acc:.4f}")
    torch.save(history, os.path.join(save_dir, "history.pt"))

    # ========================== EVALUATION ==========================
    print(f"\n{'='*60}\n  FINAL EVALUATION ON TEST SET\n{'='*60}")

    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_loader = test_loader if test_ds and len(test_ds) > 0 else val_loader
    eval_name = "Test" if test_ds and len(test_ds) > 0 else "Val"

    _, _, y_pred, y_true = validate(model, eval_loader, criterion, device)

    if len(y_true) > 0:
        metrics = compute_metrics(y_true, y_pred, class_names)
        print_metrics(metrics, title=f"{state} — {eval_name} Results")

        torch.save({
            "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
            "y_true": y_true,
            "y_pred": y_pred,
            "class_names": class_names,
            "report": metrics["classification_report"],
        }, os.path.join(result_dir, "test_metrics.pt"))
    else:
        print("\n  ⚠ No data available for final evaluation.")

    print(f"\n  📁 Checkpoint saved: {save_dir}/best_model.pt")
    print(f"  📁 Results saved:    {result_dir}/")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTNet Training")
    parser.add_argument("--state", type=str, default="Both",
                        choices=["Arkansas", "California", "Both"],
                        help="Study area (default: Both)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu (default: auto)")
    args = parser.parse_args()

    states_to_process = ["Arkansas", "California"] if args.state == "Both" else [args.state]

    for state in states_to_process:
        print("\n" + "=" * 60)
        print(f" PROCESSING STATE: {state.upper()}")
        print("=" * 60)

        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"\n🖥️  Device: {device.type}\n")

        print(f"📂 Loading preprocessed data: {state}")
        try:
            loaders = get_dataloaders(state, batch_size=args.batch_size)
        except Exception as e:
            print(f"Failed to load dataloaders for {state}: {e}")
            continue

        train_ds = CropDataset(state, "train")
        val_ds   = CropDataset(state, "val")
        test_ds  = CropDataset(state, "test")

        print(f"  Train : {len(train_ds)} samples ({len(loaders['train'])} batches)")
        print(f"  Val   : {len(val_ds)} samples ({len(loaders['val'])} batches)")
        print(f"  Test  : {len(test_ds)} samples ({len(loaders['test'])} batches)")

        if state == "Arkansas":
            class_names = [AR_CLASSES[i] for i in range(5)]
        else:
            class_names = [CA_CLASSES[i] for i in range(6)]

        print(f"  Classes: {len(class_names)} → {class_names}")

        model = MCTNet(
            in_channels=10,
            n_classes=len(class_names),
            n_heads=5,
            ffn_factor=4,
            dropout=0.2
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n🧠 MCTNet: {total_params:,} trainable parameters")

        train(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"],
            num_epochs=args.epochs,
            device=device,
            state=state,
            class_names=class_names,
            test_ds=test_ds,
            val_ds=val_ds
        )
