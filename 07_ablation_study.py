"""
=======================================================================
PART 2 — STEP 5 : ABLATION STUDY
Project: Deep Learning for Crop Classification (USTHB)

Training and evaluating MCTNet under 5 environmental configurations:
1. Sentinel-2 only (baseline)
2. Sentinel-2 + climate
3. Sentinel-2 + soil
4. Sentinel-2 + topography
5. Sentinel-2 + all combined
=======================================================================
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Path setup for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mctnet import MCTNet
from utils.losses import FocalLoss, compute_class_weights
from utils.metrics import compute_metrics

# -----------------------------------------------------------------------
# DATASET UTILS (Adapted for ablation folders)
# -----------------------------------------------------------------------
class AblationDataset(Dataset):
    def __init__(self, state, config, split, root="./preprocessed_ablation"):
        path = os.path.join(root, state, config, f"{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tensor file not found: {path}")
        data = torch.load(path, weights_only=False)
        self.X = data["X"]
        self.mask = data["mask"]
        self.y = data["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]

def get_ablation_loaders(state, config, batch_size=32):
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = AblationDataset(state, config, split)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"))
    return loaders

# -----------------------------------------------------------------------
# CONFIGS
# -----------------------------------------------------------------------
CONFIGS = {
    "baseline":   10,
    "s2_climate": 12,
    "s2_soil":    14,
    "s2_topo":    13,
    "all":        19
}

AR_CLASSES = ["Others", "Corn", "Cotton", "Rice", "Soybeans"]
CA_CLASSES = ["Others", "Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios"]

# -----------------------------------------------------------------------
# TRAIN / VAL LOGIC
# -----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for X, mask, y in loader:
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        logits = model(X, mask)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss/total, correct/total, np.array(all_labels), np.array(all_preds)

def train_config(state, config_name, in_channels, epochs=100, device="cuda"):
    print(f"\n🚀 Training: {state} | Configuration: {config_name}")
    
    loaders = get_ablation_loaders(state, config_name)
    class_names = AR_CLASSES if state == "Arkansas" else CA_CLASSES
    n_classes = len(class_names)
    
    model = MCTNet(in_channels=in_channels, n_classes=n_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    weights = compute_class_weights(loaders["train"].dataset.y, n_classes)
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    
    best_acc = 0
    summary_results = {}

    for epoch in range(epochs):
        model.train()
        for X, m, y in loaders["train"]:
            X, m, y = X.to(device), m.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X, m), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        _, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        if val_acc > best_acc:
            best_acc = val_acc
            # Save temporary best
            save_path = f"./checkpoints/ablation/{state}_{config_name}_best.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Val Acc: {val_acc:.4f} (Best: {best_acc:.4f})")

    # Final Eval on Test
    model.load_state_dict(torch.load(f"./checkpoints/ablation/{state}_{config_name}_best.pt"))
    _, test_acc, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    metrics = compute_metrics(y_true, y_pred, class_names)
    
    print(f"  ✅ Finished {config_name}. Test Acc: {test_acc:.4f} | F1: {metrics['macro_f1']:.4f}")
    
    return {
        "state": state,
        "config": config_name,
        "test_accuracy": test_acc,
        "macro_f1": metrics["macro_f1"],
        "kappa": metrics["kappa"]
    }

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    
    os.makedirs("./results/ablation", exist_ok=True)
    
    for state in ["Arkansas", "California"]:
        for cfg_name, in_ch in CONFIGS.items():
            try:
                res = train_config(state, cfg_name, in_ch, epochs=100, device=device)
                results.append(res)
            except Exception as e:
                print(f"  ❌ Failed {cfg_name} for {state}: {e}")
                
    df_results = pd.DataFrame(results)
    df_results.to_csv("./results/ablation/ablation_results.csv", index=False)
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("Summary saved to ./results/ablation/ablation_results.csv")
    print("="*60)
    print(df_results)
