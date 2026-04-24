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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    print(f"\n[TRAIN] {state} | Configuration: {config_name}")
    
    loaders = get_ablation_loaders(state, config_name)
    class_names = AR_CLASSES if state == "Arkansas" else CA_CLASSES
    n_classes = len(class_names)
    
    model = MCTNet(in_channels=in_channels, n_classes=n_classes).to(device)
    
    # Matching the optimized settings from 03_train.py
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0
    patience = 15
    trigger_times = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for X, m, y in loaders["train"]:
            X, m, y = X.to(device), m.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X, m)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step(val_acc)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            trigger_times = 0
            marker = " ★ best"
            # Save best
            save_path = f"./checkpoints/ablation/{state}_{config_name}_best.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            trigger_times += 1
        
        if (epoch+1) % 10 == 0 or epoch == 0 or marker:
            print(f"  Epoch {epoch+1:3d}/{epochs:3d} │ "
                  f"loss: {train_loss:.4f}/{val_loss:.4f} │ "
                  f"acc: {train_acc:.4f}/{val_acc:.4f} │ "
                  f"lr: {optimizer.param_groups[0]['lr']:.6f} {marker}")

        if trigger_times >= patience:
            print(f"  [!] Early stopping at epoch {epoch+1}")
            break

    # Final Eval on Test
    best_path = f"./checkpoints/ablation/{state}_{config_name}_best.pt"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        
    _, test_acc, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    metrics = compute_metrics(y_true, y_pred, class_names)
    
    print(f"  [OK] Finished {config_name}. Test Acc: {test_acc:.4f} | F1: {metrics['F1_macro']:.4f}")
    
    # Save history
    hist_path = f"./checkpoints/ablation/{state}_{config_name}_history.pt"
    torch.save(history, hist_path)
    
    return {
        "state": state,
        "config": config_name,
        "test_accuracy": test_acc,
        "macro_f1": metrics["F1_macro"],
        "kappa": metrics["Kappa"]
    }

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MCTNet Ablation Study")
    parser.add_argument("--state", type=str, default=None, choices=["Arkansas", "California"], help="Run for specific state")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    
    os.makedirs("./results/ablation", exist_ok=True)
    
    states = [args.state] if args.state else ["Arkansas", "California"]
    
    for state in states:
        for cfg_name, in_ch in CONFIGS.items():
            try:
                res = train_config(state, cfg_name, in_ch, epochs=100, device=device)
                results.append(res)
            except Exception as e:
                print(f"  [ERROR] Failed {cfg_name} for {state}: {e}")
                
    if results:
        df_results = pd.DataFrame(results)
        output_path = f"./results/ablation/ablation_results_{args.state.lower() if args.state else 'all'}.csv"
        df_results.to_csv(output_path, index=False)
        print("\n" + "="*60)
        print("ABLATION STUDY COMPLETE")
        print(f"Summary saved to {output_path}")
        print("="*60)
        print(df_results)
