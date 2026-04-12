"""
=======================================================================
PART 1 — STEP 4 : DATA PREPROCESSING (V3: Exact pixel_id sequences)
Project: Deep Learning for Crop Classification Using Multi-Source
         Satellite Data (M1 SII 2025/2026 — USTHB)
Paper  : MCTNet — Wang et al., 2024

Pipeline de prétraitement (nouvelle structure exact pixel_id) :
    1. Chargement des CSV concaténés, tri par `pixel_id` et `timestep`
       (Garantit une série temporelle physique parfaite, pas de synthétique).
    2. Reshape vers (N, 36, 10).
    3. Train/Val/Test Split exact (240 train, 60 val, reste en test)
       par classe (Table 2 du papier).
    4. Normalisation Min-Max (fit sur train).
    5. Sauvegarde des tenseurs PyTorch.
=======================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
Donnees    = "./Donnees"
OUTPUT_DIR  = "./preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
N_BANDS = len(BANDS)   # 10
N_DATES = 36           # 36 composites de 10 jours
RANDOM_SEED = 42

# -----------------------------------------------------------------------
# ÉTAPE 1 : CHARGEMENT EXACT PAR PIXEL_ID
# -----------------------------------------------------------------------
def load_and_reconstruct(state, seed=RANDOM_SEED):
    folder  = f"MCTNet_{state.lower()}"
    pattern = os.path.join(Donnees, folder, "*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé pour {state} : {pattern}")

    print(f"  Loading {len(files)} CSV files for {state}...")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # L'ordre chronologique exact est vital
    df.sort_values(by=['pixel_id', 'timestep'], inplace=True)

    pixel_ids = df['pixel_id'].unique()
    N = len(pixel_ids)
    
    print(f"  Total unique pixels: {N}")
    print(f"  Total rows read: {len(df)}")

    if len(df) != N * N_DATES:
        print(f"  ⚠ Attention: {len(df)} rows != {N} * {N_DATES}. Some pixels have missing timesteps.")
        # Filtrer pour ne garder que les pixels qui ont exactement 36 timesteps
        valid_pixels = df.groupby('pixel_id').size()
        valid_pixels = valid_pixels[valid_pixels == N_DATES].index
        df = df[df['pixel_id'].isin(valid_pixels)]
        pixel_ids = df['pixel_id'].unique()
        N = len(pixel_ids)
        print(f"  Filtered down to {N} completely tracked pixels.")

    X_raw = df[BANDS].values.reshape((N, N_DATES, N_BANDS)).astype(np.float32)
    mask = df['valid'].values.reshape((N, N_DATES)).astype(np.float32)
    
    labels = df.groupby('pixel_id')['label'].first().values.astype(np.int64)

    return X_raw, mask, labels

# -----------------------------------------------------------------------
# ÉTAPE 2 : SPLIT FIXE 240 / 60 / TEST (Table 2)
# -----------------------------------------------------------------------
def split_train_val_test(X, mask, labels, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(labels)
    
    n_train_per_class = 240
    n_val_per_class = 60
    
    train_idx, val_idx, test_idx = [], [], []
    
    for cls in unique_classes:
        idx_cls = np.where(labels == cls)[0]
        rng.shuffle(idx_cls)
        
        # S'il y a moins de 300 pixels (ne devrait pas arriver avec les nouvelles extractions)
        if len(idx_cls) <= n_train_per_class + n_val_per_class:
            n_tr = int(len(idx_cls) * 0.8)
            train_idx.extend(idx_cls[:n_tr])
            val_idx.extend(idx_cls[n_tr:])
        else:
            train_idx.extend(idx_cls[:n_train_per_class])
            val_idx.extend(idx_cls[n_train_per_class:n_train_per_class + n_val_per_class])
            test_idx.extend(idx_cls[n_train_per_class + n_val_per_class:])
            
    # Shuffle global des splits
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    splits = {
        "train": (X[train_idx], mask[train_idx], labels[train_idx]),
        "val":   (X[val_idx],   mask[val_idx],   labels[val_idx]),
        "test":  (X[test_idx],  mask[test_idx],  labels[test_idx])
    }
    
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return splits

# -----------------------------------------------------------------------
# ÉTAPE 3 : NORMALISATION (Z-SCORE)
# -----------------------------------------------------------------------
def normalize(X_train, X_val, X_test):
    N_tr, T, C = X_train.shape
    X_tr_2d   = X_train.reshape(-1, C)
    X_val_2d  = X_val.reshape(-1, C)
    X_test_2d = X_test.reshape(-1, C) if len(X_test) > 0 else np.empty((0, C))

    # Calculate Mean and Std on training data only (ignoring masked 0s)
    X_tr_2d_masked = X_tr_2d.copy().astype(np.float32)
    X_tr_2d_masked[X_tr_2d_masked == 0] = np.nan

    band_mean = np.nanmean(X_tr_2d_masked, axis=0) 
    band_std  = np.nanstd(X_tr_2d_masked, axis=0) 
    band_std[band_std == 0] = 1.0  # Prevent division by zero

    def z_score_scale(X_2d):
        if len(X_2d) == 0: return np.empty((0, T, C), dtype=np.float32)
        X_scaled = (X_2d - band_mean) / band_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0) # Restore 0s for masks
        return X_scaled.reshape(-1, T, C).astype(np.float32)

    X_train_norm = z_score_scale(X_tr_2d)
    X_val_norm   = z_score_scale(X_val_2d)
    X_test_norm  = z_score_scale(X_test_2d)

    scaler_dict = {"mean": band_mean, "std": band_std}
    return X_train_norm, X_val_norm, X_test_norm, scaler_dict

def save_tensors(splits, scaler, state):
    save_dir = os.path.join(OUTPUT_DIR, state)
    os.makedirs(save_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        X, mask, y = splits[split_name]
        torch.save({
            "X": torch.tensor(X, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.int64)
        }, os.path.join(save_dir, f"{split_name}.pt"))
        print(f"  Saved {split_name}.pt  → X:{X.shape}")

    torch.save(scaler, os.path.join(save_dir, "meta.pt"))

# -----------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------
class CropDataset(torch.utils.data.Dataset):
    def __init__(self, state, split="train", preprocessed_dir=OUTPUT_DIR):
        path = os.path.join(preprocessed_dir, state, f"{split}.pt")
        data = torch.load(path, weights_only=False)
        self.X    = data["X"]
        self.mask = data["mask"]
        self.y    = data["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]

def get_dataloaders(state, batch_size=64):
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = CropDataset(state, split)
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=(split=="train"), num_workers=0
        )
    return loaders

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4 — DATA PREPROCESSING (V3)")
    print("=" * 60)

    for state in ["Arkansas", "California"]:
        folder = os.path.join(Donnees, f"MCTNet_{state.lower()}")
        if not os.path.exists(folder):
            print(f"\n{state} data not found at {folder}, skipping.")
            continue
            
        print(f"\n{'='*55}\nPREPROCESSING : {state}\n{'='*55}")
        
        # 1. Load data
        X_raw, mask, labels = load_and_reconstruct(state)
        
        # 2. Split
        print("\n[2] Splitting train / val / test...")
        splits_raw = split_train_val_test(X_raw, mask, labels)
        
        # 3. Normalize
        print("\n[3] Normalizing...")
        X_tr, X_val, X_test = splits_raw["train"][0], splits_raw["val"][0], splits_raw["test"][0]
        X_tr_n, X_val_n, X_test_n, scaler = normalize(X_tr, X_val, X_test)
        
        splits_norm = {
            "train": (X_tr_n,   splits_raw["train"][1], splits_raw["train"][2]),
            "val":   (X_val_n,  splits_raw["val"][1],   splits_raw["val"][2]),
            "test":  (X_test_n, splits_raw["test"][1],  splits_raw["test"][2]),
        }
        
        # 4. Save
        print("\n[4] Saving...")
        save_tensors(splits_norm, scaler, state)
        print(f"✅ {state} — preprocessing complete.")