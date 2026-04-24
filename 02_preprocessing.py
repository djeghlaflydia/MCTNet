"""
=======================================================================
PART 2 — STEP 4 : DATA PREPROCESSING (Ablation Version)
Project: Deep Learning for Crop Classification Using Multi-Source
         Satellite Data (M1 SII 2025/2026 — USTHB)

Pipeline de prétraitement multi-configurations :
    1. Chargement des CSV fusionnés (Donnees_Merged).
    2. Définition des 5 configurations d'ablation.
    3. Normalisation et sauvegarde des tenseurs par configuration.
=======================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import torch

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
DATA_DIR    = "./Donnees_Merged"
OUTPUT_DIR  = "./preprocessed_ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_DATES = 36
RANDOM_SEED = 42

# Define Feature Sets
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CLIMATE  = ["temp", "precip"]
SOIL     = ["clay", "sand", "org_carbon", "ph"]
TOPO     = ["elevation", "slope", "aspect"]

CONFIGS = {
    "baseline":   S2_BANDS,
    "s2_climate": S2_BANDS + CLIMATE,
    "s2_soil":    S2_BANDS + SOIL,
    "s2_topo":    S2_BANDS + TOPO,
    "all":        S2_BANDS + CLIMATE + SOIL + TOPO
}

# -----------------------------------------------------------------------
# ÉTAPE 1 : CHARGEMENT
# -----------------------------------------------------------------------
def load_and_reconstruct(state, feature_cols, seed=RANDOM_SEED):
    folder  = f"MCTNet_{state.lower()}"
    pattern = os.path.join(DATA_DIR, folder, "*.csv")
    files   = sorted(glob.glob(pattern))
    
    if not files:
        # Fallback to original Donnees if baseline and no merged yet
        if set(feature_cols) == set(S2_BANDS):
            print(f"  ℹ No merged data for {state}, falling back to original Donnees for baseline.")
            pattern = os.path.join("./Donnees", folder, "*.csv")
            files = sorted(glob.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"Aucun CSV trouvé pour {state} : {pattern}")

    print(f"  Loading {len(files)} files for {state}...")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Sort to ensure temporal sequence
    df.sort_values(by=['pixel_id', 'timestep'], inplace=True)
    
    # Filter complete pixels
    valid_pixels = df.groupby('pixel_id').size()
    valid_pixels = valid_pixels[valid_pixels == N_DATES].index
    df = df[df['pixel_id'].isin(valid_pixels)]
    
    pixel_ids = df['pixel_id'].unique()
    N = len(pixel_ids)
    C = len(feature_cols)

    X_raw = df[feature_cols].values.reshape((N, N_DATES, C)).astype(np.float32)
    mask = df['valid'].values.reshape((N, N_DATES)).astype(np.float32)
    labels = df.groupby('pixel_id')['label'].first().values.astype(np.int64)

    return X_raw, mask, labels

# -----------------------------------------------------------------------
# ÉTAPE 2 : SPLIT FIXE 240 / 60 / TEST
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
        
        if len(idx_cls) <= n_train_per_class + n_val_per_class:
            n_tr = int(len(idx_cls) * 0.8)
            train_idx.extend(idx_cls[:n_tr])
            val_idx.extend(idx_cls[n_tr:])
        else:
            train_idx.extend(idx_cls[:n_train_per_class])
            val_idx.extend(idx_cls[n_train_per_class:n_train_per_class + n_val_per_class])
            test_idx.extend(idx_cls[n_train_per_class + n_val_per_class:])
            
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    return {
        "train": (X[train_idx], mask[train_idx], labels[train_idx]),
        "val":   (X[val_idx],   mask[val_idx],   labels[val_idx]),
        "test":  (X[test_idx],  mask[test_idx],  labels[test_idx])
    }

# -----------------------------------------------------------------------
# ÉTAPE 3 : NORMALISATION (Z-SCORE)
# -----------------------------------------------------------------------
def normalize(splits):
    X_train, mask_train, _ = splits["train"]
    N_tr, T, C = X_train.shape
    
    X_tr_2d = X_train.reshape(-1, C)
    
    # Calculate Mean and Std ignoring 0s (masked)
    X_tr_masked = X_tr_2d.copy()
    X_tr_masked[X_tr_masked == 0] = np.nan
    mean = np.nanmean(X_tr_masked, axis=0)
    std  = np.nanstd(X_tr_masked, axis=0)
    std[std == 0] = 1.0

    def scale(X):
        if len(X) == 0: return X
        N, T, C = X.shape
        X_2d = X.reshape(-1, C)
        X_scaled = (X_2d - mean) / std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        return X_scaled.reshape(N, T, C).astype(np.float32)

    return {
        k: (scale(v[0]), v[1], v[2]) for k, v in splits.items()
    }, {"mean": mean, "std": std}

# -----------------------------------------------------------------------
# ÉTAPE 4 : SAUVEGARDE
# -----------------------------------------------------------------------
def save_config(splits, meta, state, config_name):
    path = os.path.join(OUTPUT_DIR, state, config_name)
    os.makedirs(path, exist_ok=True)
    
    for split, (X, mask, y) in splits.items():
        torch.save({"X": torch.tensor(X), "mask": torch.tensor(mask), "y": torch.tensor(y)}, 
                   os.path.join(path, f"{split}.pt"))
    
    torch.save(meta, os.path.join(path, "meta.pt"))
    print(f"    Saved {config_name} tensors for {state}")

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ABLATION PREPROCESSING COMPLETE")
    print("="*60)

    for state in ["Arkansas", "California"]:
        print(f"\nProcessing {state}...")
        
        for config_name, feature_cols in CONFIGS.items():
            print(f"  -> Config: {config_name} ({len(feature_cols)} features)")
            
            try:
                # 1. Load
                X, mask, y = load_and_reconstruct(state, feature_cols)
                
                # 2. Split
                splits = split_train_val_test(X, mask, y)
                
                # 3. Normalize
                splits_norm, meta = normalize(splits)
                
                # 4. Save
                save_config(splits_norm, meta, state, config_name)
                
            except Exception as e:
                print(f"    [ERROR] Error in {config_name}: {e}")

    print("\nPreprocessing complete.")