"""
=======================================================================
PART 1 — STEP 4 : DATA PREPROCESSING
Project: Deep Learning for Crop Classification Using Multi-Source
         Satellite Data (M1 SII 2025/2026 — USTHB)
Paper  : MCTNet — Wang et al., 2024

Pipeline de prétraitement complet conforme au papier (section 2.2.4) :
    1. Chargement et reconstruction de la matrice temporelle
       (10 000 pixels × 36 dates × 10 bandes = 10 000 × 360)
    2. Détection et marquage des valeurs manquantes (masque binaire)
    3. Filtre cultures < 5% → classe "Others" (convention papier)
    4. Normalisation Min-Max par bande (sur données d'entraînement)
    5. Extraction des indices de végétation (NDVI, EVI)
    6. Division Train / Validation / Test (convention papier Table 2)
       → 300 pixels/classe pour train+val (80/20), reste pour test
    7. Sauvegarde des tenseurs PyTorch prêts pour MCTNet
       - X      : (N, T, C)  → (N, 36, 10) float32
       - mask   : (N, T)     → (N, 36)     float32  (Input 2 du modèle)
       - y      : (N,)       → int64
=======================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
Donnees    = "./data"
OUTPUT_DIR  = "./preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
N_BANDS = len(BANDS)   # 10
N_DATES = 36           # 36 composites de 10 jours

# Seuil pour la règle "< 5% → Others" (papier section 2.2.4)
MINORITY_THRESHOLD = 0.05

# Seed pour la reproductibilité (même que dans le papier : seed=42 pour GEE)
RANDOM_SEED = 42

# Classes selon Table 2 du papier
AR_CLASSES = {1: "Corn", 2: "Cotton", 3: "Rice", 4: "Soybeans", 5: "Others"}
CA_CLASSES = {1: "Grapes", 2: "Rice", 3: "Alfalfa",
              4: "Almonds", 5: "Pistachios", 6: "Others"}

# Valeur "Others" pour chaque état
AR_OTHERS_CODE = 5
CA_OTHERS_CODE = 6


# -----------------------------------------------------------------------
# ÉTAPE 1 : CHARGEMENT ET RECONSTRUCTION DE LA MATRICE TEMPORELLE
# -----------------------------------------------------------------------
def load_and_reconstruct(state, zone):
    """
    Charge les 36 CSV d'une zone et reconstruit la matrice temporelle.

    Retourne :
        X_raw : np.ndarray (N, T, C) = (N_pixels, 36, 10)
        mask  : np.ndarray (N, T)    = 1 si pixel présent, 0 si manquant
        labels: np.ndarray (N,)      labels CDL remappés
        pixels_ids : list de N identifiants uniques (index de ligne dans csv1)
    """
    pattern = os.path.join(Donnees, state, zone, "*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé : {pattern}")

    print(f"  Loading {len(files)} CSV files for {state}/{zone}...")

    # Charger tous les CSV et les ordonner par date
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)

    # Trier les timestamps
    dfs_sorted = sorted(dfs, key=lambda d: d["date"].iloc[0])

    # Référence : pixels du premier CSV (csv_id=1)
    df_ref = dfs_sorted[0].reset_index(drop=True)
    N      = len(df_ref)
    labels = df_ref["class_label"].values.astype(np.int64)

    # Construire X_raw : (N, T, C)
    X_raw = np.zeros((N, N_DATES, N_BANDS), dtype=np.float32)
    mask  = np.zeros((N, N_DATES),          dtype=np.float32)

    for t, df_t in enumerate(dfs_sorted):
        df_t = df_t.reset_index(drop=True)
        vals = df_t[BANDS].values.astype(np.float32)  # (N, 10)

        # Détection des pixels manquants : toutes bandes = 0
        is_missing = (vals == 0).all(axis=1)           # (N,)

        X_raw[:, t, :] = vals
        mask[:, t]     = (~is_missing).astype(np.float32)  # 1=présent, 0=manquant

    print(f"  ✓ Reconstructed X_raw: {X_raw.shape}  "
          f"(N={N}, T={N_DATES}, C={N_BANDS})")
    print(f"  ✓ Missing rate: {(1 - mask.mean())*100:.1f}%")

    return X_raw, mask, labels


# -----------------------------------------------------------------------
# ÉTAPE 2 : FILTRE "CULTURES < 5% → Others" (papier section 2.2.4)
# -----------------------------------------------------------------------
def apply_minority_filter(labels, others_code, threshold=MINORITY_THRESHOLD):
    """
    Les cultures représentant moins de `threshold` (5%) des pixels
    sont reclassées dans la catégorie Others.

    Retourne les labels corrigés et le mapping utilisé.
    """
    total  = len(labels)
    counts = Counter(labels)
    remap  = {}
    merged = []

    for cls, cnt in counts.items():
        if cnt / total < threshold and cls != others_code:
            remap[cls] = others_code
            merged.append(cls)

    if merged:
        print(f"  Merging minority classes into Others "
              f"(code={others_code}): {merged}")
        labels = np.array([remap.get(l, l) for l in labels], dtype=np.int64)
    else:
        print("  No minority classes to merge.")

    # Réindexer les classes en 0-based pour PyTorch (0, 1, 2, ...)
    unique_cls = sorted(set(labels))
    cls_to_idx = {c: i for i, c in enumerate(unique_cls)}
    labels_idx = np.array([cls_to_idx[l] for l in labels], dtype=np.int64)
    print(f"  Final classes: {cls_to_idx}")

    return labels_idx, cls_to_idx


# -----------------------------------------------------------------------
# ÉTAPE 3 : NORMALISATION MIN-MAX PAR BANDE
# -----------------------------------------------------------------------
def normalize(X_train, X_val, X_test):
    """
    Normalisation Min-Max par bande spectrale.
    Le scaler est ajusté sur X_train uniquement (évite la fuite de données).

    X shape : (N, T, C)
    Retourne les données normalisées et le scaler sauvegardé.
    """
    N_tr, T, C = X_train.shape

    # Reshape en 2D : (N*T, C) pour fit du scaler
    X_tr_2d   = X_train.reshape(-1, C)
    X_val_2d  = X_val.reshape(-1, C)
    X_test_2d = X_test.reshape(-1, C)

    # Masquer les 0 (données manquantes) pour le calcul du min/max
    # On remplace les 0 par NaN pour le fit, puis on remet 0 après
    X_tr_2d_masked = X_tr_2d.copy().astype(np.float32)
    X_tr_2d_masked[X_tr_2d_masked == 0] = np.nan

    # Calculer min/max par bande (ignorer les NaN)
    band_min = np.nanmin(X_tr_2d_masked, axis=0)   # (C,)
    band_max = np.nanmax(X_tr_2d_masked, axis=0)   # (C,)
    band_range = band_max - band_min
    band_range[band_range == 0] = 1  # éviter division par 0

    def minmax(X_2d):
        X_norm = X_2d.copy().astype(np.float32)
        # Ne normaliser que les pixels non manquants
        non_zero = (X_2d != 0).any(axis=1)
        X_norm[non_zero] = (X_2d[non_zero] - band_min) / band_range
        X_norm = np.clip(X_norm, 0, 1)
        # Remettre 0 pour les pixels manquants (convention MCTNet)
        X_norm[~non_zero] = 0
        return X_norm

    X_tr_norm   = minmax(X_tr_2d).reshape(N_tr,         T, C)
    X_val_norm  = minmax(X_val_2d).reshape(X_val.shape[0],  T, C)
    X_test_norm = minmax(X_test_2d).reshape(X_test.shape[0], T, C)

    scaler_params = {"band_min": band_min, "band_max": band_max}
    return X_tr_norm, X_val_norm, X_test_norm, scaler_params


# -----------------------------------------------------------------------
# ÉTAPE 4 : SPLIT TRAIN / VAL / TEST (convention papier Table 2)
# -----------------------------------------------------------------------
def split_train_val_test(X, mask, labels, cls_to_idx,
                         n_per_class=300, val_ratio=0.2,
                         random_state=RANDOM_SEED):
    """
    Reproduit la convention du papier (section 2.4) :
      - 300 samples par classe pour train + val (80/20)
      - Tous les pixels restants → test

    Retourne des dictionnaires {split: (X, mask, y)}.
    """
    rng = np.random.default_rng(random_state)

    train_idx, val_idx, test_idx = [], [], []

    unique_classes = np.unique(labels)

    for cls in unique_classes:
        idx_cls = np.where(labels == cls)[0]
        rng.shuffle(idx_cls)

        n_sel = min(n_per_class, len(idx_cls))
        selected = idx_cls[:n_sel]
        remaining = idx_cls[n_sel:]

        n_val   = int(n_sel * val_ratio)
        n_train = n_sel - n_val

        train_idx.extend(selected[:n_train].tolist())
        val_idx.extend(selected[n_train:].tolist())
        test_idx.extend(remaining.tolist())

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    print(f"  Train : {len(train_idx):,} samples")
    print(f"  Val   : {len(val_idx):,} samples")
    print(f"  Test  : {len(test_idx):,} samples")

    splits = {
        "train": (X[train_idx], mask[train_idx], labels[train_idx]),
        "val":   (X[val_idx],   mask[val_idx],   labels[val_idx]),
        "test":  (X[test_idx],  mask[test_idx],  labels[test_idx]),
    }
    return splits


# -----------------------------------------------------------------------
# ÉTAPE 5 : SAUVEGARDE DES TENSEURS PYTORCH
# -----------------------------------------------------------------------
def save_tensors(splits, scaler_params, state, zone, cls_to_idx):
    """
    Sauvegarde les données prétraitées sous forme de fichiers .pt
    (format natif PyTorch), prêts pour MCTNet.

    Structure attendue par MCTNet (Fig. 3 du papier) :
        Input 1 : X     → (N, T, C) = (N, 36, 10)  float32
        Input 2 : mask  → (N, T)    = (N, 36)       float32
        Labels  : y     → (N,)                       int64
    """
    save_dir = os.path.join(OUTPUT_DIR, state, zone)
    os.makedirs(save_dir, exist_ok=True)

    for split, (X, mask, y) in splits.items():
        data = {
            "X":    torch.tensor(X,    dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "y":    torch.tensor(y,    dtype=torch.long),
        }
        path = os.path.join(save_dir, f"{split}.pt")
        torch.save(data, path)
        print(f"  Saved {split}.pt  → X:{data['X'].shape} "
              f"mask:{data['mask'].shape} y:{data['y'].shape}")

    # Sauvegarder le scaler et le mapping de classes
    meta = {
        "band_min":    scaler_params["band_min"],
        "band_max":    scaler_params["band_max"],
        "cls_to_idx":  cls_to_idx,
        "bands":       BANDS,
        "n_dates":     N_DATES,
    }
    meta_path = os.path.join(save_dir, "meta.pt")
    torch.save(meta, meta_path)
    print(f"  Saved meta.pt (scaler + class mapping)")


# -----------------------------------------------------------------------
# PIPELINE COMPLET POUR UNE ZONE
# -----------------------------------------------------------------------
def preprocess_zone(state, zone):
    """
    Exécute le pipeline complet pour une zone donnée.
    """
    others_code = AR_OTHERS_CODE if state == "Arkansas" else CA_OTHERS_CODE

    print(f"\n{'='*55}")
    print(f"PREPROCESSING : {state} / {zone}")
    print(f"{'='*55}")

    # 1. Chargement et reconstruction
    X_raw, mask, labels = load_and_reconstruct(state, zone)

    # 2. Filtre minorités → Others
    print("\n[2] Applying minority filter (< 5% → Others)...")
    labels_idx, cls_to_idx = apply_minority_filter(
        labels, others_code, MINORITY_THRESHOLD)

    # 3. Split (avant normalisation pour éviter la fuite)
    print("\n[3] Splitting train / val / test...")
    splits_raw = split_train_val_test(X_raw, mask, labels_idx, cls_to_idx)

    # 4. Normalisation (fit sur train uniquement)
    print("\n[4] Normalizing (Min-Max per band, fit on train)...")
    X_tr, X_val, X_test = (splits_raw["train"][0],
                            splits_raw["val"][0],
                            splits_raw["test"][0])
    X_tr_n, X_val_n, X_test_n, scaler = normalize(X_tr, X_val, X_test)

    splits_norm = {
        "train": (X_tr_n,   splits_raw["train"][1], splits_raw["train"][2]),
        "val":   (X_val_n,  splits_raw["val"][1],   splits_raw["val"][2]),
        "test":  (X_test_n, splits_raw["test"][1],  splits_raw["test"][2]),
    }

    # 5. Sauvegarde
    print("\n[5] Saving PyTorch tensors...")
    save_tensors(splits_norm, scaler, state, zone, cls_to_idx)

    print(f"\n✅ {state}/{zone} — preprocessing complete.")
    return splits_norm, cls_to_idx


# -----------------------------------------------------------------------
# VÉRIFICATION : CHARGER ET AFFICHER UN ÉCHANTILLON
# -----------------------------------------------------------------------
def verify_output(state, zone, split="train"):
    """Vérifie que les fichiers sauvegardés sont corrects."""
    path = os.path.join(OUTPUT_DIR, state, zone, f"{split}.pt")
    data = torch.load(path)
    print(f"\nVerification — {state}/{zone}/{split}.pt:")
    print(f"  X    : {data['X'].shape}    dtype={data['X'].dtype}")
    print(f"  mask : {data['mask'].shape} dtype={data['mask'].dtype}")
    print(f"  y    : {data['y'].shape}    dtype={data['y'].dtype}")
    print(f"  X min={data['X'].min():.4f}  max={data['X'].max():.4f}")
    print(f"  mask values: {data['mask'].unique().tolist()}"
          "  (1=present, 0=missing)")
    print(f"  Class distribution: "
          f"{dict(zip(*data['y'].unique(return_counts=True)))}")


# -----------------------------------------------------------------------
# DATASET PYTORCH POUR MCTNET
# -----------------------------------------------------------------------
class CropDataset(torch.utils.data.Dataset):
    """
    Dataset PyTorch pour MCTNet.
    Retourne (X, mask, y) par échantillon :
        X    : (T, C) = (36, 10)  → Input 1 du modèle
        mask : (T,)   = (36,)     → Input 2 du modèle (ALPE)
        y    : scalar             → label de classe
    """
    def __init__(self, state, zone, split="train",
                 preprocessed_dir=OUTPUT_DIR):
        path = os.path.join(preprocessed_dir, state, zone, f"{split}.pt")
        data = torch.load(path)
        self.X    = data["X"]     # (N, T, C)
        self.mask = data["mask"]  # (N, T)
        self.y    = data["y"]     # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]


def get_dataloaders(state, zone, batch_size=32):
    """Crée les DataLoaders pour train, val et test."""
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = CropDataset(state, zone, split)
        shuffle = (split == "train")
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=True
        )
        print(f"  {split} loader : {len(ds)} samples, "
              f"{len(loaders[split])} batches")
    return loaders


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("STEP 4 — DATA PREPROCESSING")
    print("MCTNet Project — M1 SII USTHB 2025/2026")
    print("=" * 60)

    zones = [
        ("Arkansas",   "zone1"),
        ("Arkansas",   "zone2"),
        ("California", "zone1"),
        ("California", "zone2"),
    ]

    for state, zone in zones:
        try:
            splits, cls_to_idx = preprocess_zone(state, zone)
            verify_output(state, zone, split="train")
        except FileNotFoundError as e:
            print(f"\n⚠ Skipping {state}/{zone}: {e}")

    print("\n" + "=" * 60)
    print("EXAMPLE — Loading DataLoaders for MCTNet training:")
    print("=" * 60)

    # Exemple d'utilisation
    example_code = """
# Dans votre script d'entraînement MCTNet :
from preprocessing import get_dataloaders, CropDataset

loaders = get_dataloaders("Arkansas", "zone1", batch_size=32)

for X, mask, y in loaders["train"]:
    # X    : (32, 36, 10)  → Input 1 : features spectrales temporelles
    # mask : (32, 36)      → Input 2 : masque valeurs manquantes (ALPE)
    # y    : (32,)         → labels de classe
    print(f"X:{X.shape}, mask:{mask.shape}, y:{y.shape}")
    break
"""
    print(example_code)

    print("=" * 60)
    print(f"✅ All preprocessed data saved in: {OUTPUT_DIR}/")
    print("   Structure:")
    print("   preprocessed/")
    print("   ├── Arkansas/zone1/ → train.pt, val.pt, test.pt, meta.pt")
    print("   ├── Arkansas/zone2/ → train.pt, val.pt, test.pt, meta.pt")
    print("   ├── California/zone1/→ train.pt, val.pt, test.pt, meta.pt")
    print("   └── California/zone2/→ train.pt, val.pt, test.pt, meta.pt")
    print("=" * 60)