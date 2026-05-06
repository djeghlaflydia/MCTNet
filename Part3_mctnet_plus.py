"""
=============================================================================
PARTIE 3 - Modèle Amélioré : MCTNet+ avec Cross-Modal Attention Fusion (CMAF)
=============================================================================

Justification de l'architecture :
- En Arkansas, la concaténation naïve de TOUTES les covariables DÉGRADE les
  performances (all: OA=0.804 vs baseline: OA=0.864). Cela suggère que certaines
  covariables introduisent du bruit selon la région.
- En Californie, "all" aide (+3.7%) mais la topographie et le climat ont des
  effets indépendants forts.
- Solution : Cross-Modal Attention Fusion (CMAF) qui apprend à PONDÉRER
  dynamiquement chaque modalité de covariable plutôt que de les concaténer
  naïvement. Chaque groupe de covariables est encodé séparément, puis une
  attention croisée décide de leur importance relative par rapport aux
  features Sentinel-2.

Architecture MCTNet+ :
  1. Branche principale : MCTNet original (CNN + Transformer multi-stages)
  2. Branche covariables : encodeur MLP séparé par groupe (climat, sol, topo)
  3. CMAF : Cross-Modal Attention entre features S2 et chaque groupe
  4. Fusion finale par gating adaptatif
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# Hyperparamètres d'entraînement
BATCH_SIZE   = 32
EPOCHS       = 200
LR           = 0.001
N_STAGES     = 3
N_HEAD       = 4
KERNEL_SIZE  = 3
N_BANDS      = 10
N_TIMESTEPS  = 36

# Chemins (à adapter selon votre structure de fichiers)
DATA_DIR     = "./data"
RESULTS_DIR  = "./part3_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

STATES = ["Arkansas", "California"]

# Dimensions des groupes de covariables (basé sur la config "all" de 02_preprocessing.py)
COVARIATE_DIMS = {
    "climate": 2,    # temp, precip
    "soil":    4,    # clay, sand, org_carbon, ph
    "topo":    3,    # elevation, slope, aspect
}

# ============================================================
# DATASET
# ============================================================

class CropDatasetPart3(Dataset):
    """
    Dataset qui charge les features S2 + les covariables par groupe.
    Adapté pour utiliser les données générées par 02_preprocessing.py
    
    Charge le fichier preprocessed_ablation/{state}/all/{split}.pt
    et sépare les tenseurs concaténés en :
      - s2 (10 bandes)
      - climate (2 bandes)
      - soil (4 bandes)
      - topo (3 bandes)
    """
    def __init__(self, state, split="train", data_dir="./preprocessed_ablation"):
        self.state = state
        self.split = split
        
        path = os.path.join(data_dir, state, "all", f"{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Données introuvables: {path}. Lancez 02_preprocessing.py d'abord.")
            
        data = torch.load(path, weights_only=False)
        self.X = data["X"]       # shape (N, T, 19)
        self.mask = data["mask"] # shape (N, T)
        self.y = data["y"]       # shape (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # On extrait les features du tenseur concaténé
        # Indices: S2(0-9), Climate(10-11), Soil(12-15), Topo(16-18)
        X_i = self.X[idx]  # (T, 19)
        
        return {
            "s2":      X_i[:, 0:10],          # (T, 10)
            "mask":    self.mask[idx],        # (T,)
            "climate": X_i[0, 10:12],         # (2,)
            "soil":    X_i[0, 12:16],         # (4,)
            "topo":    X_i[0, 16:19],         # (3,)
            "label":   self.y[idx],           # scalaire
        }


# ============================================================
# BLOCS DE BASE - MCTNet original (reproduit depuis Partie 1)
# ============================================================

class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., 2020)."""
    def __init__(self, channels, k=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # x : (B, C, T)
        y = self.avg_pool(x)          # (B, C, 1)
        y = y.transpose(-1, -2)       # (B, 1, C)
        y = self.conv(y)              # (B, 1, C)
        y = y.transpose(-1, -2)       # (B, C, 1)
        y = self.sigmoid(y)
        return x * y


class ALPE(nn.Module):
    """Attention-based Learnable Positional Encoding."""
    def __init__(self, d_model, max_len=36, kernel_size=3):
        super().__init__()
        # Encodage positionnel sinusoïdal fixe
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # (T, d_model)

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                              padding=kernel_size//2)
        self.eca  = ECA(d_model)

    def forward(self, x, mask):
        """
        x    : (B, T, d_model)
        mask : (B, T) - 1 si valide, 0 si manquant
        """
        B, T, D = x.shape
        pe = self.pe[:T, :].unsqueeze(0).expand(B, -1, -1)  # (B, T, D)
        # Masquage des positions manquantes
        pe = pe * mask.unsqueeze(-1)                          # (B, T, D)
        pe = pe.transpose(1, 2)                               # (B, D, T)
        pe = self.conv(pe)                                    # (B, D, T)
        pe = self.eca(pe)                                     # (B, D, T)
        pe = pe.transpose(1, 2)                               # (B, T, D)
        return x + pe


class TransformerSubModule(nn.Module):
    """Transformer sub-module avec ALPE optionnel."""
    def __init__(self, d_model, n_head, use_alpe=False, max_len=36, kernel_size=3):
        super().__init__()
        self.use_alpe = use_alpe
        if use_alpe:
            self.alpe = ALPE(d_model, max_len=max_len, kernel_size=kernel_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=d_model*4, dropout=0.1,
            batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x, mask=None):
        if self.use_alpe and mask is not None:
            x = self.alpe(x, mask)
        return self.encoder(x)


class CNNSubModule(nn.Module):
    """CNN sub-module 1D avec connexion résiduelle."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels,  out_channels, kernel_size, padding=padding)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.skip  = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x : (B, C, T)
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class CTFusionModule(nn.Module):
    """Module CTFusion = CNN sub-module || Transformer sub-module + concat."""
    def __init__(self, in_channels, out_channels, n_head, kernel_size=3,
                 use_alpe=False, max_len=36):
        super().__init__()
        self.cnn   = CNNSubModule(in_channels, out_channels, kernel_size)
        self.trans = TransformerSubModule(in_channels, n_head, use_alpe,
                                          max_len=max_len, kernel_size=kernel_size)
        self.proj  = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, mask=None):
        """
        x    : (B, T, C)
        mask : (B, T)
        """
        # CNN path : opère sur (B, C, T)
        cnn_out   = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # (B, T, out_C)
        # Transformer path
        trans_out = self.trans(x, mask)                             # (B, T, in_C)
        # Fusion
        concat = torch.cat([cnn_out, trans_out], dim=-1)            # (B, T, out_C+in_C)
        return F.relu(self.proj(concat))                            # (B, T, out_C)


# ============================================================
# BRANCHE COVARIABLE + CMAF
# ============================================================

class CovariateEncoder(nn.Module):
    """
    Encodeur MLP pour un groupe de covariables statiques.
    Produit un vecteur de taille d_cov.
    """
    def __init__(self, in_dim, d_cov=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_cov * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_cov * 2, d_cov),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)  # (B, d_cov)


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion (CMAF).
    
    Mécanisme :
    - La feature S2 (query) interroge chaque groupe de covariables (key/value).
    - Un score d'attention apprend à pondérer dynamiquement l'importance
      de chaque groupe selon le contexte S2, évitant la dégradation observée
      avec la concaténation naïve.
    
    Référence : inspiré de cross-attention dans ViLBERT, CLIP, etc.
    """
    def __init__(self, d_s2, d_cov, n_groups=3):
        super().__init__()
        self.d_s2    = d_s2
        self.d_cov   = d_cov
        self.n_groups = n_groups

        # Projection query depuis S2
        self.q_proj = nn.Linear(d_s2, d_cov)
        # Projection key/value depuis chaque groupe (partagée)
        self.k_proj = nn.Linear(d_cov, d_cov)
        self.v_proj = nn.Linear(d_cov, d_cov)
        # Gating final
        self.gate   = nn.Sequential(
            nn.Linear(d_s2 + d_cov, d_s2),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_s2 + d_cov, d_s2)
        self.norm     = nn.LayerNorm(d_s2)

    def forward(self, s2_feat, cov_feats):
        """
        s2_feat  : (B, d_s2)           - features S2 agrégées
        cov_feats: list of (B, d_cov)  - encodage de chaque groupe
        """
        # Stack covariables → (B, n_groups, d_cov)
        cov = torch.stack(cov_feats, dim=1)

        # Query depuis S2
        q = self.q_proj(s2_feat).unsqueeze(1)   # (B, 1, d_cov)
        # Key / Value depuis covariables
        k = self.k_proj(cov)                     # (B, n_groups, d_cov)
        v = self.v_proj(cov)                     # (B, n_groups, d_cov)

        # Attention scores
        scale  = self.d_cov ** 0.5
        scores = (q @ k.transpose(-1, -2)) / scale  # (B, 1, n_groups)
        attn   = F.softmax(scores, dim=-1)           # (B, 1, n_groups)
        # Agrégation pondérée
        context = (attn @ v).squeeze(1)              # (B, d_cov)

        # Fusion avec gating adaptatif
        combined = torch.cat([s2_feat, context], dim=-1)  # (B, d_s2+d_cov)
        gate_val = self.gate(combined)                     # (B, d_s2)
        out      = self.out_proj(combined)                 # (B, d_s2)
        # Residual gated
        fused = gate_val * out + (1 - gate_val) * s2_feat
        return self.norm(fused), attn.squeeze(1)           # (B, d_s2), (B, n_groups)


# ============================================================
# MODÈLE COMPLET : MCTNet+
# ============================================================

class MCTNetPlus(nn.Module):
    """
    MCTNet+ : MCTNet étendu avec Cross-Modal Attention Fusion.
    
    Architecture :
      1. Multi-stage CTFusion (identique à MCTNet baseline)
      2. Encodeurs séparés pour chaque groupe de covariables
      3. CMAF : attention croisée S2 ↔ covariables
      4. MLP classifier sur features fusionnées
    
    Avantage vs concaténation naive :
      - Chaque groupe est encodé dans le même espace latent
      - L'attention apprend à ignorer les groupes non pertinents
        (résout la dégradation en Arkansas avec "all")
    """
    def __init__(self,
                 n_classes,
                 n_bands=N_BANDS,
                 n_timesteps=N_TIMESTEPS,
                 n_stages=N_STAGES,
                 n_head=N_HEAD,
                 kernel_size=KERNEL_SIZE,
                 d_model=64,
                 d_cov=32,
                 cov_dims=None,
                 use_covariates=True):
        super().__init__()
        self.use_covariates = use_covariates

        # --- Input projection ---
        self.input_proj = nn.Linear(n_bands, d_model)

        # --- Stages CTFusion ---
        self.stages = nn.ModuleList()
        for i in range(n_stages):
            self.stages.append(CTFusionModule(
                in_channels=d_model,
                out_channels=d_model,
                n_head=n_head,
                kernel_size=kernel_size,
                use_alpe=(i == 0),
                max_len=n_timesteps
            ))
        self.poolings = nn.ModuleList(
            [nn.MaxPool1d(kernel_size=2, stride=2) for _ in range(n_stages - 1)]
        )

        # --- Encodeurs covariables ---
        if use_covariates and cov_dims is not None:
            self.cov_encoders = nn.ModuleDict({
                name: CovariateEncoder(dim, d_cov)
                for name, dim in cov_dims.items()
            })
            n_groups = len(cov_dims)
            self.cmaf = CrossModalAttentionFusion(d_model, d_cov, n_groups)
        else:
            self.cov_encoders = None
            self.cmaf = None

        # --- MLP Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, x, mask=None, covariates=None):
        """
        x          : (B, T, n_bands)
        mask       : (B, T)
        covariates : dict {"climate": (B,D1), "soil": (B,D2), "topo": (B,D3)}
        """
        # Input projection
        out = self.input_proj(x)   # (B, T, d_model)

        # Multi-stage CTFusion
        for i, stage in enumerate(self.stages):
            out = stage(out, mask)
            if i < len(self.poolings):
                out = out.transpose(1, 2)           # (B, d_model, T)
                out = self.poolings[i](out)         # (B, d_model, T//2)
                out = out.transpose(1, 2)           # (B, T//2, d_model)
                if mask is not None:
                    mask = mask[:, ::2]

        # Global max pooling → vecteur S2
        s2_feat = out.max(dim=1).values            # (B, d_model)

        # CMAF avec covariables
        attn_weights = None
        if self.use_covariates and self.cov_encoders is not None and covariates is not None:
            cov_feats = [
                self.cov_encoders[name](covariates[name])
                for name in self.cov_encoders.keys()
            ]
            s2_feat, attn_weights = self.cmaf(s2_feat, cov_feats)

        logits = self.classifier(s2_feat)
        return logits, attn_weights


# ============================================================
# ENTRAÎNEMENT
# ============================================================

def get_num_classes(state):
    """Arkansas: 5 classes, California: 6 classes."""
    return 5 if state == "Arkansas" else 6

def get_class_names(state):
    if state == "Arkansas":
        return ["Soybeans", "Rice", "Corn", "Cotton", "Others"]
    else:
        return ["Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios", "Others"]


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for batch in loader:
        s2   = batch["s2"].to(DEVICE)       # (B, T, bands)
        mask = batch["mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        covariates = {
            "climate": batch["climate"].to(DEVICE),
            "soil":    batch["soil"].to(DEVICE),
            "topo":    batch["topo"].to(DEVICE),
        } if model.use_covariates else None

        optimizer.zero_grad()
        logits, _ = model(s2, mask, covariates)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss    += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += len(labels)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    all_attn = []
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        s2   = batch["s2"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        labels = batch["label"]
        covariates = {
            "climate": batch["climate"].to(DEVICE),
            "soil":    batch["soil"].to(DEVICE),
            "topo":    batch["topo"].to(DEVICE),
        } if model.use_covariates else None

        logits, attn = model(s2, mask, covariates)
        
        if criterion is not None:
            loss = criterion(logits, labels.to(DEVICE))
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
        if attn is not None:
            all_attn.append(attn.cpu().numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    attn   = np.concatenate(all_attn) if all_attn else None

    oa    = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(labels, preds)
    cm    = confusion_matrix(labels, preds)
    
    if criterion is not None:
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return oa, f1, kappa, cm, preds, labels, attn, avg_loss
    return oa, f1, kappa, cm, preds, labels, attn


def train_model(state, use_covariates=True, d_model=64, d_cov=32):
    print(f"\n{'='*60}")
    print(f"  Entraînement MCTNet+ — {state} "
          f"({'avec' if use_covariates else 'sans'} covariables)")
    print(f"{'='*60}")

    n_classes = get_num_classes(state)

    # Datasets
    train_ds = CropDatasetPart3(state, "train")
    val_ds   = CropDatasetPart3(state, "val")
    test_ds  = CropDatasetPart3(state, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Dimensions covariables depuis la constante définie plus haut
    cov_dims = COVARIATE_DIMS if use_covariates else None

    # Modèle
    model = MCTNetPlus(
        n_classes=n_classes,
        n_bands=N_BANDS,
        n_timesteps=N_TIMESTEPS,
        n_stages=N_STAGES,
        n_head=N_HEAD,
        kernel_size=KERNEL_SIZE,
        d_model=d_model,
        d_cov=d_cov,
        cov_dims=cov_dims,
        use_covariates=use_covariates
    ).to(DEVICE)

    # Compter les paramètres
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_kappa = -1
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_oa": [], "val_f1": [], "val_kappa": []}
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_oa, val_f1, val_kappa, _, _, _, _, val_loss = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_oa"].append(val_oa)
        history["val_f1"].append(val_f1)
        history["val_kappa"].append(val_kappa)

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss={train_loss:.4f} "
                  f"| Train Acc={train_acc:.4f} | Val OA={val_oa:.4f} "
                  f"| Val Kappa={val_kappa:.4f}")

    # Évaluation finale sur test
    model.load_state_dict(best_state)
    test_oa, test_f1, test_kappa, cm, preds, labels, attn = evaluate(model, test_loader)
    print(f"\n  TEST — OA={test_oa:.4f} | F1={test_f1:.4f} | Kappa={test_kappa:.4f}")

    # Sauvegarde
    tag  = f"{state}_{'with_cov' if use_covariates else 'no_cov'}"
    save_dir = os.path.join(RESULTS_DIR, tag)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state, os.path.join(save_dir, "best_model.pth"))
    np.save(os.path.join(save_dir, "confusion_matrix.npy"), cm)
    np.save(os.path.join(save_dir, "preds.npy"), preds)
    np.save(os.path.join(save_dir, "labels.npy"), labels)
    if attn is not None:
        np.save(os.path.join(save_dir, "attn_weights.npy"), attn)

    metrics = {"test_oa": test_oa, "test_f1": test_f1, "test_kappa": test_kappa,
               "n_params": n_params}
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "state": state,
        "model": model,
        "history": history,
        "metrics": metrics,
        "cm": cm,
        "preds": preds,
        "labels": labels,
        "attn": attn,
        "save_dir": save_dir,
        "tag": tag,
    }


# ============================================================
# VISUALISATIONS
# ============================================================

def plot_training_curves(results_list):
    """Courbes d'entraînement pour tous les modèles."""
    n = len(results_list)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, res in enumerate(results_list):
        hist = res["history"]
        tag  = res["tag"]
        epochs_range = range(1, len(hist["train_loss"]) + 1)

        axes[i, 0].plot(epochs_range, hist["train_loss"], color="#378ADD", lw=1.5, label="Train Loss")
        if "val_loss" in hist and len(hist["val_loss"]) > 0:
            axes[i, 0].plot(epochs_range, hist["val_loss"], color="#D85A30", lw=1.5, label="Val Loss", linestyle="--")
        axes[i, 0].set_title(f"{tag}\nLoss", fontsize=11)
        axes[i, 0].set_xlabel("Epoch")
        axes[i, 0].set_ylabel("Loss")
        axes[i, 0].legend(fontsize=9)
        axes[i, 0].grid(alpha=0.3)

        axes[i, 1].plot(epochs_range, hist["train_acc"], color="#1D9E75",
                        lw=1.5, label="Train Acc")
        axes[i, 1].plot(epochs_range, hist["val_oa"],    color="#D85A30",
                        lw=1.5, label="Val OA",   linestyle="--")
        axes[i, 1].set_title(f"{tag}\nAccuracy", fontsize=11)
        axes[i, 1].set_xlabel("Epoch")
        axes[i, 1].legend(fontsize=9)
        axes[i, 1].grid(alpha=0.3)

        axes[i, 2].plot(epochs_range, hist["val_kappa"], color="#7F77DD", lw=1.5)
        axes[i, 2].set_title(f"{tag}\nVal Kappa", fontsize=11)
        axes[i, 2].set_xlabel("Epoch")
        axes[i, 2].set_ylabel("Kappa")
        axes[i, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
    plt.show()
    print("Courbes d'entraînement sauvegardées.")


def plot_confusion_matrices(results_list):
    """Matrices de confusion normalisées."""
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        state     = res["state"]
        cm        = res["cm"]
        class_names = get_class_names(state)
        cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, vmin=0, vmax=1,
                    annot_kws={"size": 9})
        ax.set_title(f"{res['tag']}\nOA={res['metrics']['test_oa']:.4f} | "
                     f"Kappa={res['metrics']['test_kappa']:.4f}", fontsize=11)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Vrai")
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150, bbox_inches='tight')
    plt.show()
    print("Matrices de confusion sauvegardées.")


def plot_attention_weights(results_list):
    """
    Visualise les poids d'attention CMAF moyens par groupe de covariables.
    Montre quels groupes sont les plus utilisés selon l'état.
    """
    cov_names = ["Climate", "Soil", "Topo"]
    results_with_attn = [r for r in results_list if r.get("attn") is not None]
    if not results_with_attn:
        print("Aucun poids d'attention disponible.")
        return

    fig, axes = plt.subplots(1, len(results_with_attn),
                             figsize=(5 * len(results_with_attn), 4))
    if len(results_with_attn) == 1:
        axes = [axes]

    for ax, res in zip(axes, results_with_attn):
        attn = res["attn"]              # (N_test, n_groups)
        mean_attn = attn.mean(axis=0)   # (n_groups,)
        std_attn  = attn.std(axis=0)

        bars = ax.bar(cov_names, mean_attn, yerr=std_attn,
                      color=["#378ADD", "#1D9E75", "#D85A30"],
                      capsize=5, edgecolor='white', linewidth=0.5)
        ax.set_title(f"{res['tag']}\nPoids d'attention CMAF (moy ± std)", fontsize=11)
        ax.set_ylabel("Attention weight")
        ax.set_ylim(0, 1)
        ax.axhline(1/3, color='gray', linestyle='--', alpha=0.5, label="Uniforme (1/3)")
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, mean_attn):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cmaf_attention_weights.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Poids d'attention CMAF sauvegardés.")


def plot_comparison_table(results_list, part2_results=None):
    """
    Tableau comparatif : MCTNet baseline vs Part2 best vs MCTNet+.
    
    part2_results : dict optionnel avec les meilleurs résultats de la partie 2
    Format: {"Arkansas": {"oa": ..., "f1": ..., "kappa": ...}, "California": {...}}
    """
    # Résultats Part 2 (vos données réelles)
    if part2_results is None:
        part2_results = {
            "Arkansas": {
                "baseline":  {"oa": 0.8644, "f1": 0.8148, "kappa": 0.8025},
                "best_part2": {"oa": 0.9093, "f1": 0.8683, "kappa": 0.8642,
                               "config": "S2+Soil"},
            },
            "California": {
                "baseline":  {"oa": 0.8466, "f1": 0.7861, "kappa": 0.7995},
                "best_part2": {"oa": 0.8838, "f1": 0.8282, "kappa": 0.8469,
                               "config": "All cov."},
            }
        }

    rows = []
    for res in results_list:
        state = res["state"]
        rows.append({
            "State":  state,
            "Model":  "MCTNet (baseline)",
            "OA":     part2_results[state]["baseline"]["oa"],
            "F1":     part2_results[state]["baseline"]["f1"],
            "Kappa":  part2_results[state]["baseline"]["kappa"],
        })
        rows.append({
            "State": state,
            "Model": f"Best Part2 ({part2_results[state]['best_part2']['config']})",
            "OA":    part2_results[state]["best_part2"]["oa"],
            "F1":    part2_results[state]["best_part2"]["f1"],
            "Kappa": part2_results[state]["best_part2"]["kappa"],
        })
        rows.append({
            "State": state,
            "Model": "MCTNet+ (CMAF) [Ours]",
            "OA":    res["metrics"]["test_oa"],
            "F1":    res["metrics"]["test_f1"],
            "Kappa": res["metrics"]["test_kappa"],
        })

    df = pd.DataFrame(rows)

    # Plot
    states_list = df["State"].unique()
    fig, axes = plt.subplots(1, len(states_list), figsize=(7 * len(states_list), 5))
    if len(states_list) == 1:
        axes = [axes]

    metrics_to_plot = ["OA", "F1", "Kappa"]
    colors = ["#B4B2A9", "#378ADD", "#D85A30"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for ax, state in zip(axes, states_list):
        sub = df[df["State"] == state].reset_index(drop=True)
        for j, (_, row) in enumerate(sub.iterrows()):
            vals = [row["OA"], row["F1"], row["Kappa"]]
            bars = ax.bar(x + j * width, vals, width, label=row["Model"],
                          color=colors[j], edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha='center', va='bottom', fontsize=8)

        ax.set_title(f"{state}", fontsize=13, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_to_plot, fontsize=11)
        ax.set_ylim(0.7, 1.0)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Comparaison : Baseline vs Part2 Best vs MCTNet+ (CMAF)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Tableau de comparaison sauvegardé.")
    print("\n" + df.to_string(index=False))
    return df


def plot_per_class_f1(results_list):
    """F1-score par classe pour MCTNet+ vs baseline."""
    fig, axes = plt.subplots(1, len(results_list),
                             figsize=(6 * len(results_list), 4))
    if len(results_list) == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        state       = res["state"]
        class_names = get_class_names(state)
        report      = classification_report(res["labels"], res["preds"],
                                            target_names=class_names,
                                            output_dict=True, zero_division=0)
        f1_scores   = [report[c]["f1-score"] for c in class_names]
        colors = plt.cm.RdYlGn(np.array(f1_scores))
        bars = ax.barh(class_names, f1_scores, color=colors, edgecolor='white')
        ax.set_xlim(0, 1)
        ax.axvline(0.8, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, f1_scores):
            ax.text(min(val + 0.02, 0.97), bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va='center', fontsize=10)
        ax.set_title(f"{res['tag']}\nF1 par classe", fontsize=11)
        ax.set_xlabel("F1-score")
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_class_f1.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("F1 par classe sauvegardé.")


def plot_all_results(results_list):
    """Lance toutes les visualisations."""
    print("\n" + "="*60)
    print("  GÉNÉRATION DES GRAPHIQUES")
    print("="*60)

    plot_training_curves(results_list)
    plot_confusion_matrices(results_list)
    plot_attention_weights(results_list)
    plot_per_class_f1(results_list)
    df_comparison = plot_comparison_table(results_list)

    # Résumé final
    print("\n" + "="*60)
    print("  RÉSUMÉ DES RÉSULTATS MCTNet+")
    print("="*60)
    for res in results_list:
        m = res["metrics"]
        print(f"\n  {res['tag']}")
        print(f"    OA    = {m['test_oa']:.4f}")
        print(f"    F1    = {m['test_f1']:.4f}")
        print(f"    Kappa = {m['test_kappa']:.4f}")
        print(f"    Params = {m['n_params']:,}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    all_results = []

    for state in STATES:
        # MCTNet+ avec covariables (modèle proposé)
        res = train_model(state, use_covariates=True)
        all_results.append(res)

    # Génération de tous les graphiques
    plot_all_results(all_results)

    print(f"\nTous les résultats sauvegardés dans : {RESULTS_DIR}/")


# ============================================================
# ABLATION INTERNE PARTIE 3 : MCTNet+ sans covariables
# (pour comparer avec/sans CMAF)
# ============================================================

def ablation_cmaf(state):
    """
    Compare MCTNet+ avec et sans CMAF pour un état donné.
    Permet de vérifier l'apport isolé du mécanisme CMAF.
    """
    print(f"\n{'='*60}")
    print(f"  Ablation CMAF — {state}")
    print(f"{'='*60}")

    results = {}
    for use_cov in [False, True]:
        label = "avec_CMAF" if use_cov else "sans_CMAF"
        res   = train_model(state, use_covariates=use_cov)
        results[label] = res["metrics"]
        print(f"\n  {label} : OA={res['metrics']['test_oa']:.4f} "
              f"| Kappa={res['metrics']['test_kappa']:.4f}")

    print(f"\n  Gain CMAF : "
          f"ΔOA = {results['avec_CMAF']['test_oa'] - results['sans_CMAF']['test_oa']:+.4f} "
          f"| ΔKappa = {results['avec_CMAF']['test_kappa'] - results['sans_CMAF']['test_kappa']:+.4f}")
    return results


if __name__ == "__main__":
    main()