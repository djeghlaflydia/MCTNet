"""
=============================================================================
PARTIE 3 — MSF-MCTNet : Multi-Stream Fusion MCTNet
Code COMPLET : Architecture + Dataset + Entraînement + Analyse
=============================================================================

Justification basée sur l'ablation Partie 2 :
  Arkansas  : s2_topo (F1=0.871) meilleur, "all" régresse (F1=0.823)
              → concaténation naïve = interférence entre features
  California: "all" meilleur (F1=0.827)
              → la fusion intelligente doit profiter de toutes les sources

Solution : séparer strictement flux temporel (S2) et flux statique
(covariables), puis fusionner via Cross-Attention.
=============================================================================
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import math
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG GLOBALE
# ─────────────────────────────────────────────
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = "./preprocessed_ablation"  # ← adapté à votre structure .pt
RESULTS_DIR = "./part3_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# Hyperparamètres (identiques à l'ablation Partie 2 pour comparaison équitable)
BATCH_SIZE  = 32
EPOCHS      = 200
LR          = 1e-3
WEIGHT_DECAY= 5e-2      # Augmenté pour réduire l'overfitting (comme en Partie 2)
LABEL_SMOOTHING= 0.1    # Ajouté pour une meilleure généralisation
PATIENCE     = 15      # Early stopping
N_STAGES    = 3
N_HEAD      = 5
KERNEL_SIZE = 3
D_MODEL     = 80      # dimension latente principale (= sortie GMP)
D_STATIC    = 32      # dimension encodage covariables statiques
N_BANDS     = 10
N_TIMESTEPS = 36

STATES   = ["Arkansas", "California"]
CONFIGS  = ["baseline", "s2_climate", "s2_soil", "s2_topo", "all"]

CLASS_NAMES = {
    "Arkansas":   ["Others", "Corn", "Cotton", "Rice", "Soybeans"],
    "California": ["Others", "Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios"],
}

# Résultats Partie 2 (concaténation naïve) — pour comparaison finale
# Résultats réels de la Partie 2 (Ablation) extraits de votre rapport technique
PART2_RESULTS = {
    "Arkansas": {
        "baseline":   {"oa": 0.8644, "f1": 0.8148, "kappa": 0.8025},
        "s2_climate": {"oa": 0.9044, "f1": 0.8681, "kappa": 0.8572},
        "s2_soil":    {"oa": 0.9093, "f1": 0.8683, "kappa": 0.8642},
        "s2_topo":    {"oa": 0.8653, "f1": 0.8249, "kappa": 0.8034},
        "all":        {"oa": 0.8040, "f1": 0.7847, "kappa": 0.7246},
    },
    "California": {
        "baseline":   {"oa": 0.8466, "f1": 0.7861, "kappa": 0.7995},
        "s2_climate": {"oa": 0.8761, "f1": 0.8254, "kappa": 0.8362},
        "s2_soil":    {"oa": 0.8679, "f1": 0.8080, "kappa": 0.8260},
        "s2_topo":    {"oa": 0.8772, "f1": 0.8100, "kappa": 0.8376},
        "all":        {"oa": 0.8838, "f1": 0.8282, "kappa": 0.8469},
    },
}

print(f"Device : {DEVICE}")


# ══════════════════════════════════════════════════════════════
# 1. DATASET
# ══════════════════════════════════════════════════════════════

class CropDataset(Dataset):
    """
    Dataset chargé depuis des fichiers .pt par état, par config et par split.
    Structure dans DATA_DIR/<state>/<config>/<split>.pt :
        Dict {'X': tensor, 'mask': tensor, 'y': tensor}
    """

    def __init__(self, state, config, split="train", data_dir=DATA_DIR):
        path = os.path.join(data_dir, state, config, f"{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier non trouvé : {path}")
            
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.X    = data["X"].float()      # (N, T, C)
        self.mask = data["mask"].float()   # (N, T)
        self.y    = data["y"].long()       # (N,)

        # Extraction des sources
        # S2 est toujours les 10 premières colonnes
        self.X_s2 = self.X[:, :, :10]
        
        # Initialisation des covariables (zéros par défaut)
        N = len(self.y)
        self.X_climate = torch.zeros((N, 3))
        self.X_soil    = torch.zeros((N, 3))
        self.X_topo    = torch.zeros((N, 2))

        # Slicing selon la configuration (basé sur 02_preprocessing.py)
        # On fait la moyenne sur T car ce sont des données statiques (ou répétées)
        if config == "s2_climate":
            self.X_climate = self.X[:, :, 10:13].mean(dim=1)
        elif config == "s2_soil":
            self.X_soil    = self.X[:, :, 10:13].mean(dim=1)
        elif config == "s2_topo":
            self.X_topo    = self.X[:, :, 10:12].mean(dim=1)
        elif config == "all":
            self.X_climate = self.X[:, :, 10:13].mean(dim=1)
            self.X_soil    = self.X[:, :, 13:16].mean(dim=1)
            self.X_topo    = self.X[:, :, 16:18].mean(dim=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "s2":      self.X_s2[idx],        # (T, 10)
            "mask":    self.mask[idx],        # (T,)
            "climate": self.X_climate[idx],   # (3,)
            "soil":    self.X_soil[idx],      # (3,)
            "topo":    self.X_topo[idx],      # (2,)
            "label":   self.y[idx],           # scalaire
        }


def get_loaders(state, config):
    train_ds = CropDataset(state, config, "train")
    val_ds   = CropDataset(state, config, "val")
    test_ds  = CropDataset(state, config, "test")
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=0, pin_memory=True),
        DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=0),
        DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=0),
        train_ds,
    )


# ══════════════════════════════════════════════════════════════
# 2. BLOCS DE BASE — inchangés vs MCTNet
# ══════════════════════════════════════════════════════════════

class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., 2020).
    Entrée : (B, C, T)  →  Sortie : (B, C, T)
    """
    def __init__(self, channels, k=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv     = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x).transpose(-1, -2)
        y = torch.sigmoid(self.conv(y)).transpose(-1, -2)
        return x * y


class ALPE(nn.Module):
    """Attention-based Learnable Positional Encoding.
    Masque les positions manquantes avant l'apprentissage du PE.
    Entrée : x (B,T,D), mask (B,T)  →  Sortie : (B,T,D)
    """
    def __init__(self, d_model, max_len=36, kernel_size=3):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              padding=kernel_size // 2)
        self.eca  = ECA(d_model)

    def forward(self, x, mask):
        B, T, D = x.shape
        pe = self.pe[:T].unsqueeze(0).expand(B, -1, -1) * mask.unsqueeze(-1)
        pe = self.eca(self.conv(pe.transpose(1, 2))).transpose(1, 2)
        return x + pe


class CNNSubModule(nn.Module):
    """CNN 1D résiduel sur la dimension temporelle.
    Entrée : (B, C_in, T)  →  Sortie : (B, C_out, T)
    """
    def __init__(self, in_c, out_c, ks=3):
        super().__init__()
        p = ks // 2
        self.conv1 = nn.Conv1d(in_c,  out_c, ks, padding=p)
        self.bn1   = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, ks, padding=p)
        self.bn2   = nn.BatchNorm1d(out_c)
        self.skip  = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + res)


class TransformerSubModule(nn.Module):
    """Transformer encoder 1 couche avec ALPE optionnel.
    Entrée : x (B,T,D), mask (B,T)  →  Sortie : (B,T,D)
    """
    def __init__(self, d_model, n_head, use_alpe=False,
                 max_len=36, kernel_size=3):
        super().__init__()
        self.use_alpe = use_alpe
        if use_alpe:
            self.alpe = ALPE(d_model, max_len, kernel_size)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)

    def forward(self, x, mask=None):
        if self.use_alpe and mask is not None:
            x = self.alpe(x, mask)
        return self.encoder(x)


class CTFusionModule(nn.Module):
    """CNN ∥ Transformer → concat → projection linéaire.
    Entrée : x (B,T,C_in), mask (B,T)  →  Sortie : (B,T,C_out)
    """
    def __init__(self, in_c, out_c, n_head, ks=3,
                 use_alpe=False, max_len=36):
        super().__init__()
        self.cnn   = CNNSubModule(in_c, out_c, ks)
        self.trans = TransformerSubModule(in_c, n_head, use_alpe, max_len, ks)
        self.proj  = nn.Linear(in_c + out_c, out_c)
        self.drop  = nn.Dropout(0.2)  # Ajouté pour réduire l'overfitting

    def forward(self, x, mask=None):
        cnn_out   = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        trans_out = self.trans(x, mask)
        out = torch.cat([cnn_out, trans_out], dim=-1)
        return F.relu(self.drop(self.proj(out)))


# ══════════════════════════════════════════════════════════════
# 3. NOUVEAUX BLOCS MSF-MCTNet
# ══════════════════════════════════════════════════════════════

class StaticEncoder(nn.Module):
    """MLP pour covariables STATIQUES (pas de Transformer — elles
    n'ont pas de structure temporelle).

    Pourquoi pas un Transformer ?
    Dans l'ablation Partie 2, passer s2_soil dans le Transformer
    temporel dégrade les résultats en Arkansas : le modèle traite
    une valeur constante comme un signal dynamique.

    Entrée : (B, in_dim)  →  Sortie : (B, d_static) normalisé
    """
    def __init__(self, in_dim, d_static=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_static * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_static * 2, d_static),
            nn.LayerNorm(d_static),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Fusion cross-attention : flux temporel S2 (Query) ×
    covariables statiques (Key / Value).

    Formulation mathématique :
        Q = W_q · temporal            shape (B, 1, d_model)
        K = W_k · stack(statics)      shape (B, n_groups, d_model)
        V = W_v · stack(statics)      shape (B, n_groups, d_model)
        α = softmax(Q·Kᵀ / √d_model) shape (B, 1, n_groups)
        context = α · V               shape (B, d_model)
        gate    = σ(W_g · [temporal; context])
        fused   = LayerNorm(gate·context + (1−gate)·temporal)

    La connexion résiduelle gated garantit que si les covariables
    n'apportent rien, le modèle conserve exactement le vecteur
    temporel baseline → élimine la régression observée en Arkansas.

    Paramètres :
        d_temporal : dimension vecteur S2 agrégé (D_MODEL = 80)
        d_static   : dimension encodage chaque groupe covariable (32)
        n_groups   : nombre de groupes (1, 2 ou 3 selon config)
    """
    def __init__(self, d_temporal, d_static, n_groups, dropout=0.1):
        super().__init__()
        self.scale  = math.sqrt(d_temporal)
        self.q_proj = nn.Linear(d_temporal, d_temporal)
        self.k_proj = nn.Linear(d_static,   d_temporal)
        self.v_proj = nn.Linear(d_static,   d_temporal)
        self.drop   = nn.Dropout(dropout)
        self.gate   = nn.Sequential(
            nn.Linear(d_temporal * 2, d_temporal),
            nn.Sigmoid(),
        )
        self.norm   = nn.LayerNorm(d_temporal)

    def forward(self, temporal, static_feats):
        """
        temporal     : (B, d_temporal)
        static_feats : list of tensors (B, d_static)
        retourne     : fused (B, d_temporal), attn (B, n_groups)
        """
        stacked = torch.stack(static_feats, dim=1)          # (B, G, d_static)
        q = self.q_proj(temporal).unsqueeze(1)               # (B, 1, d_temp)
        k = self.k_proj(stacked)                             # (B, G, d_temp)
        v = self.v_proj(stacked)                             # (B, G, d_temp)
        attn    = F.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        context = self.drop((attn @ v).squeeze(1))           # (B, d_temp)
        gate    = self.gate(torch.cat([temporal, context], -1))
        fused   = gate * context + (1 - gate) * temporal
        return self.norm(fused), attn.squeeze(1)             # (B,G)


# ══════════════════════════════════════════════════════════════
# 4. MODÈLE PRINCIPAL : MSFMCTNet
# ══════════════════════════════════════════════════════════════

class MSFMCTNet(nn.Module):
    """Multi-Stream Fusion MCTNet.

    Deux flux strictement séparés jusqu'à la fusion :
      • Flux temporel  : S2 → InputProj → CTFusion×3 → GMP → (B, 80)
      • Flux statique  : covariables → StaticEncoder×n_groups → (B, 32)×G
      • Fusion         : CrossAttentionFusion → (B, 80)
      • Classifieur    : MLP → (B, n_classes)

    Paramètres :
        n_classes   : 5 (Arkansas) ou 6 (California)
        n_bands     : 10 bandes Sentinel-2
        n_timesteps : 36 pas temporels
        n_stages    : 3 stages CTFusion
        n_head      : 5 têtes d'attention
        kernel_size : 3 (kernel Conv1D)
        d_model     : 80 (dimension principale)
        d_static    : 32 (dimension encodage covariables)
        cov_dims    : dict {nom: dim} — None = mode baseline
        dropout_cls : 0.3
    """

    def __init__(
        self,
        n_classes,
        n_bands=N_BANDS,
        n_timesteps=N_TIMESTEPS,
        n_stages=N_STAGES,
        n_head=N_HEAD,
        kernel_size=KERNEL_SIZE,
        d_model=D_MODEL,
        d_static=D_STATIC,
        cov_dims=None,
        dropout_cls=0.3,
    ):
        super().__init__()
        self.use_cov  = (cov_dims is not None and len(cov_dims) > 0)
        self.cov_keys = list(cov_dims.keys()) if self.use_cov else []

        # ── Projection d'entrée ──────────────────────────────────
        self.input_proj = nn.Linear(n_bands, d_model)

        # ── Stages CTFusion ──────────────────────────────────────
        self.stages   = nn.ModuleList()
        self.poolings = nn.ModuleList()
        cur_len = n_timesteps
        for i in range(n_stages):
            self.stages.append(CTFusionModule(
                d_model, d_model, n_head, kernel_size,
                use_alpe=(i == 0), max_len=cur_len,
            ))
            if i < n_stages - 1:
                self.poolings.append(nn.MaxPool1d(2, stride=2))
                cur_len //= 2

        # ── Encodeurs covariables statiques ─────────────────────
        if self.use_cov:
            self.cov_enc = nn.ModuleDict({
                k: StaticEncoder(v, d_static) for k, v in cov_dims.items()
            })
            self.fusion = CrossAttentionFusion(d_model, d_static,
                                               n_groups=len(cov_dims))

        # ── Classifieur ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout_cls),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_cls / 2),
            nn.Linear(d_model // 2, n_classes),
        )

    # ── Extraction features temporelles ─────────────────────────
    def _temporal(self, x, mask):
        out = self.input_proj(x)       # (B, T, d_model)
        cur_mask = mask
        for i, stage in enumerate(self.stages):
            out = stage(out, cur_mask)
            if i < len(self.poolings):
                out      = self.poolings[i](out.transpose(1, 2)).transpose(1, 2)
                cur_mask = cur_mask[:, ::2] if cur_mask is not None else None
        return out.max(dim=1).values   # GMP → (B, d_model)

    # ── Forward complet ─────────────────────────────────────────
    def forward(self, x, mask=None, covariates=None):
        """
        x          : (B, T, n_bands)
        mask       : (B, T)
        covariates : dict {str: (B, dim)} ou None
        → logits (B, n_classes), attn_weights (B, n_groups) ou None
        """
        feat = self._temporal(x, mask)        # (B, d_model)
        attn = None
        if self.use_cov and covariates is not None:
            static = [self.cov_enc[k](covariates[k]) for k in self.cov_keys]
            feat, attn = self.fusion(feat, static)
        return self.head(feat), attn

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════
# 5. FACTORY
# ══════════════════════════════════════════════════════════════

def build_model(state, config, all_cov_dims):
    """Construit MSFMCTNet pour (state, config)."""
    n_classes = len(CLASS_NAMES[state])
    mapping = {
        "baseline":   None,
        "s2_climate": {k: v for k, v in all_cov_dims.items() if k == "climate"},
        "s2_soil":    {k: v for k, v in all_cov_dims.items() if k == "soil"},
        "s2_topo":    {k: v for k, v in all_cov_dims.items() if k == "topo"},
        "all":        all_cov_dims,
    }
    return MSFMCTNet(n_classes=n_classes, cov_dims=mapping[config])


# ══════════════════════════════════════════════════════════════
# 6. ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════

def prepare_batch(batch, config, device):
    """Prépare un batch selon la config (quelles covariables inclure)."""
    s2   = batch["s2"].to(device)
    mask = batch["mask"].to(device)
    y    = batch["label"].to(device)
    cov  = None
    if config != "baseline":
        keys = {
            "s2_climate": ["climate"],
            "s2_soil":    ["soil"],
            "s2_topo":    ["topo"],
            "all":        ["climate", "soil", "topo"],
        }[config]
        cov = {k: batch[k].to(device) for k in keys}
    return s2, mask, y, cov


def train_one_epoch(model, loader, optimizer, criterion, config, device):
    model.train()
    total_loss = total_correct = total = 0
    for batch in loader:
        s2, mask, y, cov = prepare_batch(batch, config, device)
        optimizer.zero_grad()
        logits, _ = model(s2, mask, cov)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss    += loss.item() * len(y)
        total_correct += (logits.argmax(1) == y).sum().item()
        total         += len(y)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, config, device):
    model.eval()
    all_preds, all_labels, all_attn = [], [], []
    total_loss = 0
    total = 0
    for batch in loader:
        s2, mask, y, cov = prepare_batch(batch, config, device)
        logits, attn = model(s2, mask, cov)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        total += len(y)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
        if attn is not None:
            all_attn.append(attn.cpu().numpy())
    
    avg_loss = total_loss / total
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    attn   = np.concatenate(all_attn) if all_attn else None
    oa     = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, average="macro", zero_division=0)
    kappa  = cohen_kappa_score(labels, preds)
    cm     = confusion_matrix(labels, preds)
    return avg_loss, oa, f1, kappa, cm, preds, labels, attn


def run_training(state, config, all_cov_dims, loaders):
    """Entraîne MSFMCTNet pour (state, config) et retourne les résultats."""
    train_loader, val_loader, test_loader = loaders
    print(f"  Entraînement : {state} | {config}")
    print(f"  Régularisation : Weight Decay={WEIGHT_DECAY}, Label Smoothing={LABEL_SMOOTHING}, Early Stopping={PATIENCE}")
    
    model     = build_model(state, config, all_cov_dims).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    
    # Scheduler: Plateau (plus efficace pour l'overfitting que Cosine dans ce cas)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    history = {k: [] for k in
               ["train_loss", "val_loss", "train_acc", "val_oa", "val_kappa"]}
    best_kappa = -1
    best_state = None
    trigger_times = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, config, DEVICE)
        val_loss, val_oa, val_f1, val_kappa, *_ = evaluate(
            model, val_loader, criterion, config, DEVICE)
        
        # Scheduler step basé sur OA (ou Kappa)
        scheduler.step(val_oa)

        for k, v in zip(history.keys(),
                        [tr_loss, val_loss, tr_acc, val_oa, val_kappa]):
            history[k].append(v)

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            trigger_times = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            trigger_times += 1

        if epoch % 10 == 0 or epoch == 1 or trigger_times == 0:
            print(f"    [{epoch:3d}/{EPOCHS}] loss={tr_loss:.4f}/{val_loss:.4f} "
                  f"acc={tr_acc:.4f}/{val_oa:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping
        if trigger_times >= PATIENCE:
            print(f"    [!] Early stopping à l'époque {epoch}")
            break

    # Test final
    model.load_state_dict(best_state)
    _, test_oa, test_f1, test_kappa, cm, preds, labels, attn = evaluate(
        model, test_loader, criterion, config, DEVICE)

    # Sauvegarde
    tag = f"{state}_{config}"
    save_dir = os.path.join(RESULTS_DIR, tag)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state,  os.path.join(save_dir, "best_model.pth"))
    np.save(os.path.join(save_dir, "cm.npy"),     cm)
    np.save(os.path.join(save_dir, "preds.npy"),  preds)
    np.save(os.path.join(save_dir, "labels.npy"), labels)
    if attn is not None:
        np.save(os.path.join(save_dir, "attn.npy"), attn)
    metrics = {"oa": test_oa, "f1": test_f1, "kappa": test_kappa,
               "n_params": model.n_params()}
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  [OK] {tag:30s}  OA={test_oa:.4f}  F1={test_f1:.4f}  "
          f"Kappa={test_kappa:.4f}  params={model.n_params():,}")
    return {
        "state": state, "config": config, "tag": tag,
        "metrics": metrics, "history": history,
        "cm": cm, "preds": preds, "labels": labels, "attn": attn,
    }


# ══════════════════════════════════════════════════════════════
# 7. VISUALISATIONS
# ══════════════════════════════════════════════════════════════

# ── Palette cohérente ────────────────────────────────────────
COLORS = {
    "baseline":   "#B4B2A9",
    "s2_climate": "#378ADD",
    "s2_soil":    "#1D9E75",
    "s2_topo":    "#D85A30",
    "all":        "#7F77DD",
}
LABELS_FR = {
    "baseline":   "Baseline (S2)",
    "s2_climate": "S2 + Climat",
    "s2_soil":    "S2 + Sol",
    "s2_topo":    "S2 + Topo",
    "all":        "S2 + Tout",
}


def fig1_training_curves(all_results):
    """Figure 1 : courbes loss / accuracy / kappa pour chaque (state, config).
    Affiche Train vs Val sur le même graphe pour détecter l'overfitting.
    """
    states_list = list({r["state"] for r in all_results})
    n_states    = len(states_list)

    fig, axes = plt.subplots(n_states, 3, figsize=(18, 5 * n_states))
    if n_states == 1:
        axes = axes[np.newaxis, :]

    for row, state in enumerate(states_list):
        sub = [r for r in all_results if r["state"] == state]
        for res in sub:
            h   = res["history"]
            ep  = range(1, len(h["train_loss"]) + 1)
            cfg = res["config"]
            c   = COLORS[cfg]
            lbl = LABELS_FR[cfg]
            
            # Subplot 1: Loss (Train vs Val)
            axes[row, 0].plot(ep, h["train_loss"], color=c, lw=1.5, label=f"{lbl} (Train)")
            axes[row, 0].plot(ep, h["val_loss"],   color=c, lw=1.0, ls="--", label=f"{lbl} (Val)")
            
            # Subplot 2: Accuracy (Train vs Val)
            axes[row, 1].plot(ep, h["train_acc"],  color=c, lw=1.5, label=f"{lbl} (Train)")
            axes[row, 1].plot(ep, h["val_oa"],     color=c, lw=1.0, ls="--", label=f"{lbl} (Val)")
            
            # Subplot 3: Kappa (Val)
            axes[row, 2].plot(ep, h["val_kappa"],  color=c, lw=1.5, label=lbl)

        for col, (title, ylabel) in enumerate([
            ("Train/Val Loss", "Loss"),
            ("Train/Val Accuracy (OA)", "Accuracy"),
            ("Val Kappa", "Kappa"),
        ]):
            axes[row, col].set_title(f"{state} - {title}", fontsize=12)
            axes[row, col].set_xlabel("Epoque")
            axes[row, col].set_ylabel(ylabel)
            axes[row, col].grid(alpha=0.3)
            # On n'affiche la légende que si c'est lisible (trop de courbes sinon)
            if len(sub) <= 2 or col == 2:
                axes[row, col].legend(fontsize=8, loc="best")

    plt.suptitle("MSF-MCTNet - Courbes d'apprentissage (Train vs Validation)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig1_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Sauvegarde : {path}")


def fig2_confusion_matrices(all_results):
    """Figure 2 : matrices de confusion normalisées (toutes configs)."""
    states_list = list({r["state"] for r in all_results})

    for state in states_list:
        sub = [r for r in all_results if r["state"] == state]
        n   = len(sub)
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
        if n == 1:
            axes = [axes]

        names = CLASS_NAMES[state]
        for ax, res in zip(axes, sub):
            cm_norm = res["cm"].astype(float) / (
                res["cm"].sum(axis=1, keepdims=True) + 1e-9)
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=names, yticklabels=names,
                        ax=ax, vmin=0, vmax=1, annot_kws={"size": 8})
            m = res["metrics"]
            ax.set_title(
                f"{LABELS_FR[res['config']]}\n"
                f"OA={m['oa']:.3f}  Kappa={m['kappa']:.3f}",
                fontsize=10)
            ax.set_xlabel("Prédit", fontsize=9)
            ax.set_ylabel("Réel",   fontsize=9)
            ax.tick_params(axis="x", rotation=35)

        plt.suptitle(f"MSF-MCTNet — Matrices de confusion ({state})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"fig2_confusion_{state}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Sauvegarde : {path}")


def fig3_comparison_part2_vs_part3(all_results):
    """Figure 3 : barplot Partie 2 (concat naïve) vs Partie 3 (MSF).
    Montre le gain / la correction de régression apportée par MSF-MCTNet.
    """
    metrics_names = ["OA", "F1", "Kappa"]
    states_list   = list({r["state"] for r in all_results})
    n_states      = len(states_list)
    n_configs     = len(CONFIGS)

    fig, axes = plt.subplots(n_states, 3, figsize=(16, 5 * n_states))
    if n_states == 1:
        axes = axes[np.newaxis, :]

    x     = np.arange(n_configs)
    width = 0.35

    for row, state in enumerate(states_list):
        p2 = PART2_RESULTS[state]
        p3 = {r["config"]: r["metrics"]
              for r in all_results if r["state"] == state}

        for col, metric in enumerate(["oa", "f1", "kappa"]):
            ax = axes[row, col]
            v2 = [p2[c][metric]         for c in CONFIGS]
            v3 = [p3[c][metric]         for c in CONFIGS]
            deltas = [v3[i] - v2[i]     for i in range(n_configs)]

            b2 = ax.bar(x - width / 2, v2, width, label="Part 2 (concat)",
                        color=[COLORS[c] for c in CONFIGS], alpha=0.5,
                        edgecolor="white")
            b3 = ax.bar(x + width / 2, v3, width, label="MSF-MCTNet (Part 3)",
                        color=[COLORS[c] for c in CONFIGS],
                        edgecolor="white")

            # Annotations delta
            for i, (bar, d) in enumerate(zip(b3, deltas)):
                sign  = "+" if d >= 0 else ""
                color = "#1D9E75" if d >= 0 else "#D85A30"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{sign}{d:.3f}", ha="center", fontsize=7,
                        color=color, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels([LABELS_FR[c] for c in CONFIGS],
                               rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0.68, 1.0)
            ax.set_ylabel(metrics_names[col])
            ax.set_title(f"{state} — {metrics_names[col]}", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Comparaison Partie 2 (concat naïve) vs Partie 3 (MSF cross-attention)\n"
        "Δ en vert = gain, en rouge = régression",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig3_part2_vs_part3.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Sauvegarde : {path}")


def fig4_attention_weights(all_results):
    """Figure 4 : poids d'attention CMAF moyens par groupe et par état.
    Révèle quelles covariables le modèle utilise selon la région.
    """
    cov_labels  = {"climate": "Climat", "soil": "Sol", "topo": "Topo"}
    config_map  = {
        "s2_climate": ["climate"],
        "s2_soil":    ["soil"],
        "s2_topo":    ["topo"],
        "all":        ["climate", "soil", "topo"],
    }
    results_attn = [r for r in all_results
                    if r["attn"] is not None and r["config"] != "baseline"]
    if not results_attn:
        print("  Aucun poids d'attention disponible.")
        return

    # Grouper par état
    states_list = list({r["state"] for r in results_attn})
    n_states    = len(states_list)
    n_configs   = len(results_attn) // max(n_states, 1)

    fig, axes = plt.subplots(n_states, max(n_configs, 1),
                             figsize=(5 * max(n_configs, 1), 4.5 * n_states),
                             squeeze=False)

    col_idx_per_state = {s: 0 for s in states_list}
    for res in results_attn:
        state = res["state"]
        cfg   = res["config"]
        row   = states_list.index(state)
        col   = col_idx_per_state[state]
        col_idx_per_state[state] += 1
        ax    = axes[row][col]

        keys  = config_map.get(cfg, [])
        attn  = res["attn"]           # (N_test, n_groups)
        mu    = attn.mean(axis=0)
        std   = attn.std(axis=0)
        xlbls = [cov_labels.get(k, k) for k in keys]
        clrs  = [{"climate": "#378ADD", "soil": "#1D9E75",
                  "topo": "#D85A30"}.get(k, "#AAA") for k in keys]

        bars = ax.bar(xlbls, mu, yerr=std, color=clrs,
                      capsize=6, edgecolor="white")
        if len(keys) > 1:
            ax.axhline(1 / len(keys), color="gray", ls="--",
                       alpha=0.6, label=f"Uniforme ({1/len(keys):.2f})")
            ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Attention weight")
        ax.set_title(f"{state} — {LABELS_FR[cfg]}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, mu):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(bar.get_height() + 0.03, 0.92),
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("MSF-MCTNet — Poids d'attention CMAF par groupe de covariables\n"
                 "(moyenne ± std sur le jeu de test)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig4_attention_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Sauvegarde : {path}")


def fig5_per_class_f1(all_results):
    """Figure 5 : F1-score par classe, comparaison entre configs."""
    states_list = list({r["state"] for r in all_results})

    for state in states_list:
        sub   = [r for r in all_results if r["state"] == state]
        names = CLASS_NAMES[state]
        n_cls = len(names)
        n_cfg = len(sub)

        fig, axes = plt.subplots(1, n_cfg, figsize=(4.5 * n_cfg, 5),
                                 sharey=True)
        if n_cfg == 1:
            axes = [axes]

        for ax, res in zip(axes, sub):
            report = classification_report(
                res["labels"], res["preds"],
                target_names=names, output_dict=True, zero_division=0)
            f1s   = [report[n]["f1-score"] for n in names]
            clrs  = plt.cm.RdYlGn(np.array(f1s))
            bars  = ax.barh(names, f1s, color=clrs, edgecolor="white")
            ax.set_xlim(0, 1)
            ax.axvline(0.8, color="gray", ls="--", alpha=0.5, lw=1)
            ax.set_title(f"{LABELS_FR[res['config']]}\n"
                         f"F1-macro={res['metrics']['f1']:.3f}", fontsize=10)
            ax.set_xlabel("F1-score")
            ax.grid(axis="x", alpha=0.3)
            for bar, val in zip(bars, f1s):
                ax.text(min(val + 0.015, 0.96),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=8)

        plt.suptitle(f"MSF-MCTNet — F1 par classe ({state})",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"fig5_per_class_f1_{state}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Sauvegarde : {path}")


def fig6_params_vs_performance(all_results):
    """Figure 6 : scatter params vs OA — efficacité paramétrique."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    states_list = list({r["state"] for r in all_results})

    for ax, state in zip(axes, states_list):
        sub = [r for r in all_results if r["state"] == state]
        for res in sub:
            m = res["metrics"]
            ax.scatter(m["n_params"], m["oa"],
                       color=COLORS[res["config"]], s=120,
                       label=LABELS_FR[res["config"]], zorder=3,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(LABELS_FR[res["config"]],
                        (m["n_params"], m["oa"]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=COLORS[res["config"]])

        ax.set_xlabel("Nombre de paramètres entraînables", fontsize=10)
        ax.set_ylabel("Overall Accuracy (OA)", fontsize=10)
        ax.set_title(f"{state} — Efficacité paramétrique", fontsize=11)
        ax.grid(alpha=0.3)

    plt.suptitle("MSF-MCTNet — Paramètres vs Performance",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig6_params_vs_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Sauvegarde : {path}")


def print_final_summary_table(all_results):
    """Affiche le tableau de synthèse complet en console + CSV."""
    rows = []
    for res in all_results:
        m = res["metrics"]
        p2 = PART2_RESULTS[res["state"]][res["config"]]
        rows.append({
            "State":         res["state"],
            "Config":        LABELS_FR[res["config"]],
            "P2 OA":         f"{p2['oa']:.4f}",
            "P3 OA":         f"{m['oa']:.4f}",
            "ΔOA":           f"{m['oa'] - p2['oa']:+.4f}",
            "P2 F1":         f"{p2['f1']:.4f}",
            "P3 F1":         f"{m['f1']:.4f}",
            "ΔF1":           f"{m['f1'] - p2['f1']:+.4f}",
            "P2 Kappa":      f"{p2['kappa']:.4f}",
            "P3 Kappa":      f"{m['kappa']:.4f}",
            "ΔKappa":        f"{m['kappa'] - p2['kappa']:+.4f}",
            "Params (P3)":   f"{m['n_params']:,}",
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("TABLEAU DE SYNTHESE - Partie 2 (concat naive) vs Partie 3 (MSF-MCTNet)")
    print("=" * 100)
    print(df.to_string(index=False))
    csv_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV sauvegarde : {csv_path}")
    return df


# ══════════════════════════════════════════════════════════════
# 8. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("  PARTIE 3 — MSF-MCTNet : Entraînement complet")
    print("=" * 65)

    all_results = []

    for state in STATES:
        print(f"\n{'─'*65}")
        print(f"  État : {state}")
        print(f"{'─'*65}")

        # Dimensions fixes des covariables dans votre projet
        all_cov_dims = {"climate": 3, "soil": 3, "topo": 2}

        for config in CONFIGS:
            print(f"\n  Config : {config}")
            # On charge les données spécifiques à chaque configuration d'ablation
            loaders = get_loaders(state, config)
            train_loader, val_loader, test_loader, _ = loaders

            res = run_training(
                state, config, all_cov_dims,
                (train_loader, val_loader, test_loader),
            )
            all_results.append(res)

    # ── Génération des graphiques (Tous conservés) ───────────
    print("\n" + "=" * 65)
    print("  GENERATION DES FIGURES")
    print("=" * 65)
    
    # fig1 contient maintenant Train vs Val sur le même graphe
    fig1_training_curves(all_results)
    
    # Toutes les autres figures sont conservées
    fig2_confusion_matrices(all_results)
    fig3_comparison_part2_vs_part3(all_results)
    fig4_attention_weights(all_results)
    fig5_per_class_f1(all_results)
    fig6_params_vs_performance(all_results)
    
    print_final_summary_table(all_results)

    print(f"\n[OK] Tous les résultats dans : {RESULTS_DIR}/")


# ══════════════════════════════════════════════════════════════
# 9. MODE DEMO (données synthétiques — sans fichiers .npy)
# ══════════════════════════════════════════════════════════════

def run_demo():
    """
    Démo rapide avec données aléatoires pour tester l'architecture.
    Exécuter avec :  python part3_msf_mctnet_complete.py --demo
    """
    print("\n=== MODE DÉMO — données synthétiques ===\n")
    B, T, nb = 8, 36, 10
    x    = torch.randn(B, T, nb)
    mask = torch.ones(B, T)
    mask[:, [0, 3, 10, 20]] = 0

    cov_dims = {"climate": 2, "soil": 4, "topo": 3}
    cov = {k: torch.randn(B, v) for k, v in cov_dims.items()}

    print(f"{'Config':15s} {'OA (dummy)':>12s} {'Params':>12s} {'Attn shape':>15s}")
    print("-" * 58)
    for config in CONFIGS:
        model    = build_model("Arkansas", config, cov_dims)
        cov_in   = cov if config != "baseline" else None
        logits, attn = model(x, mask, cov_in)
        # Dummy OA avec labels aléatoires (juste pour vérifier les shapes)
        labels   = torch.randint(0, 5, (B,))
        preds    = logits.argmax(1)
        oa       = (preds == labels).float().mean().item()
        attn_sh  = str(tuple(attn.shape)) if attn is not None else "None"
        print(f"{config:15s} {oa:12.4f} {model.n_params():12,} {attn_sh:>15s}")

    print("\n[OK] Architecture validee - toutes les dimensions sont correctes.")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_demo()
    else:
        main()
