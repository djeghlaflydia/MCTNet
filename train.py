"""
Entraînement MCTNet — Californie sans sol
Wang et al. (2024)

Hyperparamètres EXACTS du papier :
  epochs     = 200
  batch_size = 32
  lr         = 0.001
  optimizer  = Adam
  train      = 1440
  val        =  360
  test       = 8200
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score,
                             cohen_kappa_score,
                             f1_score,
                             confusion_matrix)
from prepa import prepare_all_data, CropDataset
from model_ca_bands import MCTNet

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION — EXACTEMENT LE PAPIER
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'batch_size'  : 32,
    'epochs'      : 200,
    'lr'          : 0.001,
    'n_classes'   : 6,
    'in_channels' : 10,
    'n_head'      : 5,
    'n_stage'     : 3,
    'device'      : 'cpu',
    'patience'    : 20,
}

CLASSES = ['Grapes','Rice','Alfalfa',
           'Almonds','Pistachios','Others']

# ─────────────────────────────────────────────────────────────
#  CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────
print("Chargement des données...")
torch.serialization.add_safe_globals([CropDataset])

if os.path.exists('datasets.pt'):
    datasets = torch.load('datasets.pt', weights_only=False)
    print("[OK] datasets.pt charge !")
else:
    datasets = prepare_all_data()

loaders = {
    'train': DataLoader(datasets['train'],
                        batch_size=CONFIG['batch_size'],
                        shuffle=True),
    'val':   DataLoader(datasets['val'],
                        batch_size=CONFIG['batch_size'],
                        shuffle=False),
    'test':  DataLoader(datasets['test'],
                        batch_size=CONFIG['batch_size'],
                        shuffle=False)
}

print(f"Train : {len(datasets['train'])} pixels ( {len(loaders['train'])} batches )")
print(f"Val   : {len(datasets['val'])} pixels ( {len(loaders['val'])} batches )")
print(f"Test  : {len(datasets['test'])} pixels ( {len(loaders['test'])} batches )")

# ─────────────────────────────────────────────────────────────
#  MODÈLE
# ─────────────────────────────────────────────────────────────
model = MCTNet(
    in_channels=CONFIG['in_channels'],
    n_classes  =CONFIG['n_classes'],
    n_head     =CONFIG['n_head'],
    n_stage    =CONFIG['n_stage']
).to(CONFIG['device'])

n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"\nMCTNet — {n_params:,} paramètres")
print(f"Architecture :")
print(f"  Stage 1 : (batch, 36, 10) to (batch, 18, 20)")
print(f"  Stage 2 : (batch, 18, 20) to (batch,  9, 40)")
print(f"  Stage 3 : (batch,  9, 40) to (batch,  4, 80)")
print(f"  MLP     : (batch, 80)     to (batch,  6)")

# ─────────────────────────────────────────────────────────────
#  LOSS ET OPTIMISEUR — EXACTEMENT LE PAPIER
# ─────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG['lr'],
    weight_decay=1e-2
)

# ─────────────────────────────────────────────────────────────
#  FONCTION D'ÉVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, mask_batch, y_batch in loader:
            x_batch    = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch    = y_batch.to(device)

            outputs = model(x_batch, mask_batch)
            loss    = criterion(outputs, y_batch)
            preds   = outputs.argmax(dim=1)

            total_loss  += loss.item()
            all_preds   += preds.cpu().tolist()
            all_labels  += y_batch.cpu().tolist()

    loss  = total_loss / len(loader)
    oa    = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1    = f1_score(all_labels, all_preds, average='macro')

    return loss, oa, kappa, f1, all_labels, all_preds


# ─────────────────────────────────────────────────────────────
#  BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  ENTRAÎNEMENT MCTNet — Wang et al. (2024)")
print(f"  epochs={CONFIG['epochs']} | "
      f"batch={CONFIG['batch_size']} | "
      f"lr={CONFIG['lr']} | "
      f"optimizer=Adam")
print(f"  patience={CONFIG['patience']}")
print("="*65)

best_val_oa    = 0.0
best_epoch     = 0
patience_count = 0
history        = {
    'train_loss': [], 'val_loss': [],
    'train_oa':   [], 'val_oa':   [],
    'train_f1':   [], 'val_f1':   []
}

for epoch in range(1, CONFIG['epochs'] + 1):

    # ── Entraînement ────────────────────────────────────
    model.train()
    train_loss   = 0.0
    train_preds  = []
    train_labels = []

    for x_batch, mask_batch, y_batch in loaders['train']:
        x_batch    = x_batch.to(CONFIG['device'])
        mask_batch = mask_batch.to(CONFIG['device'])
        y_batch    = y_batch.to(CONFIG['device'])

        optimizer.zero_grad()
        outputs = model(x_batch, mask_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        preds         = outputs.argmax(dim=1)
        train_loss   += loss.item()
        train_preds  += preds.cpu().tolist()
        train_labels += y_batch.cpu().tolist()

    train_loss /= len(loaders['train'])
    train_oa    = accuracy_score(train_labels, train_preds)
    train_f1    = f1_score(train_labels, train_preds,
                           average='macro')

    # ── Validation ──────────────────────────────────────
    val_loss, val_oa, val_kappa, val_f1, _, _ = evaluate(
        model, loaders['val'], CONFIG['device']
    )

    # Historique
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_oa'].append(train_oa)
    history['val_oa'].append(val_oa)
    history['train_f1'].append(train_f1)
    history['val_f1'].append(val_f1)

    # ── Meilleur modèle + Early Stopping ────────────────
    if val_oa > best_val_oa:
        best_val_oa    = val_oa
        best_epoch     = epoch
        patience_count = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_count += 1

    if patience_count >= CONFIG['patience']:
        print(f"\n[!]  Early Stopping a l'epoque {epoch} !")
        print(f"   Pas d'amélioration depuis "
              f"{CONFIG['patience']} époques.")
        print(f"   Meilleur Val OA : {best_val_oa:.4f} "
              f"à l'époque {best_epoch}")
        break

    # ── Affichage toutes les 10 époques ─────────────────
    if epoch % 10 == 0 or epoch == 1:
        print(f"Époque {epoch:3d}/{CONFIG['epochs']} | "
              f"Train Loss:{train_loss:.4f} OA:{train_oa:.4f} | "
              f"Val Loss:{val_loss:.4f} OA:{val_oa:.4f} "
              f"F1:{val_f1:.4f} | "
              f"Patience:{patience_count}/{CONFIG['patience']}")

print(f"\n[OK] Meilleur modele : epoque {best_epoch} ")

# ─────────────────────────────────────────────────────────────
#  ÉVALUATION FINALE — TEST SET
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  ÉVALUATION FINALE — TEST SET")
print("="*65)

model.load_state_dict(torch.load('best_model.pt',
                                 weights_only=True))

test_loss, test_oa, test_kappa, test_f1, \
    y_true, y_pred = evaluate(
        model, loaders['test'], CONFIG['device']
    )

print(f"\nRésultats sur le test set :")
print(f"  OA    : {test_oa:.4f}   (papier: 0.8524)")
print(f"  Kappa : {test_kappa:.4f}   (papier: 0.8194)")
print(f"  F1    : {test_f1:.4f}   (papier: 0.8301)")

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
print(f"\nMatrice de confusion :")
print(f"{'':12}", end='')
for cls in CLASSES:
    print(f"{cls:14}", end='')
print()
for i, cls in enumerate(CLASSES):
    print(f"{cls:12}", end='')
    for j in range(len(CLASSES)):
        print(f"{cm[i,j]:14}", end='')
    print()

# F1 par classe
f1_per_class = f1_score(y_true, y_pred, average=None)
print(f"\nF1 par classe :")
for cls, f1 in zip(CLASSES, f1_per_class):
    print(f"  {cls:<12} : {f1:.4f}")

# ─────────────────────────────────────────────────────────────
#  COURBES D'APPRENTISSAGE
# ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# ← CORRECTION : utiliser le nombre réel d'époques
actual_epochs = len(history['train_loss'])
epochs_range  = range(1, actual_epochs + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Courbes d'apprentissage — MCTNet\n"
             "Wang et al. (2024) — Californie",
             fontsize=13, fontweight='bold')

# Loss
axes[0].plot(epochs_range, history['train_loss'],
             label='Train', color='#2196F3', linewidth=1.5)
axes[0].plot(epochs_range, history['val_loss'],
             label='Val',   color='#F44336', linewidth=1.5)
axes[0].axvline(x=best_epoch, color='green',
                linestyle='--', linewidth=1.5,
                label=f'Best ({best_epoch})')
axes[0].set_title('Loss')
axes[0].set_xlabel('Époque')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# OA
axes[1].plot(epochs_range, history['train_oa'],
             label='Train', color='#2196F3', linewidth=1.5)
axes[1].plot(epochs_range, history['val_oa'],
             label='Val',   color='#F44336', linewidth=1.5)
axes[1].axhline(y=0.8524, color='green',
                linestyle='--', linewidth=1.5,
                label='Papier (0.8524)')
axes[1].axvline(x=best_epoch, color='purple',
                linestyle='--', linewidth=1.5,
                label=f'Best ({best_epoch})')
axes[1].set_title('Overall Accuracy (OA)')
axes[1].set_xlabel('Époque')
axes[1].set_ylabel('OA')
axes[1].legend()
axes[1].grid(alpha=0.3)

# F1
axes[2].plot(epochs_range, history['train_f1'],
             label='Train', color='#2196F3', linewidth=1.5)
axes[2].plot(epochs_range, history['val_f1'],
             label='Val',   color='#F44336', linewidth=1.5)
axes[2].axhline(y=0.8301, color='green',
                linestyle='--', linewidth=1.5,
                label='Papier (0.8301)')
axes[2].set_title('F1 Score (macro)')
axes[2].set_xlabel('Époque')
axes[2].set_ylabel('F1')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150,
            bbox_inches='tight')
print("\n[OK] Sauvegarde : learning_curves.png")
plt.show()

print("\n[OK] Entrainement termine !")
print(f"  Meilleur modèle : best_model.pt (époque {best_epoch})")
print("="*65)