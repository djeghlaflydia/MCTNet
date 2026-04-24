import os
import glob
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_DIR  = r'c:\Users\hp\Desktop\MCTNet\Donnees'
BANDS     = ['B2','B3','B4','B5',
             'B6','B7','B8',
             'B8A','B11','B12']
N_BANDS   = 10
N_TIMES   = 36
N_CLASSES = 6
LABEL_MAP = {
    'Grapes':     0,
    'Rice':       1,
    'Alfalfa':    2,
    'Almonds':    3,
    'Pistachios': 4,
    'Others':     5
}

# ─────────────────────────────────────────────────────────────
#  ÉTAPE 1 : CHARGER ET FUSIONNER LES 36 CSV PAR ZONE
# ─────────────────────────────────────────────────────────────
def load_zone(zone_name):
    print(f"\nChargement {zone_name}...")

    # On cherche dans le sous-dossier MCTNet_{zone_name}
    zone_path = os.path.join(DATA_DIR, f'MCTNet_{zone_name}')
    files = sorted(glob.glob(
        os.path.join(zone_path, f'{zone_name}_t*.csv')
    ))

    if len(files) == 0:
        raise FileNotFoundError(f"Aucun fichier trouvé pour {zone_name} dans {zone_path}")

    print(f"  {len(files)} fichiers trouvés")

    df_ref = pd.read_csv(files[0])

    # Utiliser les colonnes directes longitude/latitude
    df_main = df_ref[['pixel_id','label_name',
                       'longitude','latitude']].copy()
    
    # Renommer pour rester compatible avec la suite si besoin (facultatif)
    df_main = df_main.rename(columns={'longitude': 'lon', 'latitude': 'lat', 'label_name': 'class_name'})

    for i, f in enumerate(files):
        t   = i + 1
        df  = pd.read_csv(f)

        for band in BANDS:
            df_main[f'T{t:02d}_{band}'] = df[band].values

        # missing = 1 - valid
        df_main[f'T{t:02d}_missing'] = 1 - df['valid'].values

        if (i + 1) % 12 == 0:
            print(f"  {i+1}/36 fichiers chargés...")

    print(f"  Shape final : {df_main.shape}")
    return df_main


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 2 : CONSTRUIRE LES TENSEURS X, Y, MASK
# ─────────────────────────────────────────────────────────────
def build_tensors(df):
    N = len(df)

    # Tenseur des bandes : (N, T, C)
    X = np.zeros((N, N_TIMES, N_BANDS), dtype=np.float32)
    for t in range(N_TIMES):
        for c, band in enumerate(BANDS):
            col        = f'T{t+1:02d}_{band}'
            X[:, t, c] = df[col].values

    # Tenseur du masque : (N, T)
    mask = np.zeros((N, N_TIMES), dtype=np.float32)
    for t in range(N_TIMES):
        col          = f'T{t+1:02d}_missing'
        mask[:, t]   = df[col].values

    # Labels : (N,)
    # On utilise class_name car on a renommé label_name -> class_name
    Y = df['class_name'].map(LABEL_MAP).values.astype(np.int64)

    return (
        torch.FloatTensor(X),
        torch.LongTensor(Y),
        torch.FloatTensor(mask)
    )


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 3 : DATASET PYTORCH
# ─────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, X, Y, mask):
        self.X    = X
        self.Y    = Y
        self.mask = mask

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
         return self.X[idx], self.mask[idx], self.Y[idx]
        


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 4 : SPLIT TRAIN / VAL / TEST
#  te9sima : 240 train + 60 val par classe
# ─────────────────────────────────────────────────────────────
def split_data(df, X, Y, mask):
    idx_train = []
    idx_val   = []
    idx_test  = []

    classes = df['class_name'].unique()

    for cls in classes:
        idx_cls       = df[df['class_name'] == cls].index.tolist()
        idx_cls_local = [df.index.get_loc(i) for i in idx_cls]

        np.random.seed(42)
        np.random.shuffle(idx_cls_local)

        # On prend 240 pour train, 60 pour val, le reste pour test
        # On vérifie si on a assez d'échantillons
        n_cls = len(idx_cls_local)
        n_train = min(240, int(n_cls * 0.15)) # Fallback si pas assez
        n_val   = min(60, int(n_cls * 0.05))
        
        # Mais le papier dit 1440 train (240*6) / 360 val (60*6)
        # On garde 240/60 si possible
        idx_train += idx_cls_local[:240]
        idx_val   += idx_cls_local[240:300]
        idx_test  += idx_cls_local[300:]

    print(f"\nSplit train/val/test  :")
    print(f"  train : {len(idx_train):5d}  " )
    print(f"  val   : {len(idx_val):5d}  " )
    print(f"  test  : {len(idx_test):5d}  " )

    datasets = {}
    for name, idx in [('train', idx_train),
                      ('val',   idx_val),
                      ('test',  idx_test)]:
        datasets[name] = CropDataset(
            X[idx], Y[idx], mask[idx]
        )

    return datasets

# ─────────────────────────────────────────────────────────────
#  PIPELINE COMPLET
# ─────────────────────────────────────────────────────────────
def prepare_all_data():
    # Charger la zone california
    df_all = load_zone('california')
    
    print(f"\nTotal pixels : {len(df_all)}")

    df_all.to_csv('df_all.csv',  index=False)
    print("[OK] df_all.csv sauvegarde")

    
    # Construire les tenseurs
    print("\nConstruction des tenseurs...")
    X, Y, mask = build_tensors(df_all)
    print(f"  X    : {X.shape}")
    print(f"  Y    : {Y.shape}")
    print(f"  mask : {mask.shape}")

    # Vérifier la distribution
    print("\nDistribution des classes :")
    classes = list(LABEL_MAP.keys())
    for i, cls in enumerate(classes):
        n = (Y == i).sum().item()
        print(f"  {cls:<12} : {n:5d}  ({n/len(Y)*100:.1f}%)")

    # Séparer train/val/test
    datasets = split_data(df_all, X, Y, mask)

    return datasets

if __name__ == '__main__':
    datasets = prepare_all_data()
    torch.save(datasets, 'datasets.pt')
    print("\n[OK] Donnees pretes!")
    print(f"  train : {len(datasets['train'])} pixels")
    print(f"  val   : {len(datasets['val'])}   pixels")
    print(f"  test  : {len(datasets['test'])}  pixels")