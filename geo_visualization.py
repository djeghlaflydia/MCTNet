import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.colors import ListedColormap

# =========================================================
# CONFIG
# =========================================================

SEED = 42

CLASSES = [
    "Others",
    "Grapes",
    "Rice",
    "Alfalfa",
    "Almonds",
    "Pistachios"
]

COLOR_LIST = [
    "#E41A1C", # Others
    "#377EB8", # Grapes
    "#4DAF4A", # Rice
    "#984EA3", # Alfalfa
    "#FF7F00", # Almonds
    "#F781BF"  # Pistachios
]

# =========================================================
# LOAD LABELS / PREDICTIONS
# =========================================================

y_true = np.load(
    "part3_results/California_all/labels.npy"
)

y_pred = np.load(
    "part3_results/California_all/preds.npy"
)

# =========================================================
# GEO MAP FUNCTION
# =========================================================

def plot_geo_maps(y_true, y_pred):

    # =====================================================
    # Charger un fichier merged Arkansas // California
    # =====================================================

    csv_file = (
        "Donnees_Merged/MCTNet_california/"
        "california_merged_t00.csv"
    )

    if not os.path.exists(csv_file):
        print(f"❌ Fichier introuvable : {csv_file}")
        return

    df = pd.read_csv(csv_file)

    print("Colonnes disponibles :")
    print(df.columns.tolist())

    # =====================================================
    # Vérifier les colonnes géographiques
    # =====================================================

    required_cols = ['pixel_id', 'label_name', 'longitude', 'latitude']

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"❌ Colonnes manquantes : {missing}")
        return

    # =====================================================
    # Garder uniquement les colonnes utiles
    # =====================================================

    df = df[['pixel_id', 'label_name', 'longitude', 'latitude']]

    # =====================================================
    # Reproduire le split test
    # =====================================================

    np.random.seed(SEED)

    idx_test = []

    for cls in df['label_name'].unique():

        idx_cls = df[df['label_name'] == cls].index.tolist()

        np.random.shuffle(idx_cls)

        idx_test += idx_cls[300:]

    df_test = df.iloc[idx_test].reset_index(drop=True)

    # =====================================================
    # Adapter tailles
    # =====================================================

    n = min(len(df_test), len(y_true), len(y_pred))

    df_test = df_test.iloc[:n]

    df_test['y_true'] = y_true[:n]
    df_test['y_pred'] = y_pred[:n]

    df_test['correct'] = (
        df_test['y_true'] == df_test['y_pred']
    )

    # =====================================================
    # COLOR MAP
    # =====================================================

    cmap_classes = ListedColormap(COLOR_LIST)

    legend_patches = [
        mpatches.Patch(
            color=COLOR_LIST[i],
            label=CLASSES[i]
        )
        for i in range(len(CLASSES))
    ]

    error_patches = [
        mpatches.Patch(
            color='#4CAF50',
            label='Correct'
        ),
        mpatches.Patch(
            color='#F44336',
            label='Erreur'
        ),
    ]

    # =====================================================
    # FIGURE
    # =====================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 8)
    )

    fig.suptitle(
        'California — Ground Truth / Prediction / Errors',
        fontsize=16,
        fontweight='bold'
    )

    lon = df_test['longitude'].values
    lat = df_test['latitude'].values

    true_lbl = df_test['y_true'].values
    pred_lbl = df_test['y_pred'].values

    correct = df_test['correct'].values

    s = max(2, min(10, 4000 // len(df_test)))

    # =====================================================
    # Ground Truth
    # =====================================================

    ax = axes[0]

    ax.scatter(
        lon,
        lat,
        c=true_lbl,
        cmap=cmap_classes,
        vmin=0,
        vmax=len(CLASSES)-1,
        s=s,
        alpha=0.85,
        linewidths=0
    )

    ax.set_title(
        'Ground Truth',
        fontsize=12,
        fontweight='bold'
    )

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.grid(alpha=0.2)

    # =====================================================
    # Prediction
    # =====================================================

    ax = axes[1]

    ax.scatter(
        lon,
        lat,
        c=pred_lbl,
        cmap=cmap_classes,
        vmin=0,
        vmax=len(CLASSES)-1,
        s=s,
        alpha=0.85,
        linewidths=0
    )

    ax.set_title(
        'Prediction',
        fontsize=12,
        fontweight='bold'
    )

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.grid(alpha=0.2)

    # =====================================================
    # Errors
    # =====================================================

    ax = axes[2]

    colors_err = np.where(
        correct,
        '#4CAF50',
        '#F44336'
    )

    ax.scatter(
        lon,
        lat,
        c=colors_err,
        s=s,
        alpha=0.85,
        linewidths=0
    )

    n_ok = correct.sum()
    n_tot = len(correct)

    ax.set_title(
        f'Errors\n'
        f'Correct : {n_ok}/{n_tot} ({n_ok/n_tot:.1%})',
        fontsize=12,
        fontweight='bold'
    )

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.grid(alpha=0.2)

    # =====================================================
    # LEGEND
    # =====================================================

    all_patches = (
        legend_patches + error_patches
    )

    fig.legend(
        handles=all_patches,
        loc='lower center',
        ncol=len(CLASSES) + 2,
        fontsize=10,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================

    save_path = (
        "part3_results/"
        "geo_maps_california.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"OK: Sauvegardé : {save_path}")

    plt.show()


# =========================================================
# RUN
# =========================================================

plot_geo_maps(y_true, y_pred)