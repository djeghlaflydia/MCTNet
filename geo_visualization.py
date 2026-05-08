import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42

# Mapping des classes et couleurs (Cohérent avec le reste du projet)
CA_CLASSES = ["Others", "Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios"]
CA_COLORS  = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#8c564b"]

AR_CLASSES = ["Others", "Corn", "Cotton", "Rice", "Soybeans"]
AR_COLORS  = ["#9467bd", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Mapping des labels demandés par l'utilisateur
CONFIG_LABELS = {
    "baseline":   "Sentinel-2 bands only",
    "s2_climate": "Sentinel-2 + climate variables",
    "s2_soil":    "Sentinel-2 + soil variables",
    "s2_topo":    "Sentinel-2 + topographic variables",
    "all":        "Sentinel-2 + climate + soil + topography"
}

def plot_geo_maps(state, config_key):
    """
    Génère les cartes géographiques pour un état et une configuration donnés.
    Compare la Vérité Terrain vs Prédiction et affiche les erreurs.
    """
    
    # 1. Définition des dossiers de résultats
    results_folder = os.path.join("part3_results", f"{state}_{config_key}")
    y_true_path = os.path.join(results_folder, "labels.npy")
    y_pred_path = os.path.join(results_folder, "preds.npy")
    
    # Vérifier si les résultats existent pour cette configuration
    if not os.path.exists(y_true_path) or not os.path.exists(y_pred_path):
        return

    print(f"-> Traitement de {state} | Configuration: {config_key}...")

    # 2. Chargement des prédictions (Numpy)
    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    # 3. Chargement des coordonnées géographiques depuis les données originales
    csv_file = os.path.join("Donnees_Merged", f"MCTNet_{state.lower()}", f"{state.lower()}_merged_t00.csv")
    if not os.path.exists(csv_file):
        print(f"⚠️ Fichier source introuvable : {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    
    # 4. Reproduction exacte du split TEST utilisé lors de l'entraînement
    # On ignore les 300 premiers échantillons de chaque classe (utilisés pour train/val)
    np.random.seed(SEED)
    idx_test = []
    for lbl in sorted(df['label'].unique()):
        idx_cls = df[df['label'] == lbl].index.tolist()
        np.random.shuffle(idx_cls)
        idx_test += idx_cls[300:] # Le test set correspond au reste
        
    df_test = df.iloc[idx_test].reset_index(drop=True)
    
    # Alignement des tailles entre les coordonnées et les prédictions
    n = min(len(df_test), len(y_true))
    df_test = df_test.iloc[:n]
    df_test['y_true']  = y_true[:n]
    df_test['y_pred']  = y_pred[:n]
    df_test['correct'] = (df_test['y_true'] == df_test['y_pred'])

    # 5. Configuration visuelle
    classes = CA_CLASSES if state == "California" else AR_CLASSES
    colors  = CA_COLORS if state == "California" else AR_COLORS
    
    cmap_classes = ListedColormap(colors)
    legend_patches = [mpatches.Patch(color=colors[i], label=classes[i]) for i in range(len(classes))]
    error_patches = [
        mpatches.Patch(color='#4CAF50', label='Correct'),
        mpatches.Patch(color='#F44336', label='Error'),
    ]

    # 6. Création de la figure (3 colonnes)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Titre dynamique selon la configuration
    config_desc = CONFIG_LABELS.get(config_key, config_key)
    fig.suptitle(f'{state} — {config_desc}', fontsize=18, fontweight='bold', y=0.98)

    # Données à tracer
    lon = df_test['longitude'].values
    lat = df_test['latitude'].values
    true_lbl = df_test['y_true'].values
    pred_lbl = df_test['y_pred'].values
    correct  = df_test['correct'].values
    
    # Taille adaptative des points
    s = max(2, min(10, 4000 // len(df_test)))

    # --- Sous-graphe 1 : Vérité terrain (Ground Truth) ---
    ax = axes[0]
    ax.scatter(lon, lat, c=true_lbl, cmap=cmap_classes, vmin=0, vmax=len(classes)-1, s=s, alpha=0.85, linewidths=0)
    ax.set_title('Ground Truth (Réel)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.grid(alpha=0.2)

    # --- Sous-graphe 2 : Prédiction du modèle ---
    ax = axes[1]
    ax.scatter(lon, lat, c=pred_lbl, cmap=cmap_classes, vmin=0, vmax=len(classes)-1, s=s, alpha=0.85, linewidths=0)
    ax.set_title('Prediction (Modèle)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.grid(alpha=0.2)

    # --- Sous-graphe 3 : Localisation des erreurs ---
    ax = axes[2]
    colors_err = np.where(correct, '#4CAF50', '#F44336') # Vert = OK, Rouge = Erreur
    ax.scatter(lon, lat, c=colors_err, s=s, alpha=0.85, linewidths=0)
    
    n_ok, n_tot = correct.sum(), len(correct)
    ax.set_title(f'Classification Errors (OA: {n_ok/n_tot:.1%})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.grid(alpha=0.2)

    # Légende commune en bas
    all_patches = legend_patches + error_patches
    fig.legend(handles=all_patches, loc='lower center', ncol=len(classes) + 2, 
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    
    # Sauvegarde de l'image
    save_name = f"geo_maps_{state.lower()}_{config_key}.png"
    save_path = os.path.join("part3_results", save_name)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"OK: Carte sauvegardée : {save_path}")

if __name__ == "__main__":
    # Générer automatiquement les cartes pour chaque état et chaque configuration d'ablation
    for state in ["Arkansas", "California"]:
        for config in CONFIG_LABELS.keys():
            plot_geo_maps(state, config)