"""
=======================================================================
PART 1 — STEP 3 : DATA EXPLORATION
Project: Deep Learning for Crop Classification Using Multi-Source
         Satellite Data (M1 SII 2025/2026 — USTHB)
Paper  : MCTNet — Wang et al., 2024

Ce script suppose que les CSV exportés depuis GEE sont disponibles 
dans le dossier Donnees, organisés comme suit :
    Donnees/
        MCTNet_arkansas/   → 36 CSV (arkansas_tXX_MM_dN.csv)
        MCTNet_california/ → 36 CSV (california_tXX_MM_dN.csv)

Explorations réalisées :
    1. Distribution des classes (label)
    2. Taux de valeurs manquantes par date et par bande
    3. Courbes NDVI temporelles par culture (Fig. 2 du papier)
    4. Courbes EVI temporelles par culture
    5. Profils temporels moyens par bande par classe
    6. Boxplots NDVI par classe
    7. Heatmap de corrélation inter-bandes
    8. Patterns temporels : variance NDVI par culture
=======================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
DONNEES_DIR = "./Donnees"        
OUTPUT_DIR = "./exploration_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Correspondance label → nom de culture (Ajusté selon les données réelles)
AR_CLASSES = {0: "Others", 1: "Corn", 2: "Cotton", 3: "Rice", 4: "Soybeans"}
CA_CLASSES = {0: "Others", 1: "Grapes", 2: "Rice", 3: "Alfalfa", 4: "Almonds", 5: "Pistachios"}

# Palette cohérente
AR_COLORS = {0: "#7f7f7f", 1: "#2ca02c", 2: "#8c564b", 3: "#17becf", 4: "#ff7f0e"}
CA_COLORS = {0: "#7f7f7f", 1: "#9467bd", 2: "#17becf", 3: "#e377c2", 4: "#1f77b4", 5: "#bcbd22"}

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_NAMES = {
    "B2": "Blue", "B3": "Green", "B4": "Red",
    "B5": "Red Edge 1", "B6": "Red Edge 2", "B7": "Red Edge 3",
    "B8": "NIR", "B8A": "Red Edge 4",
    "B11": "SWIR 1", "B12": "SWIR 2"
}

# -----------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# -----------------------------------------------------------------------
def load_state_data(state):
    """Charge et concatène les CSV d'un état → DataFrame unique."""
    
    folder = f"MCTNet_{state.lower()}"
    pattern = os.path.join(DONNEES_DIR, folder, "*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"✗ Aucun CSV trouvé dans {pattern}")
        return pd.DataFrame()

    print(f"  Loading {len(files)} CSV files for {state}...")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date_str"])
    df_all["doy"]  = df_all["date"].dt.dayofyear
    
    # Masque de données manquantes (valid=0 ou toutes les bandes=0)
    df_all["missing"] = ((df_all[BANDS] == 0).all(axis=1) | (df_all["valid"] == 0)).astype(int)

    return df_all


def load_all():
    """Charge Arkansas et California."""
    AR = load_state_data("Arkansas")
    CA = load_state_data("California")
    return AR, CA


# -----------------------------------------------------------------------
# 2. CALCUL DES INDICES DE VÉGÉTATION
# -----------------------------------------------------------------------
def add_vegetation_indices(df):
    """Ajoute NDVI, EVI, NDWI au DataFrame."""
    if df.empty: return df
    
    # Remplacer 0 par NaN pour éviter divisions par zéro
    B4  = df["B4"].replace(0, np.nan)   # Red
    B8  = df["B8"].replace(0, np.nan)   # NIR
    B2  = df["B2"].replace(0, np.nan)   # Blue
    B3  = df["B3"].replace(0, np.nan)   # Green

    # NDVI = (NIR - Red) / (NIR + Red)
    df["NDVI"] = (B8 - B4) / (B8 + B4 + 1e-8)

    # EVI  = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    df["EVI"]  = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + 1e-8)

    # NDWI = (Green - NIR) / (Green + NIR)
    df["NDWI"] = (B3 - B8) / (B3 + B8 + 1e-8)

    return df


# -----------------------------------------------------------------------
# 3. EXPLORATION — DISTRIBUTION DES CLASSES
# -----------------------------------------------------------------------
def plot_class_distribution(AR, CA):
    """Barplot de la distribution des classes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Class Distribution (Unique Pixels)", fontsize=13, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        # Compter par classe sur un seul timestamp (évite comptage x36)
        df_t0  = df[df["timestep"] == 0].copy()
        counts = df_t0["label"].value_counts().sort_index()
        labels = [class_map.get(i, f"Class {i}") for i in counts.index]
        colors = [color_map.get(i, "gray")        for i in counts.index]
        
        bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)
        
        # Pourcentage
        total = counts.sum()
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{(height/total)*100:.1f}%', ha='center', va='bottom', fontsize=9)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Number of pixels")
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_class_distribution.png"), dpi=150)
    plt.show()


# -----------------------------------------------------------------------
# 4. EXPLORATION — VALEURS MANQUANTES
# -----------------------------------------------------------------------
def plot_missing_values(AR, CA):
    """Taux de valeurs manquantes par date."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Missing Data Rate per Date (masked pixels)", fontsize=12, fontweight="bold")

    for ax, df, title in [
        (axes[0], AR, "Arkansas"),
        (axes[1], CA, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        miss_rate = df.groupby("date")["missing"].mean() * 100
        ax.bar(miss_rate.index, miss_rate.values, color="#d62728", alpha=0.7, width=4)
        ax.axhline(miss_rate.mean(), color="black", linestyle="--", label=f"Mean: {miss_rate.mean():.1f}%")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Missing rate (%)")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_missing_values.png"), dpi=150)
    plt.show()


# -----------------------------------------------------------------------
# 5. EXPLORATION — COURBES NDVI TEMPORELLES
# -----------------------------------------------------------------------
def plot_ndvi_timeseries(AR, CA):
    """NDVI moyen par culture en fonction du Day of Year (DOY)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Mean NDVI Time-Series Profiles (Fig. 2 Reproduction)", fontsize=12, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        df_valid = df[df["missing"] == 0].copy()

        for cls, name in class_map.items():
            df_cls  = df_valid[df_valid["label"] == cls]
            if df_cls.empty: continue
            ndvi_by_doy = df_cls.groupby("doy")["NDVI"].mean()
            ax.plot(ndvi_by_doy.index, ndvi_by_doy.values,
                    label=name, color=color_map[cls], linewidth=2, marker="o", markersize=4)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Mean NDVI")
        ax.set_xlim(0, 366)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_ndvi_timeseries.png"), dpi=150)
    plt.show()


# -----------------------------------------------------------------------
# 6. EXPLORATION — COURBES EVI TEMPORELLES
# -----------------------------------------------------------------------
def plot_evi_timeseries(AR, CA):
    """EVI moyen par culture."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Mean EVI Time-Series Profiles", fontsize=12, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        df_valid = df[df["missing"] == 0].copy()

        for cls, name in class_map.items():
            df_cls  = df_valid[df_valid["label"] == cls]
            if df_cls.empty: continue
            evi_by_doy = df_cls.groupby("doy")["EVI"].mean()
            ax.plot(evi_by_doy.index, evi_by_doy.values,
                    label=name, color=color_map[cls], linewidth=2, marker="s", markersize=4)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Mean EVI")
        ax.set_xlim(0, 366)
        ax.legend(fontsize=9, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_evi_timeseries.png"), dpi=150)
    plt.show()


# -----------------------------------------------------------------------
# 7. EXPLORATION — PROFILS SPECTRAUX
# -----------------------------------------------------------------------
def plot_spectral_profiles(AR, CA, doy_target=200):
    """Profil spectral moyen par classe à un pic de végétation."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Mean Spectral Profiles at DOY ≈ {doy_target}", fontsize=12, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        doys_avail = df["doy"].unique()
        doy_sel    = doys_avail[np.argmin(np.abs(doys_avail - doy_target))]
        df_t       = df[(df["doy"] == doy_sel) & (df["missing"] == 0)]

        band_labels = [BAND_NAMES[b] for b in BANDS]

        for cls, name in class_map.items():
            df_cls = df_t[df_t["label"] == cls]
            if df_cls.empty: continue
            means = df_cls[BANDS].mean().values
            ax.plot(band_labels, means, label=name, color=color_map[cls], linewidth=2, marker="o")

        ax.set_title(f"{title} (DOY {doy_sel})", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_spectral_profiles.png"), dpi=150)
    plt.show()


# -----------------------------------------------------------------------
# 8. EXPLORATION — BOXPLOTS NDVI PAR CLASSE
# -----------------------------------------------------------------------
def plot_ndvi_boxplots(AR, CA):
    """Distribution de NDVI par classe (boxplot)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("NDVI Distribution per Crop Class", fontsize=12, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        df_valid = df[df["missing"] == 0].copy()
        df_valid["crop_name"] = df_valid["label"].map(class_map)

        order = [class_map[k] for k in sorted(class_map.keys())]
        palette = {class_map[k]: color_map[k] for k in class_map}

        sns.boxplot(data=df_valid, x="crop_name", y="NDVI", order=order, palette=palette, ax=ax,
                    flierprops={"marker": ".", "alpha": 0.3, "markersize": 2})
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_ndvi_boxplots.png"), dpi=150)
    plt.show()


# -----------------------------------------------------------------------
# 9. EXPLORATION — VARIANCE TEMPORELLE DU NDVI
# -----------------------------------------------------------------------
def plot_ndvi_variance(AR, CA):
    """Variance du NDVI par culture montre la séparabilité temporelle."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("NDVI Standard Deviation by Crop Type", fontsize=12, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        df_valid = df[df["missing"] == 0].copy()

        for cls, name in class_map.items():
            df_cls = df_valid[df_valid["label"] == cls]
            if df_cls.empty: continue
            std_by_doy = df_cls.groupby("doy")["NDVI"].std()
            ax.plot(std_by_doy.index, std_by_doy.values, label=name, color=color_map[cls], linewidth=1.5)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("NDVI Std Dev")
        ax.set_xlim(0, 366)
        ax.legend(fontsize=9, ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_ndvi_variance.png"), dpi=150)
    plt.show()


# -----------------------------------------------------------------------
# RÉSUMÉ STATISTIQUE
# -----------------------------------------------------------------------
def print_summary(AR, CA):
    """Statistiques descriptives générales."""
    for name, df, class_map in [("ARKANSAS", AR, AR_CLASSES), ("CALIFORNIA", CA, CA_CLASSES)]:
        print("\n" + "="*40)
        print(f"SUMMARY — {name}")
        print("="*40)
        if df.empty:
            print("No data loaded."); continue
            
        df_t0 = df[df["timestep"] == 0]
        print(f"Total unique pixels: {len(df_t0):,}")
        print(f"Class distribution:")
        counts = df_t0["label"].value_counts().sort_index()
        for cls, cnt in counts.items():
            cname = class_map.get(cls, f"Class {cls}")
            print(f"  - {cname:10s} : {cnt:5,} pixels ({cnt/len(df_t0)*100:.1f}%)")
        
        ndvi_valid = df[df["missing"] == 0]["NDVI"]
        print(f"NDVI Mean: {ndvi_valid.mean():.3f}")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3 — DATA EXPLORATION")
    print("=" * 60)

    # 1. Chargement
    print("\n[1/5] Loading data...")
    AR, CA = load_all()

    # 2. Calcul des indices
    print("[2/5] Computing indices...")
    AR = add_vegetation_indices(AR)
    CA = add_vegetation_indices(CA)

    # 3. Résumé
    print_summary(AR, CA)

    # 4. Plots
    print("\n[4/5] Generating plots...")
    plot_class_distribution(AR, CA)
    plot_missing_values(AR, CA)
    plot_ndvi_timeseries(AR, CA)
    plot_evi_timeseries(AR, CA)
    plot_spectral_profiles(AR, CA, doy_target=200)
    plot_ndvi_boxplots(AR, CA)
    plot_ndvi_variance(AR, CA)

    # Corrélation bands
    if not AR.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(AR[AR["missing"] == 0][BANDS].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Band Correlation - Arkansas")
        plt.savefig(os.path.join(OUTPUT_DIR, "08_correlation_ar.png"))
        plt.show()

    print(f"\n✅ Exploration complete. Plots saved in {OUTPUT_DIR}/")