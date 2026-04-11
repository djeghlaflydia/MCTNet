"""
=======================================================================
PART 1 — STEP 3 : DATA EXPLORATION
Project: Deep Learning for Crop Classification Using Multi-Source
         Satellite Data (M1 SII 2025/2026 — USTHB)
Paper  : MCTNet — Wang et al., 2024

Ce script suppose que les CSV exportés depuis GEE sont disponibles
dans le dossier Donnees, organisés comme suit :
    Donnees/
        Arkansas/zone1/  → 36 CSV (Arkansas_zone1_2021-MM-DD_csvN.csv)
        Arkansas/zone2/  → 36 CSV
        California/zone1/→ 36 CSV
        California/zone2/→ 36 CSV

Chaque CSV contient les colonnes :
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12,
    class_label, date, state, zone, csv_id

Explorations réalisées (énoncé step 3) :
    1. Distribution des classes (class_label)
    2. Taux de valeurs manquantes par date et par bande
    3. Courbes NDVI temporelles par culture (Fig. 2 du papier)
    4. Courbes EVI temporelles par culture
    5. Profils temporels moyens par bande et par culture
    6. Boxplots spectraux par classe
    7. Heatmap de corrélation inter-bandes
    8. Patterns temporels : variance NDVI par date
=======================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
Donnees = "./Donnees"        
OUTPUT_DIR = "./exploration_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Correspondance class_label → nom de culture (Table 2 papier)
AR_CLASSES = {1: "Corn", 2: "Cotton", 3: "Rice", 4: "Soybeans", 5: "Others"}
CA_CLASSES = {1: "Grapes", 2: "Rice", 3: "Alfalfa",
              4: "Almonds", 5: "Pistachios", 6: "Others"}

# Palette cohérente avec Fig. 2 du papier
AR_COLORS = {1: "#2ca02c", 2: "#8c564b", 3: "#17becf",
             4: "#ff7f0e", 5: "#7f7f7f"}
CA_COLORS = {1: "#9467bd", 2: "#17becf", 3: "#e377c2",
             4: "#1f77b4", 5: "#bcbd22", 6: "#7f7f7f"}

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
def load_zone(state, zone):
    """Charge et concatène les CSV d'une zone → DataFrame unique."""
    
    folder = f"{state}-{zone}"   # 🔥 IMPORTANT
    pattern = os.path.join(Donnees, folder, "*.csv")

    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["doy"]  = df_all["date"].dt.dayofyear

    return df_all


def load_all():
    """Charge les 4 zones et retourne deux DataFrames : AR et CA."""
    zones = [
        ("Arkansas",   "zone1"),
        ("Arkansas",   "zone2"),
        ("California", "zone1"),
        ("California", "zone2"),
    ]
    dfs = {}
    for state, zone in zones:
        key = f"{state}_{zone}"
        try:
            dfs[key] = load_zone(state, zone)
            print(f"✓ {key} : {len(dfs[key]):,} lignes chargées")
        except FileNotFoundError as e:
            print(f"✗ {key} : {e}")
    # Fusionner zones d'un même état
    ar_dfs = [v for k, v in dfs.items() if "Arkansas" in k]
    ca_dfs = [v for k, v in dfs.items() if "California" in k]
    AR = pd.concat(ar_dfs, ignore_index=True) if ar_dfs else pd.DataFrame()
    CA = pd.concat(ca_dfs, ignore_index=True) if ca_dfs else pd.DataFrame()
    return AR, CA


# -----------------------------------------------------------------------
# 2. CALCUL DES INDICES DE VÉGÉTATION
# -----------------------------------------------------------------------
def add_vegetation_indices(df):
    """Ajoute NDVI, EVI, NDWI au DataFrame."""
    B4  = df["B4"].replace(0, np.nan)   # Red
    B8  = df["B8"].replace(0, np.nan)   # NIR
    B2  = df["B2"].replace(0, np.nan)   # Blue
    B11 = df["B11"].replace(0, np.nan)  # SWIR1

    # NDVI = (NIR - Red) / (NIR + Red)
    df["NDVI"] = (B8 - B4) / (B8 + B4 + 1e-8)

    # EVI  = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    df["EVI"]  = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + 1e-8)

    # NDWI = (Green - NIR) / (Green + NIR)
    B3 = df["B3"].replace(0, np.nan)
    df["NDWI"] = (B3 - B8) / (B3 + B8 + 1e-8)

    # Masque de données manquantes (valeur 0 = manquant selon papier)
    df["missing"] = (df[BANDS] == 0).all(axis=1).astype(int)

    return df


# -----------------------------------------------------------------------
# 3. EXPLORATION — DISTRIBUTION DES CLASSES
# -----------------------------------------------------------------------
def plot_class_distribution(AR, CA):
    """Barplot de la distribution des classes (Table 2 du papier)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Class Distribution (Table 2 — Wang et al., 2024)",
                 fontsize=13, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        # Compter par classe sur un seul timestamp (éviter le comptage ×36)
        df_t0  = df[df["csv_id"] == 1].copy()
        counts = df_t0["class_label"].value_counts().sort_index()
        labels = [class_map.get(i, f"Class {i}") for i in counts.index]
        colors = [color_map.get(i, "gray")        for i in counts.index]
        pcts   = counts.values / counts.values.sum() * 100

        bars = ax.bar(labels, counts.values, color=colors, edgecolor="white",
                      linewidth=0.8)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 20,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Crop type")
        ax.set_ylabel("Number of samples")
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_class_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 01_class_distribution.png")


# -----------------------------------------------------------------------
# 4. EXPLORATION — VALEURS MANQUANTES
# -----------------------------------------------------------------------
def plot_missing_values(AR, CA):
    """
    Taux de valeurs manquantes par date (un pixel est manquant si
    toutes ses bandes valent 0, convention MCTNet).
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Missing Data Rate per Date\n"
                 "(pixels where all bands = 0, per MCTNet convention)",
                 fontsize=12, fontweight="bold")

    for ax, df, title in [
        (axes[0], AR, "Arkansas"),
        (axes[1], CA, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        miss_rate = df.groupby("date")["missing"].mean() * 100
        ax.bar(miss_rate.index, miss_rate.values,
               color="#d62728", alpha=0.7, width=4)
        ax.axhline(miss_rate.mean(), color="black", linestyle="--",
                   linewidth=1, label=f"Mean: {miss_rate.mean():.1f}%")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Missing rate (%)")
        ax.set_xlabel("Date")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 100)

        # Annoter les mois
        for m in range(1, 13):
            ax.axvline(pd.Timestamp(f"2021-{m:02d}-01"),
                       color="gray", alpha=0.2, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_missing_values.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 02_missing_values.png")


def plot_missing_per_band(AR, CA):
    """Taux de valeurs manquantes par bande spectrale."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Missing Rate per Spectral Band", fontsize=12,
                 fontweight="bold")

    for ax, df, title in [
        (axes[0], AR, "Arkansas"),
        (axes[1], CA, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        miss_per_band = [(b, (df[b] == 0).mean() * 100) for b in BANDS]
        bands_labels  = [BAND_NAMES[b] for b, _ in miss_per_band]
        rates         = [r for _, r in miss_per_band]

        ax.barh(bands_labels, rates, color="#1f77b4", alpha=0.8)
        ax.set_xlabel("Missing rate (%)")
        ax.set_title(title, fontsize=11)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_missing_per_band.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 03_missing_per_band.png")


# -----------------------------------------------------------------------
# 5. EXPLORATION — COURBES NDVI TEMPORELLES (Fig. 2 du papier)
# -----------------------------------------------------------------------
def plot_ndvi_timeseries(AR, CA):
    """
    Reproduit la Fig. 2 du papier : NDVI moyen par culture en fonction
    du Day of Year (DOY).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Mean NDVI Time-Series Profiles per Crop Type\n"
                 "(Reproduction of Fig. 2 — Wang et al., 2024)",
                 fontsize=12, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "(a) Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "(b) California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        # Exclure pixels manquants pour le calcul NDVI
        df_valid = df[df["missing"] == 0].copy()

        for cls, name in class_map.items():
            df_cls  = df_valid[df_valid["class_label"] == cls]
            if df_cls.empty:
                continue
            ndvi_by_doy = df_cls.groupby("doy")["NDVI"].mean()
            ax.plot(ndvi_by_doy.index, ndvi_by_doy.values,
                    label=name, color=color_map[cls],
                    linewidth=2, marker="o", markersize=4)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Mean NDVI Value")
        ax.set_xlim(0, 365)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

        # Marquer les mois
        month_doys = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(month_doys)
        ax2.set_xticklabels(month_names, fontsize=7)
        ax2.tick_params(length=0)
        ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_ndvi_timeseries.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 04_ndvi_timeseries.png")


# -----------------------------------------------------------------------
# 6. EXPLORATION — COURBES EVI TEMPORELLES
# -----------------------------------------------------------------------
def plot_evi_timeseries(AR, CA):
    """EVI moyen par culture — complément à NDVI."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Mean EVI Time-Series Profiles per Crop Type",
                 fontsize=12, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "(a) Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "(b) California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        df_valid = df[df["missing"] == 0].copy()

        for cls, name in class_map.items():
            df_cls = df_valid[df_valid["class_label"] == cls]
            if df_cls.empty:
                continue
            evi_by_doy = df_cls.groupby("doy")["EVI"].mean()
            ax.plot(evi_by_doy.index, evi_by_doy.values,
                    label=name, color=color_map[cls],
                    linewidth=2, marker="s", markersize=4)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Mean EVI Value")
        ax.set_xlim(0, 365)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_evi_timeseries.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 05_evi_timeseries.png")


# -----------------------------------------------------------------------
# 7. EXPLORATION — PROFILS SPECTRAUX PAR CLASSE (un instant t)
# -----------------------------------------------------------------------
def plot_spectral_profiles(AR, CA, doy_target=180):
    """
    Profil spectral moyen par classe à un instant donné (DOY ~180
    = pic de végétation en été).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Mean Spectral Profiles at DOY ≈ {doy_target} (peak season)",
                 fontsize=12, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        # Sélectionner la date la plus proche du DOY cible
        doys_avail = df["doy"].unique()
        doy_sel    = doys_avail[np.argmin(np.abs(doys_avail - doy_target))]
        df_t       = df[(df["doy"] == doy_sel) & (df["missing"] == 0)]

        band_labels = [BAND_NAMES[b] for b in BANDS]

        for cls, name in class_map.items():
            df_cls = df_t[df_t["class_label"] == cls]
            if df_cls.empty:
                continue
            means = df_cls[BANDS].mean().values
            ax.plot(band_labels, means,
                    label=name, color=color_map[cls],
                    linewidth=2, marker="o", markersize=5)

        ax.set_title(f"{title} — DOY {doy_sel}", fontsize=11)
        ax.set_xlabel("Spectral Band")
        ax.set_ylabel("Mean Reflectance")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_spectral_profiles.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 06_spectral_profiles.png")


# -----------------------------------------------------------------------
# 8. EXPLORATION — BOXPLOTS NDVI PAR CLASSE
# -----------------------------------------------------------------------
def plot_ndvi_boxplots(AR, CA):
    """Distribution de NDVI par classe (boxplot)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("NDVI Distribution per Crop Class", fontsize=12,
                 fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        df_valid = df[df["missing"] == 0].copy()
        df_valid["crop_name"] = df_valid["class_label"].map(class_map)

        order  = [class_map[k] for k in sorted(class_map.keys())]
        palette= {class_map[k]: color_map[k] for k in class_map}

        sns.boxplot(data=df_valid, x="crop_name", y="NDVI",
                    order=order, palette=palette,
                    flierprops={"marker": ".", "alpha": 0.3,
                                "markersize": 2},
                    ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Crop type")
        ax.set_ylabel("NDVI")
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_ndvi_boxplots.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 07_ndvi_boxplots.png")


# -----------------------------------------------------------------------
# 9. EXPLORATION — HEATMAP DE CORRÉLATION INTER-BANDES
# -----------------------------------------------------------------------
def plot_band_correlation(AR, CA):
    """Corrélation de Pearson entre les 10 bandes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Spectral Band Correlation Matrix", fontsize=12,
                 fontweight="bold")

    for ax, df, title in [
        (axes[0], AR, "Arkansas"),
        (axes[1], CA, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        df_valid = df[(df["missing"] == 0)].sample(
            min(50000, len(df)), random_state=42)
        corr = df_valid[BANDS].corr()

        band_labels = [BAND_NAMES[b] for b in BANDS]
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        sns.heatmap(corr,
                    annot=True, fmt=".2f", cmap="RdYlGn",
                    xticklabels=band_labels, yticklabels=band_labels,
                    vmin=-1, vmax=1, center=0,
                    linewidths=0.5, ax=ax,
                    annot_kws={"size": 8})
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_band_correlation.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 08_band_correlation.png")


# -----------------------------------------------------------------------
# 10. EXPLORATION — VARIANCE TEMPORELLE DU NDVI
# -----------------------------------------------------------------------
def plot_ndvi_variance(AR, CA):
    """
    Variance du NDVI par date → montre quelles périodes sont les plus
    discriminantes entre cultures.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("NDVI Temporal Variance by Crop Type\n"
                 "(high variance = discriminative period)",
                 fontsize=12, fontweight="bold")

    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California"),
    ]:
        if df.empty:
            ax.set_title(f"{title} — No data"); continue

        df_valid = df[df["missing"] == 0].copy()

        for cls, name in class_map.items():
            df_cls = df_valid[df_valid["class_label"] == cls]
            if df_cls.empty:
                continue
            var_by_doy = df_cls.groupby("doy")["NDVI"].std()
            ax.plot(var_by_doy.index, var_by_doy.values,
                    label=name, color=color_map[cls],
                    linewidth=1.5, alpha=0.8)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("NDVI Std. Deviation")
        ax.set_xlim(0, 365)
        ax.legend(fontsize=8, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_ndvi_variance.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 09_ndvi_variance.png")


# -----------------------------------------------------------------------
# 11. EXPLORATION — RÉSUMÉ STATISTIQUE
# -----------------------------------------------------------------------
def print_summary(AR, CA):
    """Statistiques descriptives générales."""
    print("\n" + "="*60)
    print("SUMMARY — ARKANSAS")
    print("="*60)
    if not AR.empty:
        df_t0 = AR[AR["csv_id"] == 1]
        print(f"Total pixels (unique) : {len(df_t0):,}")
        print(f"Missing pixels (all bands=0) : "
              f"{df_t0['missing'].sum():,} "
              f"({df_t0['missing'].mean()*100:.1f}%)")
        print("\nClass distribution :")
        counts = df_t0["class_label"].value_counts().sort_index()
        for cls, cnt in counts.items():
            name = AR_CLASSES.get(cls, f"Class {cls}")
            print(f"  {name:10s} : {cnt:5,} ({cnt/len(df_t0)*100:.1f}%)")
        print(f"\nNDVI stats (valid pixels) :")
        ndvi = AR[AR["missing"] == 0]["NDVI"]
        print(f"  Mean={ndvi.mean():.3f}, Std={ndvi.std():.3f}, "
              f"Min={ndvi.min():.3f}, Max={ndvi.max():.3f}")

    print("\n" + "="*60)
    print("SUMMARY — CALIFORNIA")
    print("="*60)
    if not CA.empty:
        df_t0 = CA[CA["csv_id"] == 1]
        print(f"Total pixels (unique) : {len(df_t0):,}")
        print(f"Missing pixels (all bands=0) : "
              f"{df_t0['missing'].sum():,} "
              f"({df_t0['missing'].mean()*100:.1f}%)")
        print("\nClass distribution :")
        counts = df_t0["class_label"].value_counts().sort_index()
        for cls, cnt in counts.items():
            name = CA_CLASSES.get(cls, f"Class {cls}")
            print(f"  {name:10s} : {cnt:5,} ({cnt/len(df_t0)*100:.1f}%)")
        print(f"\nNDVI stats (valid pixels) :")
        ndvi = CA[CA["missing"] == 0]["NDVI"]
        print(f"  Mean={ndvi.mean():.3f}, Std={ndvi.std():.3f}, "
              f"Min={ndvi.min():.3f}, Max={ndvi.max():.3f}")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("STEP 3 — DATA EXPLORATION")
    print("MCTNet Project — M1 SII USTHB 2025/2026")
    print("=" * 60)

    # 1. Chargement
    print("\n[1/8] Loading data...")
    AR, CA = load_all()

    # 2. Calcul des indices
    print("[2/8] Computing vegetation indices (NDVI, EVI, NDWI)...")
    if not AR.empty:
        AR = add_vegetation_indices(AR)
    if not CA.empty:
        CA = add_vegetation_indices(CA)

    # 3. Résumé statistique
    print("[3/8] Summary statistics...")
    print_summary(AR, CA)

    # 4. Distribution des classes
    print("\n[4/8] Class distribution plots...")
    plot_class_distribution(AR, CA)

    # 5. Valeurs manquantes
    print("[5/8] Missing values analysis...")
    plot_missing_values(AR, CA)
    plot_missing_per_band(AR, CA)

    # 6. NDVI temporel (Fig. 2 du papier)
    print("[6/8] NDVI time-series profiles (reproduction of Fig. 2)...")
    plot_ndvi_timeseries(AR, CA)

    # 7. EVI temporel
    print("[7/8] EVI time-series profiles...")
    plot_evi_timeseries(AR, CA)

    # 8. Profils spectraux
    print("[8/8] Spectral profiles, boxplots, correlations...")
    plot_spectral_profiles(AR, CA, doy_target=180)
    plot_ndvi_boxplots(AR, CA)
    plot_band_correlation(AR, CA)
    plot_ndvi_variance(AR, CA)

    print("\n" + "=" * 60)
    print(f"✅ All exploration plots saved in: {OUTPUT_DIR}/")
    print("=" * 60)