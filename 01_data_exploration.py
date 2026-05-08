"""
=======================================================================
PART 1 — STEP 3 : DATA EXPLORATION (V3 — Paper Exact Style)
Project: Deep Learning for Crop Classification Using Multi-Source
         Satellite Data (M1 SII 2025/2026 — USTHB)
Paper  : MCTNet — Wang et al., 2024
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

# Correspondance label → nom de culture (Mapping exact du papier)
AR_CLASSES = {1: "Corn", 2: "Cotton", 3: "Rice", 4: "Soybean", 0: "Others"}
CA_CLASSES = {1: "Grapes", 3: "Alfalfa", 2: "Rice", 4: "Almonds", 5: "Pistachios", 0: "Others"}

# Palette EXACTE du papier (Matplotlib tab10)
AR_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728", 0: "#9467bd"}
CA_COLORS = {1: "#1f77b4", 3: "#ff7f0e", 2: "#2ca02c", 4: "#d62728", 5: "#9467bd", 0: "#8c564b"}

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_NAMES = {
    "B2": "Blue", "B3": "Green", "B4": "Red",
    "B5": "RE 1", "B6": "RE 2", "B7": "RE 3",
    "B8": "NIR", "B8A": "RE 4",
    "B11": "SWIR 1", "B12": "SWIR 2"
}

def load_state_data(state):
    print(f"-> Loading data for {state}...")
    folder = f"MCTNet_{state.lower()}"
    pattern = os.path.join(DONNEES_DIR, folder, "*.csv")
    files = sorted(glob.glob(pattern))
    # Filter out covariate files if they are in the same folder
    files = [f for f in files if "covariates" not in f]
    
    if not files: 
        print(f"Warning: No files found for {state}")
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        df_tmp = pd.read_csv(f)
        dfs.append(df_tmp)
    
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date_str"])
    df_all["doy"]  = df_all["date"].dt.dayofyear
    
    # Missing data criteria: either all bands are zero OR valid flag is 0
    df_all["missing"] = ((df_all[BANDS] == 0).all(axis=1) | (df_all["valid"] == 0)).astype(int)
    
    return df_all

def add_vegetation_indices(df):
    if df.empty: return df
    # Use small epsilon to avoid division by zero
    eps = 1e-8
    B2 = df["B2"].astype(float) / 10000.0
    B3 = df["B3"].astype(float) / 10000.0
    B4 = df["B4"].astype(float) / 10000.0
    B8 = df["B8"].astype(float) / 10000.0
    
    df["NDVI"] = (B8 - B4) / (B8 + B4 + eps)
    df["EVI"]  = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1.0 + eps)
    
    # Clip values to reasonable ranges
    df["NDVI"] = df["NDVI"].clip(-1, 1)
    df["EVI"] = df["EVI"].clip(-1, 2)
    
    return df

def plot_class_distribution(AR, CA):
    print("-> Plotting Class Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas Class Distribution"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California Class Distribution")
    ]:
        if df.empty: continue
        
        # Count unique pixel_ids per label
        dist = df.groupby("label")["pixel_id"].nunique().reset_index()
        dist["class_name"] = dist["label"].map(class_map)
        
        # Sort by label order for consistency
        dist = dist.sort_values("label")
        
        colors = [color_map.get(l, "#7f7f7f") for l in dist["label"]]
        
        sns.barplot(data=dist, x="class_name", y="pixel_id", palette=colors, ax=ax)
        
        # Add labels on top of bars
        for i, v in enumerate(dist["pixel_id"]):
            ax.text(i, v + (v * 0.02), f"{int(v)}", ha='center', fontweight='bold')
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Crop Type", fontsize=12)
        ax.set_ylabel("Number of Pixels", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_class_distribution.png"), dpi=200)
    plt.close()

def plot_missing_data_analysis(AR, CA):
    print("-> Plotting Missing Data Analysis...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    for ax, df, title in [
        (axes[0], AR, "Arkansas Missing Data Trend"),
        (axes[1], CA, "California Missing Data Trend")
    ]:
        if df.empty: continue
        
        # Calculate percentage of missing data per DOY
        missing_stats = df.groupby("doy")["missing"].mean() * 100
        
        ax.plot(missing_stats.index, missing_stats.values, color='red', marker='o', markersize=3, linewidth=1.5)
        ax.fill_between(missing_stats.index, 0, missing_stats.values, color='red', alpha=0.1)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel("% Missing Data", fontsize=11)
        ax.set_xlabel("Day of Year (DOY)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_missing_data_analysis.png"), dpi=200)
    plt.close()

def plot_spectral_variability(AR, CA):
    print("-> Plotting Spectral Variability...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, "Arkansas Spectral Profiles"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, "California Spectral Profiles")
    ]:
        if df.empty: continue
        
        # Use only valid observations
        df_valid = df[df["missing"] == 0]
        
        for label, name in class_map.items():
            df_cls = df_valid[df_valid["label"] == label]
            if df_cls.empty: continue
            
            # Mean spectral values across all timesteps
            spectral_mean = df_cls[BANDS].mean() / 10000.0
            
            ax.plot(BANDS, spectral_mean.values, label=name, color=color_map[label], marker='s', linewidth=2)
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Reflectance", fontsize=12)
        ax.set_xlabel("Sentinel-2 Bands", fontsize=12)
        ax.set_ylim(0, 0.6)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_spectral_variability.png"), dpi=200)
    plt.close()

def plot_indices_timeseries(AR, CA, index_name="NDVI"):
    print(f"-> Plotting {index_name} Timeseries...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for ax, df, class_map, color_map, title in [
        (axes[0], AR, AR_CLASSES, AR_COLORS, f"(a) Arkansas - {index_name} Evolution"),
        (axes[1], CA, CA_CLASSES, CA_COLORS, f"(b) California - {index_name} Evolution")
    ]:
        if df.empty: continue
        
        df_valid = df[df["missing"] == 0]
        legend_order = [1, 2, 3, 4, 0] if "Arkansas" in title else [1, 3, 2, 4, 5, 0]
        
        for cls in legend_order:
            if cls not in class_map: continue
            df_cls = df_valid[df_valid["label"] == cls]
            if df_cls.empty: continue
            
            # Group by DOY and calculate mean and std
            stats = df_cls.groupby("doy")[index_name].agg(['mean', 'std'])
            
            ax.plot(stats.index, stats['mean'], label=class_map[cls], color=color_map[cls], linewidth=2, marker="o", markersize=4)
            # Add subtle error bands
            ax.fill_between(stats.index, stats['mean'] - 0.5*stats['std'], stats['mean'] + 0.5*stats['std'], color=color_map[cls], alpha=0.1)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(f"Mean {index_name}", fontsize=11)
        ax.set_ylim(-0.1, 1.0 if index_name == "NDVI" else 1.2)
        ax.set_xlim(0, 370)
        ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'], rotation=45)
        ax.legend(fontsize=9, loc="upper left", ncol=2)
        ax.grid(True, linestyle='-', alpha=0.4)

    plt.tight_layout()
    filename = f"04_{index_name.lower()}_timeseries.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()

if __name__ == "__main__":
    # 1. Load data
    AR = load_state_data("Arkansas")
    CA = load_state_data("California")
    
    # 2. Precompute Indices
    AR = add_vegetation_indices(AR)
    CA = add_vegetation_indices(CA)
    
    # 3. Perform Analysis
    plot_class_distribution(AR, CA)
    plot_missing_data_analysis(AR, CA)
    plot_spectral_variability(AR, CA)
    plot_indices_timeseries(AR, CA, index_name="NDVI")
    plot_indices_timeseries(AR, CA, index_name="EVI")
    
    print(f"\n[SUCCESS] Exploration complete. All plots saved in: {os.path.abspath(OUTPUT_DIR)}")