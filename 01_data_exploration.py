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
    folder = f"MCTNet_{state.lower()}"
    pattern = os.path.join(DONNEES_DIR, folder, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files: return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date_str"])
    df_all["doy"]  = df_all["date"].dt.dayofyear
    df_all["missing"] = ((df_all[BANDS] == 0).all(axis=1) | (df_all["valid"] == 0)).astype(int)
    return df_all

def add_vegetation_indices(df):
    if df.empty: return df
    B4, B8, B2, B3 = df["B4"].replace(0, np.nan), df["B8"].replace(0, np.nan), df["B2"].replace(0, np.nan), df["B3"].replace(0, np.nan)
    df["NDVI"] = (B8 - B4) / (B8 + B4 + 1e-8)
    df["EVI"]  = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + 1e-8)
    return df

def plot_ndvi_timeseries(AR, CA):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, df, class_map, color_map, title in [(axes[0], AR, AR_CLASSES, AR_COLORS, "(a) Arkansas"), (axes[1], CA, CA_CLASSES, CA_COLORS, "(b) California")]:
        if df.empty: continue
        df_valid = df[df["missing"] == 0].copy()
        legend_order = [1, 2, 3, 4, 0] if "Arkansas" in title else [1, 3, 2, 4, 5, 0]
        for cls in legend_order:
            if cls not in class_map: continue
            df_cls = df_valid[df_valid["label"] == cls]
            if df_cls.empty: continue
            ndvi_by_doy = df_cls.groupby("doy")["NDVI"].mean()
            ax.plot(ndvi_by_doy.index, ndvi_by_doy.values, label=class_map[cls], color=color_map[cls], linewidth=1.5, marker="o", markersize=4)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Mean NDVI Value", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 370)
        ax.set_xticks([10, 60, 110, 160, 210, 260, 310, 360])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(fontsize=9, loc="upper left", ncol=2)
        ax.grid(True, linestyle='-', alpha=0.7)
        ax.set_xlabel("Day of Year", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_ndvi_timeseries.png"), dpi=200)
    print("✓ Saved: 04_ndvi_timeseries.png (Paper Style)")

if __name__ == "__main__":
    AR, CA = load_state_data("Arkansas"), load_state_data("California")
    AR, CA = add_vegetation_indices(AR), add_vegetation_indices(CA)
    plot_ndvi_timeseries(AR, CA)
    print(f"✅ Exploration complete. Plots saved in {OUTPUT_DIR}/")