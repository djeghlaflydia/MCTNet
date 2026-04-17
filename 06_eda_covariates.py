"""
=======================================================================
PART 2 — STEP 3 : EXPLORATORY DATA ANALYSIS (COVARIATES)
Project: Deep Learning for Crop Classification (USTHB)

This script analyzes the distribution of environmental covariates
and their relationship with crop classes.
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
DATA_DIR    = "./Donnees_Merged"
OUTPUT_DIR  = "./exploration_outputs/covariates"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classes mapping
AR_CLASSES = {1: "Corn", 2: "Cotton", 3: "Rice", 4: "Soybean", 0: "Others"}
CA_CLASSES = {1: "Grapes", 3: "Alfalfa", 2: "Rice", 4: "Almonds", 5: "Pistachios", 0: "Others"}

STATIC_COLS = ["elevation", "slope", "aspect", "clay", "sand", "org_carbon", "ph"]
CLIMATE_COLS = ["temp", "precip"]

def load_merged_state(state):
    folder = f"MCTNet_{state.lower()}"
    pattern = os.path.join(DATA_DIR, folder, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  ⚠ No merged data found for {state}. Check Step 2.")
        return pd.DataFrame()
    
    # We only need one row per pixel for static analysis
    # and maybe means for climate analysis
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df

def plot_static_distributions(df, state, class_map):
    if df.empty: return
    
    # Get unique pixels
    df_static = df.groupby('pixel_id').first().reset_index()
    
    fig, axes = plt.subplots(len(STATIC_COLS), 1, figsize=(10, 3 * len(STATIC_COLS)))
    if len(STATIC_COLS) == 1: axes = [axes]
    
    for ax, col in zip(axes, STATIC_COLS):
        sns.boxplot(x='label', y=col, data=df_static, ax=ax, palette="Set2")
        ax.set_title(f"Distribution of {col} by Crop Class ({state})")
        ax.set_xticklabels([class_map.get(int(x.get_text()), x.get_text()) for x in ax.get_xticklabels()])
        ax.set_xlabel("Crop Class")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{state}_static_distributions.png"), dpi=200)
    print(f"  ✓ Saved: {state}_static_distributions.png")

def plot_correlation_matrix(df, state):
    if df.empty: return
    
    # Select bands + covariates
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    cols = bands + STATIC_COLS + CLIMATE_COLS + ["label"]
    
    corr = df[cols].corr()
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Matrix: Features vs Label ({state})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{state}_correlation_matrix.png"), dpi=200)
    print(f"  ✓ Saved: {state}_correlation_matrix.png")

def plot_climate_trends(df, state, class_map):
    if df.empty: return
    
    # Monthly averages for climate
    df_monthly = df.groupby(['month', 'label'])[CLIMATE_COLS].mean().reset_index()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    for ax, col in zip(axes, CLIMATE_COLS):
        for label in df_monthly['label'].unique():
            df_cls = df_monthly[df_monthly['label'] == label]
            ax.plot(df_cls['month'], df_cls[col], label=class_map.get(label, f"Cls {label}"), marker='o')
        
        ax.set_title(f"Monthly Average {col} ({state})")
        ax.set_xlabel("Month")
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{state}_climate_trends.png"), dpi=200)
    print(f"  ✓ Saved: {state}_climate_trends.png")

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (COVARIATES)")
    print("=" * 60)
    
    for state, class_map in [("Arkansas", AR_CLASSES), ("California", CA_CLASSES)]:
        print(f"\nAnalyzing {state}...")
        df = load_merged_state(state)
        if not df.empty:
            plot_static_distributions(df, state, class_map)
            plot_correlation_matrix(df, state)
            plot_climate_trends(df, state, class_map)

    print(f"\n✅ EDA Complete. Results saved in {OUTPUT_DIR}")
