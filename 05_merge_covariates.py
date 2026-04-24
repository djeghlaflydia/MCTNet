"""
=======================================================================
PART 2 — STEP 2 : MERGE COVARIATES
Project: Deep Learning for Crop Classification (USTHB)

This script merges the baseline Sentinel-2 data with environmental
covariates (Climate, Soil, Topography) exported from GEE.
=======================================================================
"""

import os
import glob
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
DATA_DIR    = "./Donnees"
OUTPUT_DIR  = "./Donnees_Merged"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIC_COLS = ["elevation", "slope", "aspect", "clay", "sand", "org_carbon", "ph"]
CLIMATE_COLS = ["temp", "precip"]

def load_partitioned_csv(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    print(f"    -> Loading {len(files)} partitions for {os.path.basename(pattern)}...")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def merge_state_data(state):
    folder = f"MCTNet_{state.lower()}"
    s2_path = os.path.join(DATA_DIR, folder)
    
    # 1. Load S2 Data
    s2_files = sorted(glob.glob(os.path.join(s2_path, "*.csv")))
    s2_files = [f for f in s2_files if "covariate" not in f.lower()]
    if not s2_files:
        print(f"  ⚠️ No Sentinel-2 data found for {state} in {s2_path}")
        return
    
    print(f"  Processing {state}...")
    df_s2 = pd.concat([pd.read_csv(f) for f in s2_files], ignore_index=True)
    df_s2['pixel_id'] = df_s2['pixel_id'].astype(str)

    # 2. Load Static Covariates (Partitioned or Single)
    static_pattern = os.path.join(DATA_DIR, folder, f"*static_{state.lower()}*.csv")
    df_static = load_partitioned_csv(static_pattern)
    
    # 3. Load Climate Covariates (Partitioned or Single)
    climate_pattern = os.path.join(DATA_DIR, folder, f"*climate_{state.lower()}*.csv")
    df_climate = load_partitioned_csv(climate_pattern)
    
    df_merged = df_s2

    # Merge Static
    if df_static is not None:
        print(f"    -> Merging static covariates...")
        df_static['pixel_id'] = df_static['pixel_id'].astype(str)
        # Select valid columns
        valid_static = [c for c in STATIC_COLS if c in df_static.columns]
        df_static = df_static[['pixel_id'] + valid_static].drop_duplicates(subset='pixel_id')
        df_merged = pd.merge(df_merged, df_static, on='pixel_id', how='left')
    else:
        print(f"    ⚠️ No static covariates found for {state}.")

    # Merge Climate
    if df_climate is not None:
        print(f"    -> Merging climate covariates...")
        # month = floor(timestep / 3) + 1 (Approximate month mapping)
        df_merged['month'] = (df_merged['timestep'] // 3) + 1
        df_climate['pixel_id'] = df_climate['pixel_id'].astype(str)
        df_climate['month'] = df_climate['month'].astype(int)
        
        valid_climate = [c for c in CLIMATE_COLS if c in df_climate.columns]
        df_climate = df_climate[['pixel_id', 'month'] + valid_climate].drop_duplicates(subset=['pixel_id', 'month'])
        df_merged = pd.merge(df_merged, df_climate, on=['pixel_id', 'month'], how='left')
    else:
        print(f"    ⚠️ No climate covariates found for {state}.")
    
    # 5. Save Merged Data
    state_out_dir = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(state_out_dir, exist_ok=True)
    
    print(f"    -> Saving merged files to {state_out_dir}...")
    for timestep in range(36):
        df_t = df_merged[df_merged['timestep'] == timestep]
        filename = f"{state.lower()}_merged_t{timestep:02d}.csv"
        df_t.to_csv(os.path.join(state_out_dir, filename), index=False)
        
    print(f"  [OK] {state} complete.")

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: MERGING COVARIATES")
    print("=" * 60)
    
    for state in ["Arkansas", "California"]:
        merge_state_data(state)
