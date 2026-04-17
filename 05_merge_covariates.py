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

def merge_state_data(state):
    folder = f"MCTNet_{state.lower()}"
    s2_path = os.path.join(DATA_DIR, folder)
    
    # Check for S2 files
    s2_files = sorted(glob.glob(os.path.join(s2_path, "*.csv")))
    if not s2_files:
        print(f"  ⚠️ No Sentinel-2 data found for {state} in {s2_path}")
        return
    
    # Check for Covariate files (Expected in the same folder or root)
    static_file = os.path.join(DATA_DIR, folder, f"covariates_static_{state.lower()}.csv")
    climate_file = os.path.join(DATA_DIR, folder, f"covariates_climate_{state.lower()}.csv")
    
    # 1. Load S2 Data
    print(f"  Processing {state}...")
    dfs = [pd.read_csv(f) for f in s2_files if "covariate" not in f]
    df_s2 = pd.concat(dfs, ignore_index=True)

    # 2. Check and Load Covariates
    has_static = os.path.exists(static_file)
    has_climate = os.path.exists(climate_file)
    
    if not has_static and not has_climate:
        print(f"    ⚠️ No covariate files found for {state}. Skipping merge.")
        return

    df_merged = df_s2
    if has_static:
        print(f"    -> Adding static covariates...")
        df_static = pd.read_csv(static_file)
        df_merged['pixel_id'] = df_merged['pixel_id'].astype(str)
        df_static['pixel_id'] = df_static['pixel_id'].astype(str)
        # Select only necessary columns from static
        valid_static = [c for c in STATIC_COLS if c in df_static.columns]
        df_static = df_static[['pixel_id'] + valid_static]
        df_merged = pd.merge(df_merged, df_static, on='pixel_id', how='left')
    else:
        print(f"    ⚠️ Static file missing for {state}. Skipping static columns.")

    # 4. Merge Climate Covariates if available
    if has_climate:
        print(f"    -> Adding climate covariates...")
        df_climate = pd.read_csv(climate_file)
        # month = floor(timestep / 3) + 1
        df_merged['month'] = (df_merged['timestep'] // 3) + 1
        df_climate['pixel_id'] = df_climate['pixel_id'].astype(str)
        df_climate['month'] = df_climate['month'].astype(int)
        valid_climate = [c for c in CLIMATE_COLS if c in df_climate.columns]
        df_climate = df_climate[['pixel_id', 'month'] + valid_climate]
        df_merged = pd.merge(df_merged, df_climate, on=['pixel_id', 'month'], how='left')
    else:
        print(f"    ⚠️ Climate file missing for {state}. Skipping climate columns.")
    
    # 5. Save Merged Data
    state_out_dir = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(state_out_dir, exist_ok=True)
    
    # We save in chunks to keep files small, similar to original
    for timestep in range(36):
        df_t = df_merged[df_merged['timestep'] == timestep]
        # Preserve filename pattern if possible
        filename = f"{state.lower()}_merged_t{timestep:02d}.csv"
        df_t.to_csv(os.path.join(state_out_dir, filename), index=False)
        
    print(f"  ✅ {state}: Merged data saved to {state_out_dir}")

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: MERGING COVARIATES")
    print("=" * 60)
    
    for state in ["Arkansas", "California"]:
        merge_state_data(state)
