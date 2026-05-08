import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =======================================================================
# PART 1 — STEP 4 : COVARIATES ANALYSIS (Arkansas)
# =======================================================================

# Configuration
DATA_DIR = "Donnees_Merged/MCTNet_arkansas"
OUTPUT_FILE = "AR_covariables_ranges.png"

# Arkansas specific mapping
CLASSES    = ['Corn', 'Cotton', 'Rice', 'Soybeans', 'Others']
COLORS_CLS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Variables we want to show
VARIABLES = ['ph', 'clay', 'org_carbon', 'elevation', 'landforms', 'temp', 'precip', 'solar_rad']
COL_LABELS = ['Classe', 'pH', 'Clay', 'Org Carbon', 'Elevation', 'Landforms', 'Temp (avg)', 'Prec (avg)', 'Solar Rad']

def main():
    print(f"-> Starting analysis for Arkansas covariates...")
    
    pattern = os.path.join(DATA_DIR, "arkansas_merged_t*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Error: No files found in {DATA_DIR}")
        return

    # 1. Load static data from the first file
    print(f"-> Loading static data from {os.path.basename(files[0])}...")
    # Note: In some files it might be 'Soybeans' or 'Soybean', we check label_name
    static_cols = ['pixel_id', 'label_name', 'ph', 'clay', 'org_carbon', 'elevation', 'landforms']
    df_static = pd.read_csv(files[0], usecols=static_cols)
    
    # Handle possible naming variations (e.g., Soybean vs Soybeans)
    df_static['label_name'] = df_static['label_name'].replace('Soybean', 'Soybeans')

    # 2. Accumulate climate data across all 36 timesteps
    print(f"-> Accumulating climate data across {len(files)} timesteps...")
    climate_sum = None
    climate_cols = ['pixel_id', 'temp', 'precip', 'solar_rad']
    
    for i, f in enumerate(files):
        df_c = pd.read_csv(f, usecols=climate_cols)
        df_c = df_c.set_index('pixel_id')
        
        if climate_sum is None:
            climate_sum = df_c
        else:
            climate_sum = climate_sum.add(df_c, fill_value=0)
            
    # Calculate mean
    climate_mean = climate_sum / len(files)
    climate_mean = climate_mean.reset_index()

    # 3. Merge static and climate
    print(f"-> Merging data...")
    df = df_static.merge(climate_mean, on='pixel_id')

    # 4. Generate rows for the table
    print(f"-> Calculating min-max ranges per class...")
    rows = []
    for cls in CLASSES:
        sub = df[df['label_name'] == cls]
        row = [cls]
        if sub.empty:
            print(f"Warning: No data for class {cls}")
            row += ["N/A"] * len(VARIABLES)
        else:
            for v in VARIABLES:
                mn = sub[v].min()
                mx = sub[v].max()
                row.append(f'{mn:.2f}–{mx:.2f}')
        rows.append(row)

    # 5. Plotting
    print(f"-> Creating modern table plot...")
    
    # Modern Color Palette
    BG_COLOR = '#FFFFFF'
    HEADER_BG = '#263238'  # Dark Slate
    HEADER_TEXT = '#FFFFFF'
    ROW_BG_ALT = '#F5F7F8'
    BORDER_COLOR = '#DFE6E9'
    TEXT_COLOR = '#2D3436'

    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(24, 6), facecolor=BG_COLOR)
    ax.axis('off')
    
    # Add a title and subtitle
    fig.text(0.5, 0.92, 'Arkansas Crop Covariate Analysis', 
             ha='center', fontsize=22, fontweight='bold', color=HEADER_BG)
    fig.text(0.5, 0.86, 'Min–Max Value Ranges Across All Environmental Variables (36-Timestep Average)', 
             ha='center', fontsize=12, color='#636E72', style='italic')

    col_widths = [0.08] + [0.115] * len(VARIABLES)
    table = ax.table(
        cellText=rows,
        colLabels=COL_LABELS,
        loc='center',
        cellLoc='center',
        colWidths=col_widths,
        edges='closed'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3.2)

    # Styling header
    for j in range(len(COL_LABELS)):
        cell = table[0, j]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_TEXT, fontweight='bold', fontsize=11)
        cell.set_edgecolor(HEADER_BG)

    # Styling rows
    for i, color in enumerate(COLORS_CLS):
        label_cell = table[i+1, 0]
        label_cell.set_facecolor(color)
        label_cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        label_cell.set_edgecolor(color)
        
        for j in range(1, len(COL_LABELS)):
            cell = table[i+1, j]
            bg = ROW_BG_ALT if i % 2 != 0 else BG_COLOR
            cell.set_facecolor(bg)
            cell.set_text_props(color=TEXT_COLOR)
            cell.set_edgecolor(BORDER_COLOR)
            cell.set_linewidth(0.5)

    # Add a source footer
    fig.text(0.05, 0.05, 'Source: MSF-MCTNet Arkansas Dataset (Donnees_Merged)', 
             fontsize=9, color='#B2BEC3')
    
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    print(f"SUCCESS: Premium table saved as {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
