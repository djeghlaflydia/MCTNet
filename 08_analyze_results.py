"""
=======================================================================
PART 2 — STEP 6 : ABLATION RESULTS ANALYSIS
Project: Deep Learning for Crop Classification (USTHB)

This script visualizes the results of the ablation study, comparing
the model performance across different covariate configurations.
=======================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
RESULTS_FILE = "./results/ablation/ablation_results.csv"
OUTPUT_DIR   = "./results/ablation/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_ablation_comparison(df):
    if df.empty:
        print("  ⚠️ Results file is empty. Run 07_ablation_study.py first.")
        return

    # Use a premium style
    sns.set_theme(style="whitegrid", palette="viridis")
    
    # 1. Comparison of Macro F1-Score
    plt.figure(figsize=(12, 7))
    sns.barplot(x="config", y="macro_f1", hue="state", data=df)
    
    plt.title("MCTNet Ablation Study: Macro F1-Score Comparison", fontsize=15, fontweight='bold')
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.xlabel("Configuration", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Area", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add values on top of bars
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.gca().annotate(f'{height:.3f}', 
                           (p.get_x() + p.get_width() / 2., height), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 9), 
                           textcoords = 'offset points',
                           fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_f1_comparison.png"), dpi=200)
    print("  ✓ Saved: ablation_f1_comparison.png")

    # 2. Comparison of Accuracy
    plt.figure(figsize=(12, 7))
    sns.barplot(x="config", y="test_accuracy", hue="state", data=df, palette="magma")
    
    plt.title("MCTNet Ablation Study: Test Accuracy Comparison", fontsize=15, fontweight='bold')
    plt.ylabel("Overall Accuracy", fontsize=12)
    plt.xlabel("Configuration", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Area", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_accuracy_comparison.png"), dpi=200)
    print("  ✓ Saved: ablation_accuracy_comparison.png")

def print_summary_table(df):
    if df.empty: return
    
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    # Pivot for clean comparison
    summary = df.pivot(index="config", columns="state", values=["macro_f1", "test_accuracy"]).round(4)
    print(summary)
    
    # Identify Best
    for state in df['state'].unique():
        state_df = df[df['state'] == state]
        best_cfg = state_df.loc[state_df['macro_f1'].idxmax()]
        print(f"\n🏆 Best Configuration for {state}: '{best_cfg['config']}' (F1: {best_cfg['macro_f1']:.4f})")

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6: ANALYZING ABLATION RESULTS")
    print("=" * 60)
    
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        plot_ablation_comparison(df)
        print_summary_table(df)
    else:
        # Create a dummy dataframe for demonstration if file doesn't exist
        # This allows the user to see what the script will look like
        print(f"  ⚠️ {RESULTS_FILE} not found. Use 07_ablation_study.py to generate it.")
        print("  Creating an empty template analysis script.")
