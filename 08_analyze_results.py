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
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
RESULTS_FILE = "./results/ablation/ablation_results.csv"
CHECKPOINT_DIR = "./checkpoints/ablation"
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

def plot_training_curves(df, state):
    print(f"  📊 Generating individual training curves for {state}...")
    
    configs = df['config'].unique()
    for config in configs:
        hist_path = os.path.join(CHECKPOINT_DIR, f"{state}_{config}_history.pt")
        if not os.path.exists(hist_path):
            continue
            
        history = torch.load(hist_path, weights_only=False)
        epochs = range(1, len(history["train_loss"]) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"MCTNet Training History: {state} — {config}", fontsize=15, fontweight='bold')
        
        # 1. Loss Curve
        axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
        axes[0].set_title("Loss over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Accuracy Curve
        axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#2ca02c", linewidth=2)
        axes[1].plot(epochs, history["val_acc"], label="Val Acc", color="#d62728", linewidth=2)
        axes[1].set_title("Accuracy over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1.02)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"training_curves_{state.lower()}_{config}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close()
        print(f"    ✓ Saved: {filename}")

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
        print(f"\n[BEST] Configuration for {state}: '{best_cfg['config']}' (F1: {best_cfg['macro_f1']:.4f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Ablation Results")
    parser.add_argument("--state", type=str, default=None, help="State to analyze (Arkansas/California)")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 6: ANALYZING ABLATION RESULTS")
    print("=" * 60)
    
    # Check for state-specific or all results
    if args.state:
        filename = f"ablation_results_{args.state.lower()}.csv"
    else:
        filename = "ablation_results_all.csv"
        # Fallback to old name if needed
        if not os.path.exists(os.path.join(os.path.dirname(RESULTS_FILE), filename)):
            filename = "ablation_results.csv"

    results_path = os.path.join(os.path.dirname(RESULTS_FILE), filename)

    if os.path.exists(results_path):
        print(f"  📂 Loading: {results_path}")
        df = pd.read_csv(results_path)
        plot_ablation_comparison(df)
        print_summary_table(df)
        
        # Generate curves for each state found in the results
        for state in df['state'].unique():
            plot_training_curves(df[df['state'] == state], state)
    else:
        print(f"  ⚠️ {results_path} not found.")
        print("  Available files in results/ablation/:")
        for f in os.listdir(os.path.dirname(RESULTS_FILE)):
            if f.endswith(".csv"): print(f"    - {f}")
