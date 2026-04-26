"""
=======================================================================
PART 3 — STEP 2 : VISUALIZATION & COMPARISON ANALYSIS
Project: Deep Learning for Crop Classification (USTHB)

Produces:
    - Bar chart: MCTNet vs ECMTNet on all metrics (OA, F1, Kappa)
    - Training curves comparison (both models on same plot)
    - Per-class F1 comparison
    - Summary table
=======================================================================
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR   = "./results/part3"
CHECKPOINT_DIR = "./checkpoints/part3"
OUTPUT_DIR    = "./results/part3/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["MCTNet", "ECMTNet"]
STATES = ["Arkansas", "California"]

AR_CLASSES = ["Others", "Corn", "Cotton", "Rice", "Soybeans"]
CA_CLASSES = ["Others", "Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios"]

# Paper baselines for reference
PAPER_BASELINES = {
    "Arkansas":   {"OA": 0.968, "Kappa": 0.951, "F1_macro": 0.933},
    "California": {"OA": 0.852, "Kappa": 0.806, "F1_macro": 0.829},
}


def load_metrics(state, model_name):
    path = os.path.join(RESULTS_DIR, state, model_name, "test_metrics.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


def plot_model_comparison(state, config="baseline"):
    """Bar chart comparing MCTNet vs ECMTNet across OA, Kappa, F1."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Part 3 Model Comparison — {state} ({config})",
                 fontsize=14, fontweight="bold")

    metric_keys = ["OA", "Kappa", "F1_macro"]
    metric_labels = ["Overall Accuracy", "Cohen's Kappa", "Macro F1-Score"]
    colors = {"MCTNet": "#1f77b4", "ECMTNet": "#d62728"}

    paper_ref = PAPER_BASELINES.get(state, {})

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        values = {}
        for model in MODELS:
            data = load_metrics(state, model)
            if data:
                m = data["metrics"]
                values[model] = m.get(key, m.get("F1_macro" if key == "F1_macro" else key, 0))

        if not values:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            continue

        bars = ax.bar(list(values.keys()), list(values.values()),
                      color=[colors[m] for m in values.keys()],
                      edgecolor="white", linewidth=0.8, width=0.5)

        # Paper baseline reference line
        if key in paper_ref:
            ax.axhline(paper_ref[key], color="gray", linestyle="--", linewidth=1.5,
                       label=f"Paper ({paper_ref[key]:.3f})")
            ax.legend(fontsize=8)

        for bar, val in zip(bars, values.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"comparison_{state.lower()}_{config}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_training_curves_comparison(state, config="baseline"):
    """Plot train/val accuracy for both models on the same axes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves Comparison — {state} ({config})",
                 fontsize=14, fontweight="bold")

    colors = {
        "MCTNet_train":  "#1f77b4",
        "MCTNet_val":    "#aec7e8",
        "ECMTNet_train": "#d62728",
        "ECMTNet_val":   "#f4a582",
    }

    for model in MODELS:
        hist_path = os.path.join(CHECKPOINT_DIR, state, model, "history.pt")
        if not os.path.exists(hist_path):
            continue
        history = torch.load(hist_path, weights_only=False)
        epochs  = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"],
                     label=f"{model} Train", color=colors[f"{model}_train"], linewidth=2)
        axes[0].plot(epochs, history["val_loss"],
                     label=f"{model} Val", color=colors[f"{model}_val"], linewidth=2, linestyle="--")

        axes[1].plot(epochs, history["train_acc"],
                     label=f"{model} Train", color=colors[f"{model}_train"], linewidth=2)
        axes[1].plot(epochs, history["val_acc"],
                     label=f"{model} Val", color=colors[f"{model}_val"], linewidth=2, linestyle="--")

    for ax, title, ylabel in [(axes[0], "Loss", "Loss"), (axes[1], "Accuracy", "Accuracy")]:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[1].set_ylim(0, 1.02)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"training_curves_{state.lower()}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_per_class_f1_comparison(state, config="baseline"):
    """Per-class F1 bar chart: MCTNet vs ECMTNet."""
    class_names = AR_CLASSES if state == "Arkansas" else CA_CLASSES
    n = len(class_names)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Per-Class F1 Score — {state} ({config})",
                 fontsize=13, fontweight="bold")

    colors = {"MCTNet": "#1f77b4", "ECMTNet": "#d62728"}

    for i, model in enumerate(MODELS):
        data = load_metrics(state, model)
        if not data:
            continue
        f1 = data["metrics"].get("per_class_f1", [])
        if len(f1) == n:
            bars = ax.bar(x + i * width, f1, width, label=model,
                          color=colors[model], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, f1):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Crop Type")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"per_class_f1_{state.lower()}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_summary_table():
    """Print a full comparison table for all states and models."""
    rows = []
    for state in STATES:
        paper = PAPER_BASELINES.get(state, {})
        # Add paper reference row
        rows.append({
            "State": state, "Model": "Paper (MCTNet)",
            "OA": paper.get("OA", "-"),
            "Kappa": paper.get("Kappa", "-"),
            "F1_macro": paper.get("F1_macro", "-"),
        })
        for model in MODELS:
            data = load_metrics(state, model)
            if data:
                m = data["metrics"]
                rows.append({
                    "State":    state,
                    "Model":    model,
                    "OA":       round(m.get("OA", 0), 4),
                    "Kappa":    round(m.get("Kappa", 0), 4),
                    "F1_macro": round(m.get("F1_macro", 0), 4),
                })

    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "=" * 65)
        print("  PART 3 FULL SUMMARY TABLE")
        print("=" * 65)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(RESULTS_DIR, "part3_summary.csv"), index=False)
        print(f"\n  Saved: {RESULTS_DIR}/part3_summary.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Part 3 — Results Analysis")
    parser.add_argument("--state", type=str, default="Both",
                        choices=["Arkansas", "California", "Both"])
    parser.add_argument("--config", type=str, default="baseline")
    args = parser.parse_args()

    states_to_analyze = ["Arkansas", "California"] if args.state == "Both" else [args.state]

    print("=" * 65)
    print("PART 3 — RESULTS ANALYSIS")
    print("=" * 65)

    for state in states_to_analyze:
        print(f"\n  {state}...")
        plot_model_comparison(state, args.config)
        plot_training_curves_comparison(state, args.config)
        plot_per_class_f1_comparison(state, args.config)

    print_summary_table()
    print(f"\n  All plots saved to: {OUTPUT_DIR}/")
