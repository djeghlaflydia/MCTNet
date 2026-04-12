"""
Evaluation metrics for crop classification.
Paper: MCTNet — Wang et al., 2024 (section 2.5)

Metrics:
    - OA  : Overall Accuracy
    - AA  : Average Accuracy (mean of per-class accuracies)
    - Kappa: Cohen's Kappa coefficient
    - F1  : Macro F1-Score
    - Per-class F1, Precision, Recall
    - Confusion Matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, class_names=None):
    """
    Compute all evaluation metrics used in the MCTNet paper.

    Args:
        y_true      : array-like — ground truth labels
        y_pred      : array-like — predicted labels
        class_names : list of str — optional class names for report

    Returns:
        dict with keys: OA, AA, Kappa, F1_macro, per_class_f1,
                        confusion_matrix, classification_report
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Overall Accuracy
    oa = accuracy_score(y_true, y_pred)

    # Average Accuracy (mean of per-class accuracies)
    cm = confusion_matrix(
        y_true, y_pred, 
        labels=np.arange(len(class_names)) if class_names is not None else None
    )
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    aa = per_class_acc.mean()

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Macro F1
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Per-class Precision & Recall
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Classification report (string)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=np.arange(len(class_names)) if class_names is not None else None,
        zero_division=0,
    )

    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "F1_macro": f1_macro,
        "per_class_f1": per_class_f1,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def print_metrics(metrics, title="Evaluation Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")
    print(f"  Overall Accuracy (OA)  : {metrics['OA']:.4f}")
    print(f"  Average Accuracy (AA)  : {metrics['AA']:.4f}")
    print(f"  Cohen's Kappa          : {metrics['Kappa']:.4f}")
    print(f"  Macro F1-Score         : {metrics['F1_macro']:.4f}")
    print(f"\n{metrics['classification_report']}")
    print(f"{'=' * 55}")
