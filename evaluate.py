import json
from pathlib import Path
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from utils import RESULTS_DIR, FIGURES_DIR


def compute_metrics(y_true, y_pred, labels: Sequence[str]) -> dict:
    return {
        "accuracy":      accuracy_score(y_true, y_pred),
        "macro_f1":      f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1":   f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_f1":  f1_score(y_true, y_pred, average=None,
                                   labels=list(range(len(labels))), zero_division=0).tolist(),
        "report":        classification_report(y_true, y_pred,
                                                target_names=labels,
                                                zero_division=0,
                                                output_dict=True),
    }


def save_metrics(metrics: dict, run_name: str) -> Path:
    out = RESULTS_DIR / f"{run_name}.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {out}")
    return out


def plot_confusion_matrix(y_true, y_pred, labels: Sequence[str],
                          run_name: str, normalize: bool = True) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    if normalize:
        with np.errstate(invalid="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)),
                                     max(5, 0.5 * len(labels))))
    sns.heatmap(cm, annot=(len(labels) <= 10), fmt=".2f" if normalize else "d",
                xticklabels=labels, yticklabels=labels,
                cmap="Blues", cbar=True, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {run_name}"
                 + (" (row-normalised)" if normalize else ""))
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = FIGURES_DIR / f"cm_{run_name}.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved figure -> {out}")
    return out


def rare_class_f1(per_class_f1: Sequence[float], n: int = 5) -> float:
    arr = np.array(per_class_f1)
    return float(np.sort(arr)[:n].mean())