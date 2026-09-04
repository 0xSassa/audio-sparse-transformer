"""Metriche per classe e analisi delle confusioni.

In SC-V2 lo sbilanciamento fra classi e' di 2,6x, quindi oltre all'accuratezza
aggregata si riportano quella bilanciata, le classi peggiori e le coppie piu'
confuse.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    """Matrice n x n: righe = vere, colonne = predette."""
    flat = np.bincount(y_true.astype(np.int64) * n + y_pred.astype(np.int64),
                       minlength=n * n)
    return flat.reshape(n, n)


def classification_summary(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], top_k: int = 12
) -> dict:
    """Accuratezza globale, per classe, e coppie piu' confuse."""
    n = len(labels)
    cm = confusion_matrix(y_true, y_pred, n)
    support = cm.sum(axis=1)
    correct = np.diag(cm)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(support > 0, correct / np.maximum(support, 1), np.nan)

    # media delle accuratezze per classe: insensibile allo sbilanciamento
    balanced = float(np.nanmean(per_class))

    confusions = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                confusions.append((int(cm[i, j]), labels[i], labels[j]))
    confusions.sort(reverse=True)

    order = np.argsort(per_class)
    return {
        "accuracy": float(correct.sum() / cm.sum()),
        "balanced_accuracy": balanced,
        "per_class": {
            labels[i]: {"accuracy": float(per_class[i]), "support": int(support[i])}
            for i in range(n)
        },
        "worst_classes": [
            {"label": labels[i], "accuracy": float(per_class[i]),
             "support": int(support[i])}
            for i in order[:top_k] if support[i] > 0
        ],
        "top_confusions": [
            {"true": t, "predicted": p, "count": c}
            for c, t, p in confusions[:top_k]
        ],
        "confusion_matrix": cm.tolist(),
    }


def save_summary(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_summary(summary: dict, top_k: int = 8) -> None:
    print(f"  accuratezza            : {100*summary['accuracy']:.2f}%")
    print(f"  accuratezza bilanciata : {100*summary['balanced_accuracy']:.2f}%")

    print("\n  classi peggiori:")
    for row in summary["worst_classes"][:top_k]:
        print(f"    {row['label']:<10} {100*row['accuracy']:>6.2f}%  "
              f"({row['support']} campioni)")

    print("\n  confusioni piu' frequenti:")
    for row in summary["top_confusions"][:top_k]:
        print(f"    {row['true']:<10} -> {row['predicted']:<10} {row['count']:>4}")
