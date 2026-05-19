"""Metrics for NIH ChestX-ray14 multi-label classification."""

from __future__ import annotations

from typing import Any

import numpy as np


def _load_sklearn_metrics():
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Multi-label metrics require scikit-learn. Install requirements-train.txt "
            "or run in Kaggle with sklearn available."
        ) from exc
    return f1_score, precision_score, recall_score, roc_auc_score


def _to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def sigmoid_predictions(logits: Any) -> np.ndarray:
    """Convert logits to probabilities with sigmoid."""
    logits_np = _to_numpy(logits).astype(np.float32)
    return 1.0 / (1.0 + np.exp(-logits_np))


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray, roc_auc_score_func) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    try:
        return float(roc_auc_score_func(y_true, y_prob))
    except ValueError:
        return None


def compute_multilabel_metrics(
    y_true: Any,
    y_prob: Any,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute robust multi-label metrics from true labels and probabilities."""
    y_true_np = _to_numpy(y_true).astype(np.int32)
    y_prob_np = _to_numpy(y_prob).astype(np.float32)

    if y_true_np.ndim != 2 or y_prob_np.ndim != 2:
        raise ValueError("y_true and y_prob must be 2D arrays.")
    if y_true_np.shape != y_prob_np.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_np.shape}, y_prob={y_prob_np.shape}")

    f1_score, precision_score, recall_score, roc_auc_score = _load_sklearn_metrics()
    auc_per_class = [
        _safe_auc(y_true_np[:, class_index], y_prob_np[:, class_index], roc_auc_score)
        for class_index in range(y_true_np.shape[1])
    ]
    valid_auc = [auc for auc in auc_per_class if auc is not None]
    y_pred_np = (y_prob_np >= threshold).astype(np.int32)

    return {
        "auc_per_class": auc_per_class,
        "macro_auc": float(np.mean(valid_auc)) if valid_auc else None,
        "micro_auc": _safe_auc(y_true_np.ravel(), y_prob_np.ravel(), roc_auc_score),
        "f1_macro": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        ),
    }
