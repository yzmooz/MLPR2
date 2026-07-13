from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_roc_auc(y_true, y_score, average: str | None = None) -> float:
    try:
        return float(roc_auc_score(y_true, y_score, average=average))
    except ValueError:
        return math.nan


def _safe_average_precision(y_true, y_score, average: str | None = None) -> float:
    try:
        return float(average_precision_score(y_true, y_score, average=average))
    except ValueError:
        return math.nan


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    # метрики ROC-AUC и PR-AUC используют вероятности, а F1 уже зависит от выбранного порога
    y_pred = (y_score_arr >= threshold).astype(int)
    return {
        "roc_auc": _safe_roc_auc(y_true_arr, y_score_arr),
        "pr_auc": _safe_average_precision(y_true_arr, y_score_arr),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
    }


def multilabel_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    # micro считает все ответы вместе, macro дает каждому классу одинаковый вес
    y_pred = (y_score_arr >= threshold).astype(int)
    return {
        "micro_roc_auc": _safe_roc_auc(y_true_arr, y_score_arr, average="micro"),
        "macro_roc_auc": _safe_roc_auc(y_true_arr, y_score_arr, average="macro"),
        "micro_pr_auc": _safe_average_precision(y_true_arr, y_score_arr, average="micro"),
        "macro_pr_auc": _safe_average_precision(y_true_arr, y_score_arr, average="macro"),
        "micro_f1": float(f1_score(y_true_arr, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
    }


def top_k_predictions(scores: list[float], label_names: list[str], k: int = 5) -> list[tuple[str, float]]:
    pairs = sorted(zip(label_names, scores, strict=True), key=lambda item: item[1], reverse=True)
    return [(label, float(score)) for label, score in pairs[:k]]
