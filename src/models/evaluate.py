"""Evaluation metrics for imbalanced binary risk ranking.

Accuracy is deliberately not reported as a headline metric: at ~6% positive
prevalence a trivial all-negative classifier scores ~94% accuracy while being
useless. See docs/model_evaluation.md.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_prevalence": float(np.mean(y_true)),
    }


def precision_recall_at_k(y_true, y_prob, k_fractions=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """For each review-budget fraction K, rank claims by predicted score and report
    precision@K (fraction of the top-K that are actually flagged) and recall@K
    (fraction of all true positives captured within the top-K).
    """
    n = len(y_true)
    order = np.argsort(-y_prob)
    y_sorted = np.asarray(y_true)[order]
    total_positives = y_sorted.sum()

    rows = []
    for k in k_fractions:
        k_n = max(1, int(round(n * k)))
        top_k = y_sorted[:k_n]
        precision_at_k = top_k.sum() / k_n
        recall_at_k = top_k.sum() / total_positives if total_positives > 0 else np.nan
        rows.append(
            {
                "k_fraction": k,
                "k_n_claims": k_n,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "flagged_in_top_k": int(top_k.sum()),
                "total_flagged": int(total_positives),
            }
        )
    return pd.DataFrame(rows)
