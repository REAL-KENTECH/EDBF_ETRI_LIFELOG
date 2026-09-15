"""metrics.py — the competition metric: per-label binary log-loss averaged over the 7 labels.

`labels=[0, 1]` is always passed so a single-class column cannot be mis-scored, and predictions are
clipped to [eps, 1-eps] because log-loss is infinite at a confident-wrong 0 or 1.
"""
import numpy as np
from sklearn.metrics import log_loss


def binary_logloss(y_col, p_col, eps: float = 1e-12) -> float:
    """Single-label binary log-loss with epsilon clipping and explicit [0, 1] class labels."""
    return float(log_loss(y_col, np.clip(p_col, eps, 1 - eps), labels=[0, 1]))


def perlabel_logloss(P, y, eps: float = 1e-12) -> list:
    """Per-label binary log-loss → list of length L (= P.shape[1]). The mechanism breakdown."""
    return [binary_logloss(y[:, l], P[:, l], eps) for l in range(P.shape[1])]


def macro_logloss(P, y, eps: float = 1e-12) -> float:
    """Macro (label-averaged) binary log-loss — the competition score of record."""
    return float(np.mean(perlabel_logloss(P, y, eps)))
