"""combine.py — the convex-combination rules.

`bg_alpha` is the 2-way Bates-Granger weight, clipped to [0, 1]; `global_bg_weights` is the N-way
generalization with a Ledoit-Wolf shrunk error covariance. The reduction is over axis 0, so both work
on per-label matrices of shape (N, L), giving (L,) weights, and on a single-label column of shape
(N,), giving a scalar.
"""
import numpy as np


def bg_alpha(m, n, y, eps=1e-12):
    """Bates-Granger optimal convex weight α for the blend (1-α)·m + α·n minimizing squared error.

        α* = (Var(e_m) − Cov(e_m, e_n)) / Var(e_m − e_n),   e_m = m − y,  e_n = n − y

    Reduces over axis 0: m/n/y of shape (N, L) → (L,) per-label weights; shape (N,) → a 0-d scalar.
    Clipped to [0, 1] (no extrapolation outside m, n)."""
    e_m = m - y
    e_n = n - y
    var_m = e_m.var(axis=0)
    var_diff = (e_m - e_n).var(axis=0) + eps
    cov_mn = ((e_m - e_m.mean(0, keepdims=True)) * (e_n - e_n.mean(0, keepdims=True))).mean(0)
    return np.clip((var_m - cov_mn) / var_diff, 0.0, 1.0)


def global_bg_weights(P, y, shrink='lw'):
    """N-way generalization of `bg_alpha`: closed-form, order-free simplex weights over J views.

    P:(J,m) view probabilities, y:(m,). Returns simplex weights q:(J,). The error covariance is
    regularised by Ledoit-Wolf (2004) shrinkage (delta* closed-form from the data when shrink='lw');
    the weights are the Bates-Granger optimum q ~ Pi_simplex(Sigma_e^{-1} 1). The simplex projection
    is thresholdless: a useless view gets weight 0, duplicates share weight. A fixed float for
    `shrink` is accepted for ablation."""
    e = P - y[None, :]                                    # (J,m) view errors
    if len(P) == 1:
        return np.array([1.0])
    if shrink == 'lw':
        from sklearn.covariance import ledoit_wolf
        S, _ = ledoit_wolf(e.T)                           # data-derived optimal shrinkage (no given value)
    else:
        S = np.atleast_2d(np.cov(e)); S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
    q = np.linalg.solve(S + 1e-12 * np.eye(len(S)), np.ones(len(S)))
    q = np.clip(q, 0, None)
    return q / (q.sum() + 1e-12)
