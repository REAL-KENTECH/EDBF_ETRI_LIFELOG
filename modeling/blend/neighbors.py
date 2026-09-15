"""neighbors.py — k-NN borrowing on the feature-embedding space.

Stage 2 of the tabular view. The Stage-1 per-label LGBM predictions (oof_base, sub_base) are smoothed
by borrowing from feature-space nearest neighbours, then convex-blended with the borrowed estimate by
the Bates-Granger weight (combine.bg_alpha).

Leak safety: on the OOF path each held fold's neighbours come only from the other folds, and the
standardisation statistics use train-fold rows only, so a validation row never contributes to its own
distances. The sub path uses full-train statistics, whose labels are unseen.
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from modeling.blend.combine import bg_alpha


def knn_fill(target, ref=None):
    """NaN-fill for the kNN embedding: NaN → 0 (after standardisation, the mean coordinate).

    `ref` is accepted for call-site symmetry; the deployed fill is reference-independent."""
    return np.nan_to_num(target)


def _nbr_weights(D, weighting):
    """Row-normalised neighbour weights from distances D (m,k): 'uniform' or 'invd2'
    (inverse-distance-squared)."""
    W = 1.0 / (D + 1e-6) ** 2 if weighting == 'invd2' else np.ones_like(D)
    return W / W.sum(axis=1, keepdims=True)


def _make_nn(k_nn, metric, p):
    """Deployed metric-string path (p=None), or Minkowski-p brute force when p is set (fractional
    p < 1 allowed)."""
    if p is not None:
        return NearestNeighbors(n_neighbors=k_nn, metric='minkowski', p=p, algorithm='brute')
    return NearestNeighbors(n_neighbors=k_nn, metric=metric)


def leakfree_neighbor_mean(base_oof, F_nan, fold_splits, k_nn, metric, weighting='uniform', p=None):
    """Per-row neighbour mean using only OTHER folds' rows. Per-fold: NaN-fill (ref = train-fold rows)
    then standardise with train-fold stats so val rows never contribute to their own embedding
    distances. Returns the borrowed estimate aligned to base_oof's rows."""
    n_mean = np.zeros_like(base_oof)
    for tr_, va_ in fold_splits:
        F_tr, F_va = knn_fill(F_nan[tr_], F_nan[tr_]), knn_fill(F_nan[va_], F_nan[tr_])
        mu = F_tr.mean(0)
        sd = F_tr.std(0) + 1e-06
        F_tr_fold = np.clip((F_tr - mu) / sd, -3, 3)
        F_va_fold = np.clip((F_va - mu) / sd, -3, 3)
        nn = _make_nn(k_nn, metric, p).fit(F_tr_fold)
        if weighting == 'uniform':
            idx = nn.kneighbors(F_va_fold, return_distance=False)
            n_mean[va_] = base_oof[tr_][idx].mean(axis=1)
        else:
            D, idx = nn.kneighbors(F_va_fold, return_distance=True)
            W = _nbr_weights(D, weighting)
            n_mean[va_] = (W[:, :, None] * base_oof[tr_][idx]).sum(axis=1)
    return n_mean


def knn_borrow(oof_base, sub_base, F_tr_nan, F_sb_nan, y, fold_splits, k, metric, weighting='uniform', p=None, eps=1e-12):
    """Stage 2 end-to-end: leak-safe neighbour borrow + per-label Bates-Granger convex blend.

    OOF: a held fold's α* is estimated from the OTHER folds' (m, n, y) only (leak-safe per-label).
    Sub: α* from the full OOF; neighbours from full-train (standardised with full-train stats).
    Returns (oof_hybrid, sub_hybrid, alpha) where alpha is the full-OOF per-label weight (diagnostic)."""
    F_tr_raw = knn_fill(F_tr_nan)
    F_sb_raw = knn_fill(F_sb_nan)
    F_mu, F_sd = F_tr_raw.mean(0), F_tr_raw.std(0) + 1e-06
    F_sb = np.clip((F_sb_raw - F_mu) / F_sd, -3, 3)

    n_oof = leakfree_neighbor_mean(oof_base, F_tr_nan, fold_splits, k, metric, weighting, p)

    oof_hybrid = np.zeros_like(oof_base)
    fold_of_row = np.empty(len(oof_base), dtype=int)
    for k_idx, (_, va) in enumerate(fold_splits):
        fold_of_row[va] = k_idx
    for k_idx, (_, va_) in enumerate(fold_splits):
        other = np.where(fold_of_row != k_idx)[0]
        a = bg_alpha(oof_base[other], n_oof[other], y[other])
        oof_hybrid[va_] = ((1 - a) * oof_base[va_] + a * n_oof[va_]).clip(eps, 1 - eps)

    alpha = bg_alpha(oof_base, n_oof, y)
    F_tr_full = np.clip((F_tr_raw - F_mu) / F_sd, -3, 3)
    nn_sub = _make_nn(k, metric, p).fit(F_tr_full)
    if weighting == 'uniform':
        sub_nbr_idx = nn_sub.kneighbors(F_sb, return_distance=False)
        n_sub = oof_base[sub_nbr_idx].mean(axis=1)
    else:
        D_sub, sub_nbr_idx = nn_sub.kneighbors(F_sb, return_distance=True)
        n_sub = (_nbr_weights(D_sub, weighting)[:, :, None] * oof_base[sub_nbr_idx]).sum(axis=1)
    sub_hybrid = ((1 - alpha) * sub_base + alpha * n_sub).clip(eps, 1 - eps)
    return oof_hybrid, sub_hybrid, alpha


def random_knn_borrow(oof_base, sub_base, F_tr_nan, F_sb_nan, y, fold_splits,
                      M=30, frac=0.8, k=2, metric='manhattan', seed=0, weighting='uniform', p=None, eps=1e-12):
    """Variance-reduced Stage-2 borrow: a random-subspace ensemble of M k-NN borrows, seed-fixed.

    Each member draws a random `frac` of the features, computes its own leak-safe neighbour mean, and
    the borrows are averaged."""
    rng = np.random.RandomState(seed)
    nf = F_tr_nan.shape[1]
    n_oof = np.zeros_like(oof_base)
    n_sub = np.zeros_like(sub_base)
    for _ in range(M):
        cols = rng.choice(nf, max(5, int(nf * frac)), replace=False)
        n_oof += leakfree_neighbor_mean(oof_base, F_tr_nan[:, cols], fold_splits, k, metric, weighting, p) / M
        F_tr_raw, F_sb_raw = knn_fill(F_tr_nan[:, cols]), knn_fill(F_sb_nan[:, cols])
        mu, sd = F_tr_raw.mean(0), F_tr_raw.std(0) + 1e-06
        F_tr_full = np.clip((F_tr_raw - mu) / sd, -3, 3)
        F_sb = np.clip((F_sb_raw - mu) / sd, -3, 3)
        nn_sub = _make_nn(k, metric, p).fit(F_tr_full)
        if weighting == 'uniform':
            idx = nn_sub.kneighbors(F_sb, return_distance=False)
            n_sub += oof_base[idx].mean(axis=1) / M
        else:
            D_sub, idx = nn_sub.kneighbors(F_sb, return_distance=True)
            n_sub += (_nbr_weights(D_sub, weighting)[:, :, None] * oof_base[idx]).sum(axis=1) / M
    oof_hybrid = np.zeros_like(oof_base)
    fold_of_row = np.empty(len(oof_base), dtype=int)
    for k_idx, (_, va) in enumerate(fold_splits):
        fold_of_row[va] = k_idx
    for k_idx, (_, va_) in enumerate(fold_splits):
        other = np.where(fold_of_row != k_idx)[0]
        a = bg_alpha(oof_base[other], n_oof[other], y[other])
        oof_hybrid[va_] = ((1 - a) * oof_base[va_] + a * n_oof[va_]).clip(eps, 1 - eps)
    alpha = bg_alpha(oof_base, n_oof, y)
    sub_hybrid = ((1 - alpha) * sub_base + alpha * n_sub).clip(eps, 1 - eps)
    return oof_hybrid, sub_hybrid, alpha
