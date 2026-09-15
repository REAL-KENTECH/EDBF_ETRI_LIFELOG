"""prior.py — per-subject temporal prior and autocorrelation-weighted blend.

Stage 3 of the tabular view. A distance-weighted mean of the same subject's labels within a
+/-max_days window forms a per-day prior, blended into the Stage-2 hybrid by each label's lag-1
autocorrelation, so a label that persists day to day trusts the prior more. Leak-safe: the OOF prior
for a held fold draws labels from the train fold only.

Deployed: max_days = 30, power = 2.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def lag1_autocorr(df, y, labels):
    """Per-label lag-1 autocorrelation of same-subject adjacent-day labels → blend weights."""
    df = df.reset_index(drop=False).rename(columns={'index': 'row'})
    df['lifelog_date'] = pd.to_datetime(df['lifelog_date'])
    weights = []
    for li in range(len(labels)):
        pairs = []
        for s in df['subject_id'].unique():
            sd = df[df['subject_id'] == s].sort_values('lifelog_date').reset_index(drop=True)
            for i in range(len(sd) - 1):
                if (sd.loc[i + 1, 'lifelog_date'] - sd.loc[i, 'lifelog_date']).days == 1:
                    pairs.append((y[sd.loc[i, 'row'], li], y[sd.loc[i + 1, 'row'], li]))
        if len(pairs) > 10:
            p = np.array(pairs)
            ac = np.corrcoef(p[:, 0], p[:, 1])[0, 1]
            weights.append(max(0.0, ac))
        else:
            weights.append(0.0)
    return np.array(weights, dtype=np.float32)


def temporal_prior(target_subj, target_date, train_subj, train_date, y_train,
                   max_days=7, kernel='power', power=2.0, rho=None):
    """Distance-weighted same-subject prior from train labels within ±max_days.

    Args:
        kernel: 'power' (= 1/(d+1)^power) or 'ar1' (rho^d per-label, from an AR(1) panel model)
        rho:    when kernel='ar1', shape (n_labels,) per-label lag-1 autocorrelations
    Returns:
        prior shape (n_target, n_labels), and neighbor_count shape (n_target,)
    """
    n_target, n_labels = len(target_subj), y_train.shape[1]
    train_date = pd.to_datetime(train_date).to_numpy()
    target_date = pd.to_datetime(target_date).to_numpy()
    subj_match = target_subj[:, None] == train_subj[None, :]
    dist = np.abs((target_date[:, None] - train_date[None, :]) / np.timedelta64(1, 'D')).astype(int)
    mask = subj_match & (dist > 0) & (dist <= max_days)
    nbr_count = mask.sum(axis=1).astype(np.int32)
    valid_row = nbr_count > 0
    prior = np.full((n_target, n_labels), np.nan, dtype=np.float32)
    if kernel == 'power':
        w = np.where(mask, 1.0 / np.power(dist + 1.0, power), 0.0)
        wsum = w.sum(axis=1, keepdims=True)
        w_norm = (w / np.where(wsum > 0, wsum, 1.0)).astype(y_train.dtype)
        prior[valid_row] = (w_norm @ y_train)[valid_row]
    elif kernel == 'ar1':
        for li in range(n_labels):
            w = np.where(mask, rho[li] ** dist, 0.0)
            wsum = w.sum(axis=1)
            ok = wsum > 1e-9
            weighted = (w.astype(y_train.dtype) @ y_train[:, li]) / np.where(ok, wsum, 1.0).astype(y_train.dtype)
            prior[ok, li] = weighted[ok]
    return prior, nbr_count


def oof_temporal_prior(df, y, fold_splits, **kw):
    """Leakage-safe OOF version of temporal_prior: val rows use the train fold only."""
    prior = np.full(y.shape, np.nan, dtype=np.float32)
    nbr = np.zeros(len(df), dtype=np.int32)
    subj = df['subject_id'].to_numpy()
    date = df['lifelog_date'].to_numpy()
    for (tr_, va_) in fold_splits:
        p_va, n_va = temporal_prior(subj[va_], date[va_], subj[tr_], date[tr_], y[tr_], **kw)
        prior[va_] = p_va
        nbr[va_] = n_va
    return prior, nbr


def blend_with_prior(model_pred, prior, weights, eps=1e-12):
    """Blend model predictions with the temporal prior per label (weight = lag-1 autocorr)."""
    out = model_pred.copy()
    for li in range(model_pred.shape[1]):
        has = ~np.isnan(prior[:, li])
        out[has, li] = (1 - weights[li]) * model_pred[has, li] + weights[li] * prior[has, li]
    return out.clip(eps, 1 - eps)
