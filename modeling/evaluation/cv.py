"""cv.py — cross-validation splits.

Imported by tabular_model.py and features/tabular.py to build the leak-safe (train, val) folds shared
by the CNN, LGBM, k-NN and temporal-prior stages.

`subject_blocked_temporal_kfold` sorts each subject's rows by date and splits them into K contiguous
temporal blocks, rotating block assignment by subject ordinal for balance.
"""
import numpy as np
import pandas as pd


def subject_blocked_temporal_kfold(subject_ids, dates, n_splits: int = 5):
    """Return list of (train_idx, val_idx) — same shape contract as KFold.split()."""
    subject_ids = np.asarray(subject_ids)
    dates = pd.to_datetime(np.asarray(dates)).to_numpy()
    n = len(subject_ids)
    fold_ids = np.empty(n, dtype=int)
    for s_idx, s in enumerate(sorted(set(subject_ids))):
        idx = np.where(subject_ids == s)[0]
        sorted_idx = idx[np.argsort(dates[idx])]
        for k, block in enumerate(np.array_split(sorted_idx, n_splits)):
            fold_ids[block] = (k + s_idx) % n_splits
    return [(np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]) for k in range(n_splits)]
