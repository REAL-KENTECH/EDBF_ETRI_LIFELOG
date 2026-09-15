"""Forward-CV folds: contiguous temporal blocks per subject. Used by views/build_context.py.
"""
from __future__ import annotations
import numpy as np


def lb_folds(keys, K=6):
    """`keys` is a DataFrame with [subject_id, lifelog_date]; returns K (train_idx, val_idx) folds."""
    s = keys['subject_id'].to_numpy(); d = keys['lifelog_date'].to_numpy()
    fold = np.full(len(keys), -1)
    for u in np.unique(s):
        o = np.where(s == u)[0]; o = o[np.argsort(d[o])]
        for k, c in enumerate(np.array_split(o, K)):
            fold[c] = k
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(K)]
