"""build_context.py — the nocturnal-context view.

MiniRocket (random kernels, 5 seeds, CPU-deterministic) over the 4-channel nocturnal tensor.
No pretrained weights; the kernels are seeded, so the output is reproducible.

Input   cache/context_tensor.npz   (preprocess/context.py; must be built first)
        cache/features.parquet     (canonical train/sub row keys and labels)
Output  cache/view_context.npz     keys oof (450, 7), sub (250, 7)

An existing cache/view_context.npz is reused, so views/fuse.py runs without raw data or the tensor.
Delete it to rebuild.

Run: python views/build_context.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from modeling.config import DATA_DIR, CACHE_DIR as CACHE
from modeling.features.tabular import load_features
from _folds import lb_folds

EPS = 1e-12; LAB = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
SEEDS = (42, 43, 44, 45, 46)
TENSOR = CACHE / 'context_tensor.npz'
cache = CACHE / 'view_context.npz'

# canonical train/sub keys (same order views/fuse.py uses: features.parquet, sorted by subject, date)
data = load_features(DATA_DIR, labels=LAB)
feat = data['df'].copy(); feat['lifelog_date'] = pd.to_datetime(feat['lifelog_date'])
trd = feat[feat.split == 'train'].sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
sbd = feat[feat.split == 'sub'].sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
keys = trd[['subject_id', 'lifelog_date']].copy(); keys['lifelog_date'] = keys['lifelog_date'].dt.normalize()
sub_keys = sbd[['subject_id', 'lifelog_date']].copy(); sub_keys['lifelog_date'] = sub_keys['lifelog_date'].dt.normalize()
y = trd[LAB].to_numpy().astype(float)


def _align():
    """Context-tensor rows for the canonical train and sub keys, selected by merge on those keys
    (same pattern as views/build_sleep_cnn.py:_align)."""
    tz = np.load(TENSOR, allow_pickle=True)
    tlk = pd.DataFrame({'subject_id': tz['subject_id'],
                        'lifelog_date': pd.to_datetime(tz['lifelog_date']), 'i': np.arange(len(tz['subject_id']))})
    tri = keys.merge(tlk, on=['subject_id', 'lifelog_date']).i.values.astype(int)
    sbi = sub_keys.merge(tlk, on=['subject_id', 'lifelog_date']).i.values.astype(int)
    assert len(tri) == len(keys), f'train row mismatch: keys {len(keys)}, matched {len(tri)}'
    assert len(sbi) == len(sub_keys), f'sub row mismatch: keys {len(sub_keys)}, matched {len(sbi)}'
    X = tz['X'].astype(np.float32)
    return X[tri], X[sbi], [str(c) for c in tz['channels']]


def main():
    if cache.exists():
        print(f'{cache.name} already present — reusing the shipped artifact (delete it to rebuild from /data).')
        return
    if not TENSOR.exists():
        sys.exit(f'{TENSOR.name} missing — run `python preprocess/context.py` first to build it from raw.')
    Xtr, Xsb, channels = _align()
    print('MiniROCKET context co-encoder (5 seeds, CPU-deterministic)...')
    n, m = len(y), len(sub_keys); k6 = lb_folds(keys, K=6)
    mr_oof = np.zeros((n, 7)); mr_sub = np.zeros((m, 7))
    for sd in SEEDS:
        for tr_, va_ in k6:
            t = MiniRocketMultivariate(num_kernels=10000, random_state=sd)
            Ftr = np.asarray(t.fit_transform(Xtr[tr_]), np.float32); Fva = np.asarray(t.transform(Xtr[va_]), np.float32)
            sc = StandardScaler().fit(Ftr); Ftr, Fva = sc.transform(Ftr), sc.transform(Fva)
            for l in range(7):
                if len(np.unique(y[tr_, l])) < 2: mr_oof[va_, l] += y[tr_, l].mean(); continue
                rc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(Ftr, y[tr_, l])
                pl = LogisticRegression(C=1e6, max_iter=2000).fit(rc.decision_function(Ftr).reshape(-1, 1), y[tr_, l])
                mr_oof[va_, l] += pl.predict_proba(rc.decision_function(Fva).reshape(-1, 1))[:, 1]
        t = MiniRocketMultivariate(num_kernels=10000, random_state=sd)
        Fa = np.asarray(t.fit_transform(Xtr), np.float32); Fs = np.asarray(t.transform(Xsb), np.float32)
        sc = StandardScaler().fit(Fa); Fa, Fs = sc.transform(Fa), sc.transform(Fs)
        for l in range(7):
            if len(np.unique(y[:, l])) < 2: mr_sub[:, l] += y[:, l].mean(); continue
            rc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(Fa, y[:, l])
            pl = LogisticRegression(C=1e6, max_iter=2000).fit(rc.decision_function(Fa).reshape(-1, 1), y[:, l])
            mr_sub[:, l] += pl.predict_proba(rc.decision_function(Fs).reshape(-1, 1))[:, 1]
    mr_oof = np.clip(mr_oof / len(SEEDS), EPS, 1 - EPS); mr_sub = np.clip(mr_sub / len(SEEDS), EPS, 1 - EPS)
    np.savez(cache, oof=mr_oof, sub=mr_sub, seeds=np.array(SEEDS), channels=np.array(channels))
    print(f'wrote {cache.name}  oof={mr_oof.shape} sub={mr_sub.shape}')


if __name__ == '__main__':
    main()
