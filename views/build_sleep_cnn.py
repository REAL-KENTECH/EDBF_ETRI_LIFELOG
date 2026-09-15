"""build_sleep_cnn.py — the sleep-tensor CNN view.

Trains the multi-resolution pyramid CNN and writes cache/view_sleep_cnn.npz (keys pyr_o, pyr_s).
Requires a GPU and cache/sleep_window_tensor.npz (run preprocess/sleep_tensor.py first).

The deployed build prunes the sleep-tensor MiniRocket co-view (fuse.py USE_MR=False), so only the
pyramid CNN is built by default. --with-mr also trains the {5,10,15}min MiniRocket scales (keys
mr5o/mr10o/mr15o and the sub counterparts), which fuse.py needs when USE_MR=True.

Seeded and cuDNN-deterministic, but the GPU draw differs across machines, so a rebuild does not
byte-reproduce the shipped npz. An existing cache/view_sleep_cnn.npz is reused.

Run: python views/build_sleep_cnn.py [--with-mr]
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys, argparse; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from modeling.config import DATA_DIR, CACHE_DIR as CACHE
from modeling.features.tabular import load_features

EPS = 1e-12; LAB = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
SEEDS = (42, 43, 44, 45, 46); SCALES = (1, 2, 3)   # 5/10/15-min pooling factors
OUT = CACHE / 'view_sleep_cnn.npz'
TENSOR = CACHE / 'sleep_window_tensor.npz'


def _align():
    tz = np.load(TENSOR, allow_pickle=True)
    X, Mk = tz['X'].astype(np.float32), tz['mask'].astype(np.float32)
    cov = Mk.mean(axis=(0, 2)); kc = [c for c in range(Mk.shape[1]) if cov[c] >= 0.1]
    X, Mk = X[:, kc, :], Mk[:, kc, :]
    tlk = pd.DataFrame({'subject_id': tz['subject_id'],
                        'lifelog_date': pd.to_datetime(tz['lifelog_date']).normalize(), 'i': np.arange(len(tz['subject_id']))})
    feat = load_features(DATA_DIR, labels=LAB)['df'].copy(); feat['lifelog_date'] = pd.to_datetime(feat['lifelog_date']).dt.normalize()
    trd = feat[feat.split == 'train'].sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
    sbd = feat[feat.split == 'sub'].sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
    tri = trd[['subject_id', 'lifelog_date']].merge(tlk, on=['subject_id', 'lifelog_date']).i.values.astype(int)
    sbi = sbd[['subject_id', 'lifelog_date']].merge(tlk, on=['subject_id', 'lifelog_date']).i.values.astype(int)
    subj_u = sorted(trd.subject_id.unique()); s2i = {s: i for i, s in enumerate(subj_u)}
    str_ = np.array([s2i[s] for s in trd.subject_id], np.int64); ssub = np.array([s2i[s] for s in sbd.subject_id], np.int64)
    y = trd[LAB].to_numpy().astype(np.float32)
    return X[tri], Mk[tri], str_, X[sbi], Mk[sbi], ssub, y, len(subj_u)


def _pool(X, f):
    if f == 1: return X
    N, C, T = X.shape; t = (T // f) * f
    return X[:, :, :t].reshape(N, C, t // f, f).mean(axis=3)


def _train_cnn_5seed(model_cls, Xtr, Mtr, str_, Xsb, Msb, ssub, y, n_subj, folds):
    from modeling.encoders.cnn import normalize_per_fold, train_cnn, predict_cnn
    n, m, nl, nch = len(y), len(Xsb), 7, Xtr.shape[1]
    oof = np.zeros((len(SEEDS), n, nl)); sub = np.zeros((len(SEEDS), m, nl))
    for si, seed in enumerate(SEEDS):
        acc = np.zeros((m, nl))
        for tr_, va_ in folds:
            Xn, Xvn, Xsn = normalize_per_fold(Xtr[tr_], Mtr[tr_], Xtr[va_], Xsb)
            np.random.seed(seed); perm = np.random.permutation(len(tr_)); v = max(20, len(tr_) // 10)
            ts, vs = perm[v:], perm[:v]
            mdl = train_cnn(Xn[ts], Mtr[tr_][ts], str_[tr_][ts], y[tr_][ts], Xn[vs], Mtr[tr_][vs], str_[tr_][vs], y[tr_][vs],
                            n_ch=nch, n_subj=n_subj, seed=seed, model_cls=model_cls)
            oof[si, va_] = predict_cnn(mdl, Xvn, Mtr[va_], str_[va_]); acc += predict_cnn(mdl, Xsn, Msb, ssub)
        sub[si] = acc / len(folds)
    return oof.mean(0), sub.mean(0)


def _mr_scale(Xtr, Xsb, y, folds, f):
    Xt, Xs = _pool(Xtr, f).astype(np.float32), _pool(Xsb, f).astype(np.float32)
    n, m = len(y), len(Xsb); oof = np.zeros((n, 7)); sub = np.zeros((m, 7))
    for sd in SEEDS:
        for tr_, va_ in folds:
            t = MiniRocketMultivariate(num_kernels=10000, random_state=sd)
            Ft = np.asarray(t.fit_transform(Xt[tr_]), np.float32); Fv = np.asarray(t.transform(Xt[va_]), np.float32)
            sc = StandardScaler().fit(Ft); Ft, Fv = sc.transform(Ft), sc.transform(Fv)
            for l in range(7):
                if len(np.unique(y[tr_, l])) < 2: oof[va_, l] += y[tr_, l].mean(); continue
                rc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(Ft, y[tr_, l])
                pl = LogisticRegression(C=1e6, max_iter=2000).fit(rc.decision_function(Ft).reshape(-1, 1), y[tr_, l])
                oof[va_, l] += pl.predict_proba(rc.decision_function(Fv).reshape(-1, 1))[:, 1]
        t = MiniRocketMultivariate(num_kernels=10000, random_state=sd)
        Fa = np.asarray(t.fit_transform(Xt), np.float32); Fs = np.asarray(t.transform(Xs), np.float32)
        sc = StandardScaler().fit(Fa); Fa, Fs = sc.transform(Fa), sc.transform(Fs)
        for l in range(7):
            if len(np.unique(y[:, l])) < 2: sub[:, l] += y[:, l].mean(); continue
            rc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(Fa, y[:, l])
            pl = LogisticRegression(C=1e6, max_iter=2000).fit(rc.decision_function(Fa).reshape(-1, 1), y[:, l])
            sub[:, l] += pl.predict_proba(rc.decision_function(Fs).reshape(-1, 1))[:, 1]
    return np.clip(oof / len(SEEDS), EPS, 1 - EPS), np.clip(sub / len(SEEDS), EPS, 1 - EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-mr', action='store_true', help='also train the {5,10,15}min MiniRocket scales (only needed for fuse.py USE_MR=True)')
    args = ap.parse_args()
    if OUT.exists():
        print(f'{OUT.name} already present — REUSING the frozen artifact (a from-raw rebuild is within-noise but '
              f'NOT byte-identical; delete it to retrain).'); return
    if not TENSOR.exists():
        sys.exit(f'{TENSOR} missing — run `python preprocess/sleep_tensor.py` first to build it from raw.')
    from modeling.encoders.cnn import PyramidSleepCNN
    Xtr, Mtr, str_, Xsb, Msb, ssub, y, n_subj = _align()
    folds = list(KFold(5, shuffle=True, random_state=42).split(np.arange(len(y))))
    print('training pyramid CNN (5/10/15min branches, 5 seeds)...')
    pyr_o, pyr_s = _train_cnn_5seed(PyramidSleepCNN, Xtr, Mtr, str_, Xsb, Msb, ssub, y, n_subj, folds)
    extra = {}
    if args.with_mr:
        for f, tag in zip(SCALES, ('5', '10', '15')):
            print(f'MiniROCKET scale {tag}min...'); extra[f'mr{tag}o'], extra[f'mr{tag}s'] = _mr_scale(Xtr, Xsb, y, folds, f)
    np.savez(OUT, pyr_o=pyr_o, pyr_s=pyr_s, **extra)
    print(f'wrote {OUT.name} (pyr{" + mr" if args.with_mr else ""}) — fresh retrain, within-noise-equivalent, NOT byte-identical to the shipped draw.')


if __name__ == '__main__':
    main()
