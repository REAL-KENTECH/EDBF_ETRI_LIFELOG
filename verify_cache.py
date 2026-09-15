#!/usr/bin/env python
"""verify_cache.py — check the input caches against the shapes in README.md.

Runs no model and writes nothing. Exits 1 on any FAIL.

features.parquet, sleep_window_tensor.npz and context_tensor.npz are built by three separate scripts
and merged on (subject_id, lifelog_date), so their row order is checked against each other.

Run: python verify_cache.py
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache'
KEY = ['subject_id', 'lifelog_date']
LAB = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
N_ROWS, N_TRAIN, N_SUB = 700, 450, 250
EXPECT_MD5 = '3b0f2da7'

_state = {'fail': 0, 'skip': 0}


def chk(cond: bool, msg: str) -> bool:
    ok = bool(cond)
    print(('  OK    ' if ok else '  FAIL  ') + msg)
    if not ok:
        _state['fail'] += 1
    return ok


def skip(msg: str) -> None:
    print('  SKIP  ' + msg)
    _state['skip'] += 1


def check_table(name: str, n_feat: int | None, sub_labels: str, optional: bool = False) -> pd.DataFrame | None:
    """Shared contract for the three feature parquets.

    sub_labels: 'nan' (daylevel.py merges train labels only) or 'placeholder' (timing.py carries the
    label columns of ch2026_submission_sample.csv, which ship filled with 0)."""
    print(f'\n{name}')
    p = CACHE / name
    if not p.exists():
        if optional:
            skip(f'{name} not present (views/fuse.py rebuilds it on every run)')
        else:
            chk(False, f'{name} not found under {CACHE}')
        return None
    df = pd.read_parquet(p)
    chk(len(df) == N_ROWS, f'{N_ROWS} rows (got {len(df)})')
    chk(df[KEY].duplicated().sum() == 0, 'no duplicate (subject_id, lifelog_date)')
    chk(df.columns.duplicated().sum() == 0, 'no duplicate column names')
    chk((df.split == 'train').sum() == N_TRAIN and (df.split == 'sub').sum() == N_SUB,
        f'{N_TRAIN} train / {N_SUB} sub rows')
    chk(df.loc[df.split == 'train', LAB].notna().all().all(), 'train rows carry all 7 labels')
    sub_lab = df.loc[df.split == 'sub', LAB]
    if sub_labels == 'nan':
        chk(sub_lab.isna().all().all(), 'sub rows carry no labels (NaN)')
    else:
        chk((sub_lab.fillna(-1) == 0).all().all(),
            'sub label columns are the all-zero submission-sample placeholder, not labels')
    if n_feat is not None:
        feat = [c for c in df.columns if c not in KEY + ['sleep_date', 'split'] + LAB]
        chk(len(feat) == n_feat, f'{n_feat} feature columns, labels excluded (got {len(feat)})')
    return df


def main() -> int:
    print(f'cache: {CACHE}')

    feats = check_table('features.parquet', 281, sub_labels='nan')
    check_table('timing_features.parquet', 19, sub_labels='placeholder')
    # Rebuilt by views/fuse.py on every run and gitignored, so absence is not a failure.
    check_table('timing_events.parquet', 52, sub_labels='placeholder', optional=True)

    print('\nsleep_window_tensor.npz')
    p = CACHE / 'sleep_window_tensor.npz'
    if not p.exists():
        skip('not present — build it with: python preprocess/sleep_tensor.py')
    else:
        # date fields are object arrays of strings; every reader passes allow_pickle=True.
        z = np.load(p, allow_pickle=True)
        chk(z['X'].shape == (N_ROWS, 6, 108), f"X (700, 6, 108) (got {z['X'].shape})")
        chk(z['mask'].shape == z['X'].shape, 'mask matches X')
        chk(z['X'].dtype == np.float32 and z['mask'].dtype == np.int8, 'dtypes float32 / int8')
        chk(np.isfinite(z['X']).all(), 'X has no NaN or inf')
        chk(list(z['channels']) == ['hr', 'light_log1p', 'screen', 'activity_still',
                                    'step_log1p', 'ambience_silence'], '6 channels in the documented order')
        if feats is not None:
            same = (list(z['subject_id']) == list(feats.subject_id.astype(str))
                    and list(pd.to_datetime(z['lifelog_date'])) == list(pd.to_datetime(feats.lifelog_date)))
            chk(same, 'row order identical to features.parquet (the daylevel/sleep_tensor contract)')
        cov = z['mask'].mean(axis=(0, 2))
        for c, v in zip(z['channels'], cov):
            print(f'        coverage  {str(c):18s} {v * 100:5.1f}%')
        print(f'        coverage  {"overall":18s} {z["mask"].mean() * 100:5.1f}%')

    print('\ncontext_tensor.npz')
    p = CACHE / 'context_tensor.npz'
    if not p.exists():
        skip('not present — build it with: python preprocess/context.py')
    else:
        z = np.load(p, allow_pickle=True)
        chk(z['X'].shape == (N_ROWS, 4, 108), f"X (700, 4, 108) (got {z['X'].shape})")
        chk(z['X'].dtype == np.float32, 'dtype float32')
        chk(np.isfinite(z['X']).all(), 'X has no NaN or inf')
        chk(list(z['channels']) == ['charging', 'wifi', 'ble', 'usage'], '4 channels in the documented order')
        if feats is not None:
            same = (list(z['subject_id']) == list(feats.subject_id.astype(str))
                    and list(pd.to_datetime(z['lifelog_date'])) == list(pd.to_datetime(feats.lifelog_date)))
            chk(same, 'row order identical to features.parquet')
        nz = (z['X'] != 0).mean(axis=(0, 2))
        for c, v in zip(z['channels'], nz):
            print(f'        non-zero  {str(c):18s} {v * 100:5.1f}%')

    print('\nview caches (frozen per-view predictions)')
    for name, keys in [('view_tabular.npz', ('oof', 'sub')),
                       ('view_sleep_cnn.npz', ('pyr_o', 'pyr_s')),
                       ('view_context.npz', ('oof', 'sub'))]:
        p = CACHE / name
        if not p.exists():
            skip(f'{name} not present')
            continue
        z = np.load(p, allow_pickle=True)
        ko, ks = keys
        ok = ko in z.files and ks in z.files
        if not chk(ok, f'{name} has keys {ko} / {ks}'):
            continue
        chk(z[ko].shape == (N_TRAIN, 7) and z[ks].shape == (N_SUB, 7),
            f'{name} oof (450, 7) / sub (250, 7)')
        chk(((z[ko] > 0) & (z[ko] < 1)).all() and ((z[ks] > 0) & (z[ks] < 1)).all(),
            f'{name} probabilities strictly inside (0, 1)')

    print('\nsubmission')
    sub = ROOT / 'submit' / 'ch2026_submission.csv'
    ref = ROOT / 'reference' / 'ch2026_submission.csv'
    if not sub.exists():
        skip('submit/ch2026_submission.csv not present — run: python run.py')
    else:
        md5 = hashlib.md5(sub.read_bytes()).hexdigest()[:8]
        if not chk(md5 == EXPECT_MD5, f'md5 {EXPECT_MD5} (got {md5})') and ref.exists():
            a = pd.read_csv(ref).set_index(['subject_id', 'sleep_date'])[LAB]
            b = pd.read_csv(sub).set_index(['subject_id', 'sleep_date'])[LAB].reindex(a.index)
            d = np.abs(a.to_numpy() - b.to_numpy())
            print(f'        vs reference: max |delta| {d.max():.2e} over {d.size} cells')

    n_fail, n_skip = _state['fail'], _state['skip']
    print('\n' + (f'all checks passed ({n_skip} skipped)' if n_fail == 0
                  else f'{n_fail} check(s) FAILED ({n_skip} skipped)'))
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
