"""context.py — the 4-channel nocturnal context tensor.

Input   <data>/ch2025_data_items/{mACStatus,mWifi,mBle,mUsageStats}.parquet
        <data>/ch2026_metrics_train.csv, ch2026_submission_sample.csv
Output  cache/context_tensor.npz
          X    (700, 4, 108) float32   sleep_date 00:00-09:00, 5-min bins
          channels = ['charging','wifi','ble','usage']
          subject_id / lifelog_date / split
Used by views/build_context.py

charging is the bin mean of m_charging; wifi/ble/usage are record counts.

No mask: an unobserved bin is 0, and 0 is also a valid charging value. Adding one would change the
view, which was trained on that ambiguity.

Rows are all_keys sorted by (subject_id, lifelog_date), matching daylevel.py.

Run: python preprocess/context.py
"""
from __future__ import annotations
from pathlib import Path
import os, sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modeling.config import DATA_DIR

ITEMS_DIR = DATA_DIR / 'ch2025_data_items'
OUT_DIR = ROOT / 'cache'
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_SUFFIX = os.environ.get('RAW_OUT_SUFFIX', '')

NBIN = 108                                   # 00:00-09:00 in 5-minute bins
CHANNELS = ['charging', 'wifi', 'ble', 'usage']


def load_keys() -> pd.DataFrame:
    """Row keys, as in daylevel.py: this order is the tensor's row order."""
    train = pd.read_csv(DATA_DIR / 'ch2026_metrics_train.csv', parse_dates=['sleep_date', 'lifelog_date'])
    sub = pd.read_csv(DATA_DIR / 'ch2026_submission_sample.csv', parse_dates=['sleep_date', 'lifelog_date'])
    train['split'] = 'train'
    sub['split'] = 'sub'
    all_keys = pd.concat([
        train[['subject_id', 'sleep_date', 'lifelog_date', 'split']],
        sub[['subject_id', 'sleep_date', 'lifelog_date', 'split']],
    ], ignore_index=True).sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
    assert not all_keys[['subject_id', 'lifelog_date']].duplicated().any(), '키 중복!'
    all_keys['lifelog_date'] = all_keys['lifelog_date'].dt.normalize()
    return all_keys


def build_tensor(key_df: pd.DataFrame) -> np.ndarray:
    """Bin the four streams into (rows, 4, 108). Records at 00:00-09:00 belong to the night that
    follows lifelog_date, hence the -1 day."""
    kd = key_df.reset_index(drop=True)
    idx = {(r.subject_id, r.lifelog_date): i for i, r in kd.iterrows()}
    M = len(kd)

    def binn(df, fn):
        out = np.zeros((M, NBIN), np.float32)
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'].dt.hour < 9]
        df['lifelog_date'] = df['timestamp'].dt.normalize() - pd.Timedelta(days=1)
        df['bin'] = (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute) // 5
        for (s, ld, b), g in df.groupby(['subject_id', 'lifelog_date', 'bin']):
            i = idx.get((s, ld))
            if i is not None and 0 <= b < NBIN:
                out[i, b] = fn(g)
        return out

    ac = binn(pd.read_parquet(ITEMS_DIR / 'ch2025_mACStatus.parquet').rename(columns={'m_charging': 'v'}),
              lambda g: float(g['v'].mean()))
    wf = binn(pd.read_parquet(ITEMS_DIR / 'ch2025_mWifi.parquet'), lambda g: float(len(g)))
    bl = binn(pd.read_parquet(ITEMS_DIR / 'ch2025_mBle.parquet'), lambda g: float(len(g)))
    us = binn(pd.read_parquet(ITEMS_DIR / 'ch2025_mUsageStats.parquet'), lambda g: float(len(g)))
    return np.stack([ac, wf, bl, us], axis=1).astype(np.float32)


def main() -> None:
    keys = load_keys()
    print(f'building 4-channel nocturnal context tensor for {len(keys)} rows from {ITEMS_DIR} ...', flush=True)
    X = build_tensor(keys)
    out = OUT_DIR / f'context_tensor{RAW_SUFFIX}.npz'
    np.savez_compressed(
        out,
        X=X,
        subject_id=keys.subject_id.values.astype(str),
        lifelog_date=pd.to_datetime(keys.lifelog_date).dt.strftime('%Y-%m-%d').values.astype(str),
        split=keys.split.values.astype(str),
        channels=np.array(CHANNELS),
    )
    print(f'saved {out}  X={X.shape} {X.dtype}')
    nz = (X != 0).mean(axis=(0, 2))
    for c, v in zip(CHANNELS, nz):
        print(f'  non-zero bins  {c:10s} {v * 100:5.1f}%')


if __name__ == '__main__':
    main()
