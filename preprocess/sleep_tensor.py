"""sleep_tensor.py — the 6-channel nocturnal sleep tensor.

Input   <data>/ch2025_data_items/{wHr,mLight,mScreenStatus,mActivity,wPedo,mAmbience}.parquet
        <data>/ch2026_metrics_train.csv, ch2026_submission_sample.csv
Output  cache/sleep_window_tensor.npz
          X    (700, 6, 108) float32   sleep_date 00:00-09:00, 5-min bins
          mask (700, 6, 108) int8      1 where the bin was observed
          channels = ['hr','light_log1p','screen','activity_still','step_log1p','ambience_silence']
          subject_id / lifelog_date / sleep_date / split
Used by views/build_sleep_cnn.py, modeling/stages/tabular_model.py

Rows are all_keys sorted by (subject_id, lifelog_date), matching daylevel.py.

Run: python preprocess/sleep_tensor.py     (PREPROCESS_VERBOSE=1 for the coverage tables)
"""

from pathlib import Path
import os, sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modeling.config import DATA_DIR

ITEMS_DIR = DATA_DIR / 'ch2025_data_items'
OUT_DIR   = ROOT / 'cache'
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_SUFFIX = os.environ.get('RAW_OUT_SUFFIX', '')

# PREPROCESS_VERBOSE=1 enables the diagnostic tables; `log` always prints.
VERBOSE = os.environ.get('PREPROCESS_VERBOSE', '0') == '1'
log = print


def print(*args, **kwargs):
    if VERBOSE:
        log(*args, **kwargs)


assert DATA_DIR.exists()
assert ITEMS_DIR.exists()

KEY = ['subject_id', 'lifelog_date']

# Row keys, as in daylevel.py: this order is the tensor's row order.
train = pd.read_csv(DATA_DIR/'ch2026_metrics_train.csv', parse_dates=['sleep_date','lifelog_date'])
sub   = pd.read_csv(DATA_DIR/'ch2026_submission_sample.csv', parse_dates=['sleep_date','lifelog_date'])

train['split'] = 'train'
sub['split']   = 'sub'

all_keys = pd.concat([
    train[['subject_id','sleep_date','lifelog_date','split']],
    sub[['subject_id','sleep_date','lifelog_date','split']],
], ignore_index=True).sort_values(['subject_id','lifelog_date']).reset_index(drop=True)

assert not all_keys[['subject_id','lifelog_date']].duplicated().any(), '키 중복!'

whr_raw    = pd.read_parquet(ITEMS_DIR/'ch2025_wHr.parquet')
mlight_raw = pd.read_parquet(ITEMS_DIR/'ch2025_mLight.parquet')
mscr_raw   = pd.read_parquet(ITEMS_DIR/'ch2025_mScreenStatus.parquet')
mact_raw   = pd.read_parquet(ITEMS_DIR/'ch2025_mActivity.parquet')
wpedo_raw  = pd.read_parquet(ITEMS_DIR/'ch2025_wPedo.parquet')
amb_raw    = pd.read_parquet(ITEMS_DIR/'ch2025_mAmbience.parquet')


SLEEP_HOURS = 9
BIN_MIN     = 5
N_BINS      = SLEEP_HOURS * 60 // BIN_MIN
CHANNELS    = ['hr', 'light_log1p', 'screen', 'activity_still', 'step_log1p', 'ambience_silence']
HR_MIN, HR_MAX = 30, 200

def bin_sleep_window(raw_df, value_col, agg='mean'):
    """sensor raw → groupby (sid, lifelog_date, bin_idx)[value_col].agg.

    sleep window = sleep_date 00:00~09:00 = label row의 (lifelog_date+1) 새벽.
    """
    df = raw_df[['subject_id', 'timestamp', value_col]].copy()
    df = df[df.timestamp.dt.hour < SLEEP_HOURS]
    df['sleep_date']   = df.timestamp.dt.normalize()
    df['lifelog_date'] = df.sleep_date - pd.Timedelta(days=1)
    df['bin_idx']      = ((df.timestamp.dt.hour * 60 + df.timestamp.dt.minute) // BIN_MIN).astype(np.int16)
    df = df.merge(all_keys[KEY], on=KEY, how='inner')
    return df.groupby(KEY + ['bin_idx'])[value_col].agg(agg)

def to_dense(agg_series):
    """groupby Series → (N_ROWS, N_BINS) dense + mask, all_keys 순서."""
    wide = agg_series.unstack('bin_idx')
    for i in range(N_BINS):
        if i not in wide.columns:
            wide[i] = np.nan
    wide = wide[list(range(N_BINS))]
    wide = wide.reindex(all_keys.set_index(KEY).index)
    mask = (~wide.isna()).astype(np.int8).values
    arr  = wide.fillna(0.0).astype(np.float32).values
    return arr, mask

import time as _t
t0 = _t.time()

if 'hr_mean_per_min' not in whr_raw.columns:
    whr_raw['hr_mean_per_min'] = whr_raw.heart_rate.map(np.mean)
whr_clean = whr_raw[whr_raw.hr_mean_per_min.between(HR_MIN, HR_MAX)]
print(f'HR filter: {len(whr_clean):,} / {len(whr_raw):,} minutes kept')
hr_arr, hr_mask = to_dense(bin_sleep_window(whr_clean, 'hr_mean_per_min', 'mean'))

ml_arr, ml_mask = to_dense(bin_sleep_window(mlight_raw, 'm_light', 'mean'))
ml_arr = np.log1p(np.maximum(ml_arr, 0.0)).astype(np.float32)

sc_arr, sc_mask = to_dense(bin_sleep_window(mscr_raw, 'm_screen_use', 'mean'))

ma = mact_raw[['subject_id', 'timestamp', 'm_activity']].copy()
ma['is_still'] = (ma.m_activity == 3).astype(np.int8)
st_arr, st_mask = to_dense(bin_sleep_window(ma, 'is_still', 'mean'))

step_arr, step_mask = to_dense(bin_sleep_window(wpedo_raw, 'step', 'sum'))
step_arr = np.log1p(np.maximum(step_arr, 0.0)).astype(np.float32)

amb_for_tensor = amb_raw[['subject_id', 'timestamp', 'm_ambience']].copy()
amb_for_tensor['is_silence'] = amb_for_tensor.m_ambience.map(
    lambda x: 1.0 if (len(x) > 0 and str(x[0][0]) == 'Silence') else 0.0
).astype(np.float32)
am_arr, am_mask = to_dense(bin_sleep_window(amb_for_tensor, 'is_silence', 'mean'))

X    = np.stack([hr_arr, ml_arr, sc_arr, st_arr, step_arr, am_arr], axis=1).astype(np.float32)
mask = np.stack([hr_mask, ml_mask, sc_mask, st_mask, step_mask, am_mask], axis=1).astype(np.int8)

print(f'tensor build: {_t.time()-t0:.1f}s')
print(f'X    shape={X.shape}    dtype={X.dtype}')
print(f'mask shape={mask.shape} dtype={mask.dtype}')

SLEEP_TENSOR_PATH = OUT_DIR / f'sleep_window_tensor{RAW_SUFFIX}.npz'
np.savez_compressed(
    SLEEP_TENSOR_PATH,
    X=X, mask=mask,
    subject_id   = all_keys.subject_id.values.astype(str),
    lifelog_date = pd.to_datetime(all_keys.lifelog_date).dt.strftime('%Y-%m-%d').values,
    sleep_date   = pd.to_datetime(all_keys.sleep_date).dt.strftime('%Y-%m-%d').values,
    split        = all_keys.split.values.astype(str),
    channels     = np.array(CHANNELS),
)
log(f'saved {SLEEP_TENSOR_PATH}  X={X.shape} mask={mask.shape}')

chk = np.load(SLEEP_TENSOR_PATH, allow_pickle=False)
assert chk['X'].shape == X.shape and chk['mask'].shape == mask.shape
print('reload OK')

print('\n=== Channel별 row coverage (per-row mask sum / 108) ===')
print(f'  {"channel":>18}  {"mean":>6} {"p50":>6} {"p95":>6} {"empty":>10}')
for c, name in enumerate(CHANNELS):
    cov = mask[:, c].mean(1)
    n_empty = int((cov == 0).sum())
    print(f'  {name:>18}  {cov.mean():>6.3f} {np.median(cov):>6.3f} {np.quantile(cov, 0.95):>6.3f} {n_empty:>4}/{len(cov)}')

print('\n=== Channel별 값 분포 (mask=1 bin만) ===')
print(f'  {"channel":>18}  {"n_bins":>8} {"mean":>8} {"std":>8} {"min":>8} {"max":>8}')
for c, name in enumerate(CHANNELS):
    has = mask[:, c].astype(bool)
    v = X[:, c][has]
    print(f'  {name:>18}  {has.sum():>8} {v.mean():>8.3f} {v.std():>8.3f} {v.min():>8.3f} {v.max():>8.3f}')

print('\n=== train vs sub coverage 비교 ===')
is_train = (all_keys.split == 'train').values
is_sub   = (all_keys.split == 'sub').values
for c, name in enumerate(CHANNELS):
    tr = mask[is_train, c].mean()
    sb = mask[is_sub, c].mean()
    print(f'  {name:>18}  train={tr:.3f}  sub={sb:.3f}  diff={tr-sb:+.3f}')

print('\n--- 사용법 ---')
print("d = np.load('cache/sleep_window_tensor.npz', allow_pickle=False)")
print("X    = d['X']            # (700, 6, 108) float32")
print("mask = d['mask']         # (700, 6, 108) int8")

