"""timing.py — sleep-rhythm features.

Input   <data>/ch2025_data_items/{mActivity,mScreenStatus,mLight,mACStatus,wHr}.parquet
        <data>/ch2026_metrics_train.csv, ch2026_submission_sample.csv
Output  cache/timing_features.parquet   700 x (2 keys + sleep_date + split + 7 labels + 19 features)
Used by preprocess/events.py only; no model reads this file directly.

  sleep episode  multi-signal asleep score (still / dark / no-screen / low-HR) -> bedtime, wake, TIB
  sleep window   HR, light, activity and screen inside the detected sleep period
  rhythm         per-subject regularity, this-night deviation, social jetlag (train rows only)

DET_THR / DET_SMOOTH / DET_MIN_RUN / DET_MERGE and the per-subject resting-HR quantile fix the episode
boundaries, hence all 19 columns. Frozen for the shipped submission.

Run: python preprocess/timing.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modeling.config import DATA_DIR

ITEMS = DATA_DIR / 'ch2025_data_items'
CACHE = ROOT / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
KEY = ['subject_id', 'lifelog_date']
LABELS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

EV_START_MIN, EV_END_MIN = 18 * 60, 36 * 60
ACT_STILL = 3


def load_keys() -> pd.DataFrame:
    """700-row (subject_id, lifelog_date, sleep_date, split) + labels, from the raw CSVs."""
    tr = pd.read_csv(DATA_DIR / 'ch2026_metrics_train.csv'); tr['split'] = 'train'
    sb = pd.read_csv(DATA_DIR / 'ch2026_submission_sample.csv'); sb['split'] = 'sub'
    df = pd.concat([tr, sb], ignore_index=True)
    df['lifelog_date'] = pd.to_datetime(df['lifelog_date'])
    return df.sort_values(KEY).reset_index(drop=True)


def load_hr() -> pd.DataFrame:
    hr = pd.read_parquet(ITEMS / 'ch2025_wHr.parquet')
    hr['hr'] = hr['heart_rate'].map(lambda a: float(np.mean(a)) if a is not None and len(a) else np.nan)
    return hr[['subject_id', 'timestamp', 'hr']].dropna(subset=['hr'])


DET_THR, DET_SMOOTH, DET_MIN_RUN, DET_MERGE = 0.75, 7, 30, 30
DET_LMIN = float(os.environ.get('DET_LMIN', 0))


def detect_episode(night: pd.DataFrame, rest_hr: float) -> tuple[float, float]:
    """night: 1-min grid for one (subject, night) window with columns hr/act/scr/light.
    Returns (bedtime_min, wake_min) from lifelog 00:00, or (nan, nan) if no qualifying sleep block.

    Per-minute asleep evidence from up to 4 signals (still / no-screen / dark / low-HR); the score is
    the FRACTION of AVAILABLE signals that say 'asleep' — so HR's 89% missingness does not force an
    over-strict all-phone-signals rule. A short smoothing removes single-minute flicker (the cause of
    the fragmented short episodes a hard rule produces); the longest smoothed run anchors the episode,
    and brief arousals (< DET_MERGE min) are merged in as WASO. (Charging is kept as a tensor channel
    but NOT used in the score — adding it over-extends the episode and lowers corr(tib,S1).)"""
    sig = pd.DataFrame({'still': night['act'] == ACT_STILL, 'no_scr': night['scr'] == 0,
                        'dark': night['light'] < 10, 'low_hr': night['hr'] < rest_hr + 5})
    avail = pd.DataFrame({'still': night['act'].notna(), 'no_scr': night['scr'].notna(),
                          'dark': night['light'].notna(), 'low_hr': night['hr'].notna()})
    score = ((sig & avail).sum(axis=1) / avail.sum(axis=1).clip(lower=1)).where(avail.any(axis=1), 0.0)
    score = score.rolling(DET_SMOOTH, center=True, min_periods=2).mean().fillna(0.0)
    asleep = (score >= DET_THR).to_numpy()
    runs, cur = [], None
    for i, v in enumerate(asleep):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if i - cur >= DET_MIN_RUN:
                runs.append((cur, i))
            cur = None
    if cur is not None and len(asleep) - cur >= DET_MIN_RUN:
        runs.append((cur, len(asleep)))
    if not runs:
        return np.nan, np.nan
    ai = max(range(len(runs)), key=lambda i: runs[i][1] - runs[i][0])
    start, end = runs[ai]
    j = ai + 1
    while j < len(runs) and runs[j][0] - end < DET_MERGE:
        end = runs[j][1]
        j += 1
    off = night['offset'].to_numpy()
    return float(off[start]), float(off[min(end, len(off) - 1)])


def build_nights(keys: pd.DataFrame, act, scr, light, hr, charge):
    """For every (subject, lifelog_date), build a 1-min grid over the sleep-search window and
    return (episode DataFrame, {(subj,date): grid}). Grids are reused for the tensor."""
    def idx(df, col):
        return {s: g.sort_values('timestamp').set_index('timestamp')[col]
                for s, g in df.groupby('subject_id', sort=False)}
    A, S, L, H, C = idx(act, 'm_activity'), idx(scr, 'm_screen_use'), idx(light, 'm_light'), \
        idx(hr, 'hr'), idx(charge, 'm_charging')
    rest = hr.groupby('subject_id')['hr'].quantile(0.10).to_dict()
    rows = []
    for r in keys.itertuples(index=False):
        sid, ld = r.subject_id, r.lifelog_date
        start = ld + pd.Timedelta(minutes=EV_START_MIN)
        mins = pd.date_range(start, ld + pd.Timedelta(minutes=EV_END_MIN - 1), freq='1min')
        g = pd.DataFrame({'ts': mins})
        g['offset'] = ((g['ts'] - ld).dt.total_seconds() / 60).round().astype(int)
        def col(d, limit):
            s = d.get(sid)
            if s is None or not len(s):
                return np.full(len(g), np.nan)
            s = s[~s.index.duplicated()]
            return s.reindex(g['ts'], method='ffill', limit=limit).to_numpy()
        g['act'], g['scr'] = col(A, 10), col(S, 10)
        g['light'], g['charge'] = col(L, 30), col(C, 120)
        g['hr'] = col(H, 10)
        b, w = detect_episode(g, rest.get(sid, 60.0))
        rows.append((sid, ld, b, w))
    ep = pd.DataFrame(rows, columns=['subject_id', 'lifelog_date', 'bedtime_min', 'wake_min'])
    if DET_LMIN > 0:
        short = (ep['wake_min'] - ep['bedtime_min']) < DET_LMIN
        ep.loc[short, ['bedtime_min', 'wake_min']] = np.nan
    ep['is_fallback'] = ep['bedtime_min'].isna()
    nf = ep[~ep['is_fallback']]
    med = nf.groupby('subject_id')[['bedtime_min', 'wake_min']].median()
    glob = nf[['bedtime_min', 'wake_min']].median()
    for c in ['bedtime_min', 'wake_min']:
        ep[c] = ep[c].fillna(ep['subject_id'].map(med[c])).fillna(glob[c])
    ep['tib_min'] = ep['wake_min'] - ep['bedtime_min']
    return ep


def sleep_rhythm(d: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    d = d.copy()
    d['midsleep_min'] = d['bedtime_min'] + d['tib_min'] / 2.0
    d['dow'] = pd.to_datetime(d['lifelog_date']).dt.dayofweek
    valid = train_mask & (~d['is_fallback'].to_numpy())
    free = d['dow'].isin([4, 5]).to_numpy()
    cols = ['bedtime_min', 'wake_min', 'midsleep_min', 'tib_min']
    subj = d['subject_id'].to_numpy()
    add = {}
    for c in cols:
        nm = c.split('_')[0]
        add[f'rhythm__{nm}_std'] = np.full(len(d), np.nan)
        add[f'rhythm__{nm}_dev'] = np.full(len(d), np.nan)
    add['rhythm__social_jetlag'] = np.full(len(d), np.nan)
    add['rhythm__fallback_frac'] = np.full(len(d), np.nan)
    for s in np.unique(subj):
        sm = subj == s
        vm = valid & sm
        for c in cols:
            nm = c.split('_')[0]
            add[f'rhythm__{nm}_std'][sm] = d.loc[vm, c].std() if vm.sum() >= 3 else np.nan
            mfit = d.loc[vm, c].median() if vm.sum() >= 1 else d.loc[valid, c].median()
            add[f'rhythm__{nm}_dev'][sm] = d.loc[sm, c].to_numpy() - mfit
        mf, mw = d.loc[vm & free, 'midsleep_min'], d.loc[vm & ~free, 'midsleep_min']
        add['rhythm__social_jetlag'][sm] = (mf.mean() - mw.mean()) if len(mf) and len(mw) else np.nan
        add['rhythm__fallback_frac'][sm] = d.loc[train_mask & sm, 'is_fallback'].mean()
    add['rhythm__bedtime_dev_abs'] = np.abs(add['rhythm__bedtime_dev'])
    add['rhythm__is_weekend'] = d['dow'].isin([5, 6]).astype(float).to_numpy()
    out = pd.DataFrame(add, index=d.index)
    for c in out.columns:
        if out[c].isna().any():
            out[c] = out[c].fillna(out.loc[train_mask, c].median())
    return out


def within_cols(d: pd.DataFrame, cols, train_mask: np.ndarray) -> pd.DataFrame:
    """__wi = value − per-subject TRAIN mean (leak-safe within deviation)."""
    g = d.loc[train_mask].groupby('subject_id')
    smean = g[cols].mean(); glob = d.loc[train_mask, cols].median()
    per = smean.reindex(d['subject_id']).set_index(d.index)
    return pd.DataFrame({f'{c}__wi': (d[c].fillna(per[c].fillna(glob[c]))
                                      - per[c].fillna(glob[c])).to_numpy() for c in cols},
                        index=d.index)


def main():
    print('=== model preprocess_timing — sleep_timing family from raw ===')
    keys = load_keys()
    train_mask = (keys['split'] == 'train').to_numpy()
    print(f'keys: {keys.shape}, train={train_mask.sum()}, sub={(~train_mask).sum()}')
    act = pd.read_parquet(ITEMS / 'ch2025_mActivity.parquet')
    scr = pd.read_parquet(ITEMS / 'ch2025_mScreenStatus.parquet')
    light = pd.read_parquet(ITEMS / 'ch2025_mLight.parquet')
    charge = pd.read_parquet(ITEMS / 'ch2025_mACStatus.parquet')
    hr = load_hr()
    print('  raw sensors loaded')
    ep = build_nights(keys, act, scr, light, hr, charge)
    print(f"  episode: tib med={ep.loc[~ep.is_fallback,'tib_min'].median():.0f} "
          f"fallback={ep.is_fallback.mean() * 100:.0f}%")
    base = keys.merge(ep[KEY + ['bedtime_min', 'wake_min', 'tib_min', 'is_fallback']], on=KEY, how='left')
    base['ep_fallback'] = base['is_fallback'].astype(float)
    rhy = sleep_rhythm(base, train_mask)
    wi = within_cols(base, ['tib_min', 'bedtime_min', 'wake_min'], train_mask)
    base = pd.concat([base.drop(columns=['is_fallback']), rhy, wi], axis=1)
    assert base[KEY].duplicated().sum() == 0 and len(base) == 700
    timing = [c for c in base.columns if c.startswith('rhythm__') or c.endswith('__wi')
              or c in ('tib_min', 'bedtime_min', 'wake_min', 'ep_fallback')]
    out = CACHE / 'timing_features.parquet'
    base[KEY + ['sleep_date', 'split'] + LABELS + timing].to_parquet(out)
    print(f'saved {out}  ({len(timing)} sleep_timing cols)')


if __name__ == '__main__':
    main()
