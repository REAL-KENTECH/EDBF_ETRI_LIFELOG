"""events.py — nocturnal event-timing features.

Input   cache/timing_features.parquet   (preprocess/timing.py; required)
        <data>/ch2025_data_items/{mScreenStatus,mActivity,mLight,mAmbience,wHr}.parquet
Output  cache/timing_events.parquet   700 x (keys + sleep_date + split + 7 labels + 52 features
                                      = 19 rhythm columns carried over + 33 event-timing)
Used by views/fuse.py, which trains the event_timing view on it.

Over the 21:00-09:00 window: event counts, on-minutes, first/last/longest-gap and early/mid timing
for screen, movement, lux, ambience and HR-presence.

views/fuse.py feeds every non-key, non-label column to the model; adding a column changes the model.

Run: python preprocess/events.py     (views/fuse.py also calls build_complete_timing() directly)
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

RAW = DATA_DIR / 'ch2025_data_items'
CACHE = ROOT / 'cache'
BASE_TIMING = CACHE / 'timing_features.parquet'
OUT_PARQUET = CACHE / 'timing_events.parquet'

# Generic event streams (lux / ambience / HR-presence). Screen and movement use the dedicated extractors below.
GENERIC_STREAMS = [
    ('ch2025_mLight.parquet',    lambda d: d['m_light'] >= 10,                                                    'lux'),
    ('ch2025_mAmbience.parquet', lambda d: d['m_ambience'].map(lambda a: a[0][0] if len(a) else '') != 'Silence', 'amb'),
    ('ch2025_wHr.parquet',       lambda d: pd.Series(1, index=d.index),                                           'hrp'),
]


def _night(df):
    """assign the 21:00-09:00 night window + minute-of-window mw in [0,720)."""
    df = df.copy(); df['timestamp'] = pd.to_datetime(df['timestamp']); h = df['timestamp'].dt.hour; ll = df['timestamp'].dt.normalize()
    nd = pd.Series(pd.NaT, index=df.index); nd[h >= 21] = ll[h >= 21]; nd[h < 9] = ll[h < 9] - pd.Timedelta(days=1)
    df = df.assign(lifelog_date=nd).dropna(subset=['lifelog_date'])
    df['mw'] = ((df['timestamp'] - (df['lifelog_date'] + pd.Timedelta(hours=21))).dt.total_seconds() // 60).astype(int)
    return df[(df.mw >= 0) & (df.mw < 720)].sort_values(['subject_id', 'timestamp'])


def screen_events():
    """Screen event-timing over the night window."""
    d = _night(pd.read_parquet(RAW / 'ch2025_mScreenStatus.parquet')); d['on'] = (d['m_screen_use'] > 0).astype(int); col = 'm_screen_use'
    def per(x):
        on = x['on'].values; mw = x['mw'].values; trans = int(((on[1:] == 1) & (on[:-1] == 0)).sum()) + int(on[0] == 1)
        gaps = []; run = 0
        for v in on:
            if v == 0: run += 1
            else: gaps.append(run); run = 0
        gaps.append(run)
        return pd.Series({f'{col}_n_events': trans, f'{col}_on_min': int(on.sum()), f'{col}_first': mw[on == 1].min() if on.any() else 720.0,
                          f'{col}_last': mw[on == 1].max() if on.any() else 0.0, f'{col}_maxgap': max(gaps) if gaps else 720,
                          f'{col}_early': int(((mw < 180) & (on == 1)).sum()), f'{col}_mid': int(((mw >= 180) & (mw < 540) & (on == 1)).sum())})
    return d.groupby(['subject_id', 'lifelog_date']).apply(per).reset_index()


def movement_events():
    """mActivity event-timing (act_* naming, matches d8a8fe35)."""
    ac = _night(pd.read_parquet(RAW / 'ch2025_mActivity.parquet')); ac['act'] = (~ac.m_activity.isin([3, 4])).astype(int)
    def per(x):
        a = x.act.values; mw = x.mw.values; tr = int(((a[1:] == 1) & (a[:-1] == 0)).sum()) + int(a[0] == 1)
        gaps = []; run = 0
        for v in a:
            if v == 0: run += 1
            else: gaps.append(run); run = 0
        gaps.append(run)
        return pd.Series({'act_n_events': tr, 'act_active_min': int(a.sum()), 'act_first': mw[a == 1].min() if a.any() else 720.0,
                          'act_last': mw[a == 1].max() if a.any() else 0.0, 'act_still_maxgap': max(gaps) if gaps else 720})
    return ac.groupby(['subject_id', 'lifelog_date']).apply(per).reset_index()


def night_events(parquet, on_fn, prefix):
    """generic event-timing ({prefix}_n / _on_min / _first / _last / _maxgap / _early / _mid)."""
    d = _night(pd.read_parquet(RAW / parquet)); d['on'] = on_fn(d).astype(int)
    def per(x):
        on = x['on'].values; mw = x['mw'].values; tr = int(((on[1:] == 1) & (on[:-1] == 0)).sum()) + int(on[0] == 1)
        gaps = []; run = 0
        for v in on:
            if v == 0: run += 1
            else: gaps.append(run); run = 0
        gaps.append(run)
        return pd.Series({f'{prefix}_n': tr, f'{prefix}_on_min': int(on.sum()), f'{prefix}_first': mw[on == 1].min() if on.any() else 720.0,
                          f'{prefix}_last': mw[on == 1].max() if on.any() else 0.0, f'{prefix}_maxgap': max(gaps) if gaps else 720,
                          f'{prefix}_early': int(((mw < 180) & (on == 1)).sum()), f'{prefix}_mid': int(((mw >= 180) & (mw < 540) & (on == 1)).sum())})
    return d.groupby(['subject_id', 'lifelog_date']).apply(per).reset_index()


def build_complete_timing(out=OUT_PARQUET):
    """ONE pass: base rhythm + screen + movement + lux + ambience + HR-presence event-timing -> out parquet."""
    if not BASE_TIMING.exists():
        raise SystemExit(f'{BASE_TIMING.name} not found under {CACHE}. Build it first:  python preprocess/timing.py')
    tim = pd.read_parquet(BASE_TIMING); tim['lifelog_date'] = pd.to_datetime(tim['lifelog_date'])
    parts = [screen_events(), movement_events()] + [night_events(p, fn, pre) for p, fn, pre in GENERIC_STREAMS]
    added = []
    for ev in parts:
        ev['lifelog_date'] = pd.to_datetime(ev['lifelog_date']); tim = tim.merge(ev, on=['subject_id', 'lifelog_date'], how='left')
        added += [c for c in ev.columns if c not in ('subject_id', 'lifelog_date')]
    # Column median over train AND sub rows. No label is involved, so this is an unsupervised
    # statistic, but it is transductive; kept exactly as-is because it is baked into the shipped
    # submission (md5 3b0f2da7) and a train-only median would move every event_timing prediction.
    tim[added] = tim[added].fillna(tim[added].median()).fillna(0.0); tim.to_parquet(out)
    print(f'built complete timing: base rhythm + {len(added)} event cols (screen+move+lux+amb+hrp) -> {Path(out).name}')
    return out


if __name__ == '__main__':
    build_complete_timing()
