"""fuse.py — fuse the four views and write submit/ch2026_submission.csv
(md5 3b0f2da7, public LB 0.5543165607, private 0.56758198).

Steps:
  1. rebuild cache/timing_events.parquet from /data via preprocess/events.py:build_complete_timing()
  2. train the event_timing view on it as BG(LGBM, elastic-net)
  3. load the tabular, sleep_cnn and context views from cache/view_*.npz
  4. combine them with the deployed EDBF tree and write the submission CSV

    final = GlobalBG-LW( Platt( BG(tabular, BG(sleep_cnn, event_timing)) ), context )

The sleep-tensor MiniRocket co-view is pruned (USE_MR=False). USE_MR=True reproduces the pre-pruning
build (public LB 0.5563241004) and needs the MiniRocket scales from
`python views/build_sleep_cnn.py --with-mr`.

Run: python views/fuse.py   (or: python run.py)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import os, sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, hashlib
from sklearn.model_selection import KFold
from modeling.features.tabular import load_features, PanelFXConfig, run_oof_per_fold, OOFRunner
from modeling.config import DEFAULTS as CFG, DATA_DIR, CACHE_DIR as CACHE, SUBMIT_DIR as SUBMIT
from modeling.blend.fusion import deployed_pipeline
from preprocess.events import build_complete_timing, OUT_PARQUET as TIMING_PARQUET
EPS = 1e-12; LAB = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']; clip = lambda p: np.clip(p, EPS, 1 - EPS)
OUT_NAME = 'ch2026_submission.csv'; EXPECT_MD5 = '3b0f2da7'; USE_MR = False


def event_timing_predictor(n, backend):
    cfg = PanelFXConfig.default_derived(); cfg.seeds_oof = (42, 43, 44, 45, 46); cfg.model_backend = backend
    if backend == 'linear': cfg.seeds_oof, cfg.seeds_sub = (42,), (42,)
    data = load_features(DATA_DIR, labels=LAB)
    trd = data['train_df'].reset_index(drop=True); trd['lifelog_date'] = pd.to_datetime(trd['lifelog_date'])
    sbd = data['df'][data['df'].split == 'sub'].reset_index(drop=True); sbd['lifelog_date'] = pd.to_datetime(sbd['lifelog_date'])
    tm = pd.read_parquet(TIMING_PARQUET); tm['lifelog_date'] = pd.to_datetime(tm['lifelog_date'])
    key = ['subject_id', 'lifelog_date']; tc = [c for c in tm.columns if c not in set(key) | {'sleep_date', 'split'} | set(LAB)]
    trd = trd.merge(tm[key + tc], on=key, how='left'); sbd = sbd.merge(tm[key + tc], on=key, how='left')
    o = run_oof_per_fold(trd, tc, np.zeros((n, 7), np.float32), cfg, LAB, verbose=False)
    s = OOFRunner(cfg, enrich_feats=[]).predict_sub(trd, sbd, base_cols=tc, subday_cols=[], labels=LAB, verbose=False).astype(float)
    return o, s, sbd


def reference_check(out: pd.DataFrame) -> str:
    """Fallback for the md5 check when the run is not on the reference platform.

    A different OS/compiler builds LightGBM, BLAS and libm differently, so the last bits can move
    even when every step is identical. Byte-identity is only claimed on the platform in
    requirements.txt; elsewhere this reports the gap against reference/ch2026_submission.csv and the
    bound it puts on macro log-loss, whose sensitivity to p is at most 1/min(p, 1-p)."""
    ref_path = Path(__file__).resolve().parents[1] / 'reference' / OUT_NAME
    if not ref_path.exists():
        return f'  reference/{OUT_NAME} is missing -- cannot compare numerically'
    key = ['subject_id', 'sleep_date']
    a = pd.read_csv(ref_path).set_index(key)[LAB]
    b = out.set_index(key)[LAB].reindex(a.index)
    if b.isna().any().any():
        return '  row keys differ from the reference -- not a floating-point issue, investigate'
    p = a.to_numpy(); d = np.abs(p - b.to_numpy())
    bound = float((d / np.minimum(p, 1 - p)).mean())
    verdict = 'score-identical' if bound < 1e-9 else 'DIFFERENT -- investigate'
    return (f'  vs reference: max |delta| {d.max():.2e} over {d.size} cells; macro log-loss moves by at most '
            f'{bound:.2e} (the leaderboard prints 1e-8) -> {verdict}')


def main():
    build_complete_timing()
    data = load_features(DATA_DIR, labels=LAB)
    trd = data['train_df'].reset_index(drop=True); trd['lifelog_date'] = pd.to_datetime(trd['lifelog_date']).dt.normalize()
    y = trd[LAB].to_numpy().astype(float); n = len(y)
    folds = list(KFold(CFG.n_splits, shuffle=True, random_state=CFG.fold_seed).split(np.arange(n)))
    scnn = np.load(CACHE / 'view_sleep_cnn.npz'); scnn_o, scnn_s = scnn['pyr_o'].astype(float), scnn['pyr_s'].astype(float)
    if USE_MR:
        assert 'mr5o' in scnn, 'USE_MR=True needs the MiniRocket scales -- rebuild with: python views/build_sleep_cnn.py --with-mr'
        mr_o = clip(np.mean([scnn['mr5o'], scnn['mr10o'], scnn['mr15o']], 0)); mr_s = clip(np.mean([scnn['mr5s'], scnn['mr10s'], scnn['mr15s']], 0))
    else:
        mr_o = mr_s = None
    cm = np.load(CACHE / 'view_context.npz'); ctx_o, ctx_s = cm['oof'].astype(float), cm['sub'].astype(float)
    tab = np.load(CACHE / 'view_tabular.npz', allow_pickle=True); tab_o, tab_s = tab['oof'].astype(float), tab['sub'].astype(float)
    print('recompute timing predictors (lgbm + linear)...', flush=True)
    evt_lgbm_o, evt_lgbm_s, sbd = event_timing_predictor(n, 'lgbm'); evt_lin_o, evt_lin_s, _ = event_timing_predictor(n, 'linear')
    _, sub = deployed_pipeline(tab_o, tab_s, scnn_o, scnn_s, mr_o, mr_s, evt_lgbm_o, evt_lgbm_s, evt_lin_o, evt_lin_s, ctx_o, ctx_s, y, folds, use_mr=USE_MR)
    samp = pd.read_csv(DATA_DIR / 'ch2026_submission_sample.csv')
    out = sbd[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    out['sleep_date'] = pd.to_datetime(out['sleep_date']).dt.strftime('%Y-%m-%d'); out['lifelog_date'] = pd.to_datetime(out['lifelog_date']).dt.strftime('%Y-%m-%d')
    for l, L in enumerate(LAB): out[L] = sub[:, l]
    out = out[list(samp.columns)].set_index(['subject_id', 'sleep_date']).reindex(samp.set_index(['subject_id', 'sleep_date']).index).reset_index()
    assert out[LAB].isna().sum().sum() == 0
    SUBMIT.mkdir(parents=True, exist_ok=True)
    dest = SUBMIT / OUT_NAME; out.to_csv(dest, index=False, lineterminator='\n'); md5 = hashlib.md5(dest.read_bytes()).hexdigest()[:8]
    if Path('/data').is_dir() and os.access('/data', os.W_OK) and (SUBMIT.resolve() != Path('/data').resolve()):
        try:
            out.to_csv(Path('/data') / OUT_NAME, index=False, lineterminator='\n')
        except Exception:
            pass
    head = f'wrote {dest.name}  {len(out)} rows  md5 {md5}'
    if md5 == EXPECT_MD5:
        print(f'{head}  byte-identical OK')
    elif os.environ.get('ETRI_FROM_RAW') == '1':
        # reference_check is for the same-inputs-different-OS case; a from-raw retrain is not that.
        print(f'{head}  (from-raw rebuild, not byte-identical: the sleep_cnn view is retrained and the\n'
              f'  GPU draw differs per machine. For byte-exact restoration run `python run.py`.)')
    else:
        print(f'{head}  (expected {EXPECT_MD5})')
        print(reference_check(out))


if __name__ == '__main__':
    main()
