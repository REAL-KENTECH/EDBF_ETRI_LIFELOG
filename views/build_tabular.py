"""build_tabular.py — the tabular view.

Writes cache/view_tabular.npz (keys oof, sub): the features-only base, BG(LGBM, elastic-net) followed
by k-NN borrowing and a temporal prior. Runs modeling/stages/tabular_model.py with the deployed
configuration (TABULAR_LEARNER=bg, RANDKNN=1, RANDKNN_K=2) and repacks its output under the per-view
name.

Requires cache/features.parquet and cache/sleep_window_tensor.npz (preprocess/daylevel.py and
preprocess/sleep_tensor.py). The stage also computes the CNN cache in the same pass; those predictions
do not enter this view. CPU-deterministic (LGBM and elastic-net, fixed seeds). An existing
cache/view_tabular.npz is reused.

Run: python views/build_tabular.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import os, sys, runpy; from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'cache'
OUT = CACHE / 'view_tabular.npz'
RAW = CACHE / 'tabular_oof_bg.npz'   # base_model's native output name for TABULAR_LEARNER=bg


def main():
    if OUT.exists():
        print(f'{OUT.name} already present — REUSING (delete to rebuild).'); return
    os.environ.update(TABULAR_LEARNER='bg', RANDKNN='1', RANDKNN_K='2')
    sys.path.insert(0, str(ROOT))
    print('building tabular view via modeling/stages/tabular_model.py (TABULAR_LEARNER=bg, randKNN k=2)...', flush=True)
    runpy.run_path(str(ROOT / 'modeling' / 'stages' / 'tabular_model.py'), run_name='__main__')
    z = np.load(RAW, allow_pickle=True)
    np.savez(OUT, oof=z['oof'], sub=z['sub'])
    RAW.unlink()   # drop the intermediate name; keep only the per-view file
    print(f'wrote {OUT.name}')


if __name__ == '__main__':
    main()
