#!/usr/bin/env python
"""run.py — reproduce submit/ch2026_submission.csv (md5 3b0f2da7, public LB 0.5543165607, private 0.56758198).

  python run.py              Fuse the shipped per-view caches. Rebuilds the event-timing features from
                             /data and retrains the event_timing view. CPU, ~2 min, no torch.
                             Writes a byte-identical CSV and prints "byte-identical OK".
  python run.py --from-raw   Rebuild every view from raw, then fuse. Requires a GPU and torch.
                             Not byte-identical: the CNN draw differs per machine.

Raw competition data (ch2025_data_items/*.parquet + ch2026_*.csv) is read from /data, or from
$ETRI_DATA. Output goes to submit/, or to $ETRI_OUT. See README.md and requirements.txt.
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIEW_CACHES = ['cache/view_tabular.npz', 'cache/view_sleep_cnn.npz', 'cache/view_context.npz']
# events.py is absent because views/fuse.py calls it directly on every run.
FROM_RAW_STEPS = [
    'preprocess/daylevel.py',     # -> cache/features.parquet          (281 day-level features)
    'preprocess/sleep_tensor.py', # -> cache/sleep_window_tensor.npz   (6-ch 00-09h sleep tensor)
    'preprocess/timing.py',       # -> cache/timing_features.parquet   (19 sleep-rhythm columns)
    'preprocess/context.py',      # -> cache/context_tensor.npz        (4-ch 00-09h context tensor)
    'views/build_tabular.py',     # -> cache/view_tabular.npz      (tabular base = BG(LGBM, elastic-net))
    'views/build_sleep_cnn.py',   # -> cache/view_sleep_cnn.npz    (pyramid CNN; --with-mr for the pruned MR view)
    'views/build_context.py',     # -> cache/view_context.npz      (nocturnal-context MiniRocket)
    'views/fuse.py',              # -> submit/ch2026_submission.csv
]


def _run(rel: str) -> None:
    rc = subprocess.run([sys.executable, str(HERE / rel)], cwd=str(HERE)).returncode
    if rc != 0:
        sys.exit(f'step failed: {rel} (exit {rc})')


def _data_dir() -> Path:
    """Same resolution order as modeling/config.py: $ETRI_DATA, then /data, then <submission>/data, then <repo>/data."""
    if os.environ.get('ETRI_DATA'):
        return Path(os.environ['ETRI_DATA']).resolve()
    if Path('/data/ch2026_submission_sample.csv').exists() or Path('/data/ch2025_data_items').exists():
        return Path('/data')
    if (HERE / 'data' / 'ch2026_submission_sample.csv').exists() or (HERE / 'data' / 'ch2025_data_items').exists():
        return (HERE / 'data').resolve()
    return (HERE.parent / 'data').resolve()


def _preflight(from_raw: bool) -> None:
    """Fail fast, with an actionable message, before any long step runs."""
    if sys.version_info < (3, 10):
        sys.exit(f'this package needs Python 3.10+ (running {sys.version.split()[0]}).\n'
                 'Not our choice: numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2 and pyarrow 25.0.1 are all\n'
                 'published as >=3.10 only, so `pip install -r requirements.txt` cannot succeed here.\n'
                 'Verified on 3.10.20 and 3.13.13:  conda create -n etri python=3.10 && pip install -r requirements.txt')
    if sys.version_info >= (3, 14):
        print(f'WARNING: Python {sys.version.split()[0]}; verified on 3.10.20 and 3.13.13.\n'
              '  The pinned numpy/scipy have no 3.14 wheels. If you installed newer ones instead, this is\n'
              '  a different numeric environment and the md5 may differ; the run then reports the gap\n'
              '  against reference/ch2026_submission.csv, which is what decides the score.\n', flush=True)
    data = _data_dir()
    missing = [n for n in ('ch2025_data_items', 'ch2026_metrics_train.csv', 'ch2026_submission_sample.csv')
               if not (data / n).exists()]
    if missing:
        sys.exit(f'raw competition data not found under {data} (missing: {", ".join(missing)}).\n'
                 f'Mount it at /data, or set ETRI_DATA=/path/to/data.')
    if from_raw:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit('--from-raw retrains the CNN and needs torch, which requirements.txt leaves commented out.\n'
                     'Install it with:  pip install "torch>=2.1"\n'
                     'Score restoration does not need torch: run `python run.py`.')
    print(f'data: {data}', flush=True)


def main() -> None:
    from_raw = '--from-raw' in sys.argv[1:]
    _preflight(from_raw)
    if from_raw:
        print('=== [from-raw] rebuild all views from raw, then fuse (GPU required for the CNN view) ===', flush=True)
        os.environ['ETRI_FROM_RAW'] = '1'   # fuse.py: a differing md5 is expected here
        for c in VIEW_CACHES:
            (HERE / c).unlink(missing_ok=True)        # force view rebuild
        for step in FROM_RAW_STEPS:
            _run(step)
    else:
        print('=== rebuild the event-timing view from /data, then fuse the shipped view caches ===', flush=True)
        _run('views/fuse.py')


if __name__ == '__main__':
    main()
