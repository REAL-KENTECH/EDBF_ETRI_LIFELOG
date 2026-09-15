"""Single source of truth for the EDBF pipeline configuration.

Paths (raw data, cache, submission output) and the pipeline defaults read by the view builders
(`views/build_*.py` and `modeling/stages/tabular_model.py`).
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import FrozenSet

# ── Portable paths (no hard-coded absolute paths) ────────────────────────────────────────────────
# Layout: <REPO_ROOT>/{data/, model/}. Code lives under model/. Raw competition data under data/.
# Override the data location with the ETRI_DATA env var (e.g. the graders' /data mount).
MODEL_DIR = Path(__file__).resolve().parents[1]          # .../model
REPO_ROOT = MODEL_DIR.parent                             # contains model/ and data/


def _resolve_data_dir() -> Path:
    """Raw competition data location, in priority order:
       1) $ETRI_DATA   2) /data (the graders' standard mount)   3) <submission>/data or <repo>/data."""
    if os.environ.get('ETRI_DATA'):
        return Path(os.environ['ETRI_DATA']).resolve()
    if Path('/data/ch2026_submission_sample.csv').exists() or Path('/data/ch2025_data_items').exists():
        return Path('/data')
    if (MODEL_DIR / 'data' / 'ch2026_submission_sample.csv').exists() or (MODEL_DIR / 'data' / 'ch2025_data_items').exists():
        return (MODEL_DIR / 'data').resolve()
    return (REPO_ROOT / 'data').resolve()


DATA_DIR = _resolve_data_dir()
CACHE_DIR = MODEL_DIR / 'cache'
# Submission output directory. Default <repo>/submit; override with ETRI_OUT (e.g. ETRI_OUT=/data).
SUBMIT_DIR = Path(os.environ['ETRI_OUT']).resolve() if os.environ.get('ETRI_OUT') else MODEL_DIR / 'submit'


@dataclass(frozen=True)
class PipelineConfig:
    n_splits: int = 5
    fold_seed: int = 42
    knn_k: int = 3
    knn_metric: str = 'manhattan'
    knn_weighting: str = 'uniform'
    knn_p: float = None
    prior_window: int = 30
    prior_power: float = 2.0
    minirocket: bool = True
    multires: bool = False        # multi-resolution MiniROCKET scales {5,10,15}
    cnn_blend: bool = True
    base_temporal: bool = False
    ablate: FrozenSet[str] = frozenset()
    research_dump: bool = False


DEFAULTS = PipelineConfig()


def to_env(cfg: PipelineConfig) -> None:
    """Export a config to the environment variables the stage modules read.

    Not used on the reproduction path, which runs on the DEFAULTS above."""
    os.environ['KNN_K'] = str(cfg.knn_k)
    os.environ['KNN_METRIC'] = str(cfg.knn_metric)
    os.environ['KNN_WEIGHTING'] = str(cfg.knn_weighting)
    if cfg.knn_p is not None:
        os.environ['KNN_P'] = str(cfg.knn_p)
    os.environ['PRIOR_WINDOW'] = str(cfg.prior_window)
    os.environ['MINIROCKET'] = 'on' if cfg.minirocket else 'off'
    os.environ['MULTIRES'] = 'on' if cfg.multires else 'off'
    os.environ['CNN_BLEND'] = 'on' if cfg.cnn_blend else 'off'
    if cfg.base_temporal:
        os.environ['BASE_TEMPORAL'] = '1'
    if cfg.ablate:
        os.environ['ABLATE'] = ','.join(sorted(cfg.ablate))
    if cfg.research_dump:
        os.environ['RESEARCH_DUMP'] = '1'
