# ETRI Lifelog 2026 Submission Package (team SML)

Reproduces the submitted predictions for 7 binary sleep targets (macro log-loss).

| file | md5 | public LB | private LB |
|---|---|---|---|
| `submit/ch2026_submission.csv` | `3b0f2da7` | 0.5543165607 | **0.56758198** |

## Reproduce

Python 3.10-3.13, verified on 3.10.20 and 3.13.13. Not 3.14: the pinned numpy/scipy have no wheels
for it and pip falls back to a source build that fails.

```bash
pip install -r requirements.txt     # CPU-only; torch needed only for --from-raw
                                    # requirements-lock.txt pins transitive deps for the byte-identical check

python run.py              # byte-identical -> md5 3b0f2da7   (CPU, ~2 min; see Reproducibility)
python run.py --from-raw   # rebuild every view from raw, then fuse   (GPU; not byte-identical)
```

On Windows, in PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py
```

Calling `.venv\Scripts\python.exe` needs no activation. If you prefer to activate, PowerShell blocks
the script until you allow it for the session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
```

- Input: raw competition data auto-detected at `/data` (`ch2025_data_items/*.parquet`, `ch2026_metrics_train.csv`, `ch2026_submission_sample.csv`); otherwise `export ETRI_DATA=/path/to/data`.
- Output: `submit/ch2026_submission.csv`; override the directory with `export ETRI_OUT=/data`.
- `python run.py` rebuilds `cache/timing_events.parquet` from `/data` and retrains the `event_timing` view on every run; the `tabular`, `sleep_cnn` and `context` views come from the shipped `cache/view_*.npz` (our own trained outputs).
- `python verify_cache.py` checks the shipped input caches against the shapes above without running a model.
- No external pretrained models. MiniRocket uses random kernels; no downloaded weights.
- Only the GPU CNN draw is non-byte-reproducible, so `--from-raw` is not byte-identical.

### Reproducibility

`md5 3b0f2da7` is reproduced on the platform in `requirements.txt` (Linux x86_64, PyPI wheels); it was
verified there under five independent conditions: conda py3.10.20, a fresh py3.13.13 venv, single-thread
(`OMP/MKL/OPENBLAS_NUM_THREADS=1`), a deleted `submit/`, and a redirected `ETRI_OUT`.

On another OS the last bits can move, because LightGBM, BLAS and libm are built by a different compiler.
When the md5 differs, `run.py` compares the output against the shipped `reference/ch2026_submission.csv`
and reports the gap. Macro log-loss is smooth on (0, 1) with sensitivity at most `1/min(p, 1-p)`, so the
per-cell gap bounds the score shift directly. A Windows 11 run, for example, gives

```
  vs reference: max |delta| 1.33e-15 over 1750 cells; macro log-loss moves by at most 5.44e-16
  (the leaderboard prints 1e-8) -> score-identical
```

which is 7 orders of magnitude below the reported precision: the same score, a different md5.

## Method

Error-Diverse Bounded Fusion: four probability views combined by clipped Bates–Granger, sleep-tensor
MiniRocket co-view pruned (`USE_MR=False`):

```
final = GlobalBG-LW( Platt( BG(tabular, BG(sleep_cnn, event_timing)) ), context )
```

| view | input | model |
|---|---|---|
| `tabular` | ~281 day-level features | BG(LGBM, elastic-net) → kNN → ±30d prior |
| `sleep_cnn` | 6-ch 00–09h sleep tensor | multi-resolution pyramid CNN |
| `event_timing` | nocturnal event-timing (screen/move/lux/ambience/HR) | BG(LGBM, elastic-net) |
| `context` | 4-ch night tensor (charge/wifi/ble/usage) | MiniRocket |

Each event-timing signal enriches the `event_timing` view's features (`preprocess/events.py`), not a
separate branch. `USE_MR=True` (+ `build_sleep_cnn.py --with-mr`) reproduces the pre-prune build
`f3602b09` (0.5563241004).

## Layout

```
run.py                  python run.py  |  python run.py --from-raw
verify_cache.py         checks the input caches against the shapes documented here (reads only)
preprocess/             raw sensors → input caches.  One script, one artifact; see preprocess/README.md
  daylevel.py             → cache/features.parquet          700 × 281 day-level features
  sleep_tensor.py         → cache/sleep_window_tensor.npz   X/mask (700, 6, 108)
  timing.py               → cache/timing_features.parquet   700 × 19  (intermediate, for events.py)
  events.py               → cache/timing_events.parquet     700 × 52  (regenerated on every run)
  context.py              → cache/context_tensor.npz        X (700, 4, 108)
views/                  one builder per view + the fuser
  build_tabular.py        → cache/view_tabular.npz
  build_sleep_cnn.py      → cache/view_sleep_cnn.npz   (--with-mr adds the pruned MR view)
  build_context.py        → cache/view_context.npz
  fuse.py                 → submit/ch2026_submission.csv
modeling/               library: config · blend/{fusion,combine,neighbors,prior} · encoders · features · evaluation · stages
cache/                  view_tabular.npz · view_sleep_cnn.npz · view_context.npz · features.parquet · timing_features.parquet
submit/                 ch2026_submission.csv  (md5 3b0f2da7)
reference/              ch2026_submission.csv  (pristine copy; never written, used by the numeric check)
requirements.txt        library versions + OS/Python/CUDA;  requirements-lock.txt = exact verified environment
```
