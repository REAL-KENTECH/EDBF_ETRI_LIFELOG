# `preprocess/`: raw sensors to input caches

One script, one artifact. Nothing here trains a model.

## Order

```bash
python preprocess/daylevel.py       # -> cache/features.parquet
python preprocess/sleep_tensor.py   # -> cache/sleep_window_tensor.npz
python preprocess/timing.py         # -> cache/timing_features.parquet
python preprocess/context.py        # -> cache/context_tensor.npz
python preprocess/events.py         # -> cache/timing_events.parquet   (needs timing.py first)

python verify_cache.py              # confirms every shape below
```

Only `events.py` has a prerequisite; the other four are independent. `python run.py --from-raw` runs
those four and then the view builders. It does not list `events.py` because `views/fuse.py` calls
`build_complete_timing()` itself on every run.

## Artifacts

| script | artifact | shape | consumed by |
|---|---|---|---|
| `daylevel.py` | `cache/features.parquet` | 700 × 281 features | `tabular` view; also the canonical row keys |
| `sleep_tensor.py` | `cache/sleep_window_tensor.npz` | `X`, `mask` (700, 6, 108) | `sleep_cnn` view |
| `timing.py` | `cache/timing_features.parquet` | 700 × 19 | `events.py` only |
| `events.py` | `cache/timing_events.parquet` | 700 × 52 (19 rhythm + 33 event) | `event_timing` view |
| `context.py` | `cache/context_tensor.npz` | `X` (700, 4, 108) | `context` view |

`daylevel.py` also writes `cache/feature_column_groups.json`, a diagnostic sidecar nothing reads.

## Before you edit

**Row order is a contract.** `daylevel.py`, `sleep_tensor.py` and `context.py` each build their rows
from `all_keys` = train + submission-sample keys sorted by `(subject_id, lifelog_date)`. The consumers
merge on those keys. Change one, change all three; `verify_cache.py` checks them against each other.

**Label columns on `sub` rows differ by table.** `daylevel.py` merges train labels only, so `sub` rows
are NaN. `timing.py` carries the label columns of `ch2026_submission_sample.csv`, which the organisers
ship filled with `0`. Those zeros are placeholders; every consumer drops the label columns.

**Every non-key column of `timing_events.parquet` becomes a model input.** `views/fuse.py` takes all
columns except the keys, `sleep_date`, `split` and the 7 labels. Adding one changes the model. This is
why `timing.py` computes no fold index.

**Frozen constants.** The sleep-episode detector in `timing.py` (`DET_THR`, `DET_SMOOTH`,
`DET_MIN_RUN`, `DET_MERGE`, the per-subject resting-HR quantile) and the median fill in
`events.py:build_complete_timing` are baked into the shipped submission (md5 `3b0f2da7`). The median
is taken over train and sub rows together: no label involved, but transductive. Changing either moves
the predictions.

## History

`daylevel.py` and `sleep_tensor.py` were one file, `features.py`. `context.py` was the `build_tensor`
function inside `views/build_context.py`, and now builds all 700 rows in one pass instead of once per
split. Both splits were verified to produce identical artifacts; see `PREPROCESS_REFACTOR.txt`.
