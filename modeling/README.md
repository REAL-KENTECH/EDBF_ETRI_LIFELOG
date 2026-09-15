# `modeling/`: the model library

Pure library code imported by `preprocess/` (feature builders) and `views/` (per-view builders + the
fuser). It defines the predictors, the convex-combination rules, and the deployed fusion tree. Nothing here
is an entry point; run the pipeline from the bundle root with `python run.py` (see top-level `README.md`).

## Deployed fusion (what `views/fuse.py` assembles)

Four frozen probability **views** are combined by clipped Bates–Granger (BG) with the sleep-tensor
MiniRocket co-view **pruned** (`use_mr=False`):

```
final = GlobalBG-LW( Platt( BG(tabular, BG(sleep_cnn, event_timing)) ), context )
```

| view | builder | what it is |
|---|---|---|
| tabular | `views/build_tabular.py`    | tabular `BG(LGBM, elastic-net)` over ~281 day-level features → kNN borrow → ±30d temporal prior |
| sleep_cnn| `views/build_sleep_cnn.py`     | multi-resolution pyramid CNN over the 6-channel 00–09h sleep tensor |
| event_timing | `views/fuse.py` (inline) | `BG(LGBM, elastic-net)` over the nocturnal event-timing features (`preprocess/events.py`) |
| context | `views/build_context.py` | MiniRocket over a 4-channel night tensor (charging / wifi / ble / app-usage) |

## Module map

| module | role |
|---|---|
| `config.py` | paths (`$ETRI_DATA → /data → <repo>/data`, `cache/`, `submit/`) + pipeline defaults |
| `blend/fusion.py` | the deployed tree as one call, `deployed_pipeline(...)`, plus the `bg_oof` / `bg_sub` / `platt_full` primitives |
| `blend/combine.py` | the convex-weight rules: `bg_alpha` (2-way Bates–Granger) and `global_bg_weights` (N-way Ledoit-Wolf) |
| `blend/neighbors.py` | k-NN / randomized-kNN feature-space borrowing (base view) |
| `blend/prior.py` | ±30d same-subject temporal prior (base view) |
| `encoders/cnn.py` | `PyramidSleepCNN`, the sleep-tensor CNN (PyTorch; only the from-raw path needs torch) |
| `features/tabular.py` | `load_features`, `PanelFXConfig`, OOF/sub runners for the LGBM/elastic-net tabular learners |
| `features/wb.py` | within–between (Mundlak) feature engineering used by the tabular base |
| `evaluation/cv.py`, `evaluation/metrics.py` | fold rulers + the macro log-loss of record |
| `stages/tabular_model.py` | the orchestrator `views/build_tabular.py` runs to produce the base view (LGBM⊕elastic-net → kNN → prior) |

## Reproduce

From the bundle root:

```bash
python run.py              # byte-identical (CPU, ~2 min) → md5 3b0f2da7
python run.py --from-raw   # rebuild every view from raw (GPU); not byte-identical
```

`python run.py` reuses the frozen `cache/view_*.npz` for the tabular, sleep_cnn and context views and
retrains the event_timing view from `/data`. The GPU CNN draw is not byte-reproducible, so `--from-raw`
is not byte-identical; score restoration uses `python run.py`.
