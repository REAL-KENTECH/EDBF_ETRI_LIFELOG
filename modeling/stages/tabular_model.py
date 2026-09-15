"""tabular_model.py — the features-only tabular view (per-label LGBM with neighbour borrowing).

Orchestrator for the tabular branch: it assembles the leaf modules (features/tabular.py and
features/wb.py for the LGBM, evaluation/cv.py for the folds) and runs the borrowing stages inline.

  Stage 1  per-label LGBM over the day-level features
  Stage 2  k-NN borrowing plus Bates-Granger blend (blend/neighbors.py)
  Stage 3  temporal prior (blend/prior.py)

The sleep-dimension predictors (the sleep-tensor CNN, event-timing) stay out of this base and are
fused late in views/fuse.py. The CNN is still computed here and cached for that later fusion.

Environment overrides used by views/build_tabular.py: TABULAR_LEARNER=bg selects the
BG(LGBM, elastic-net) base and writes cache/tabular_oof_bg.npz; RANDKNN=1 with RANDKNN_K=2 selects the
randomized k-NN borrow.

Inputs:  cache/features.parquet, cache/sleep_window_tensor.npz
Output:  cache/tabular_oof.npz, or cache/tabular_oof_bg.npz under TABULAR_LEARNER=bg
         (keys oof, sub, y, subj), repacked by views/build_tabular.py

Run: python modeling/stages/tabular_model.py
"""
from __future__ import annotations
import warnings, time, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from modeling.config import DEFAULTS as CFG, DATA_DIR
HERE = Path(__file__).resolve().parents[2]
FEATURES_PATH = HERE / 'cache/features.parquet'
SENSOR_TENSOR_PATH = HERE / 'cache/sleep_window_tensor.npz'
CNN_OOF_CACHE_PATH = HERE / 'cache/cnn_cache.npz'
CNN_TEMPORAL_CACHE_PATH = HERE / 'cache/cnn_cache_temporal.npz'
TABULAR_ARTIFACT_PATH = HERE / 'cache/tabular_oof.npz'
BASE_TEMPORAL = bool(os.environ.get('BASE_TEMPORAL')) or CFG.base_temporal
ABLATE = set(c.strip() for c in os.environ.get('ABLATE', '').split(',') if c.strip()) | set(CFG.ablate)
TABULAR_LEARNER = os.environ.get('TABULAR_LEARNER', 'lgbm')   # 'lgbm' (deployed) | 'linear' | 'bg' (opt-in)

LABELS = ('Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4')
EPS = 1e-12
HYPERPARAMS = {'n_splits': CFG.n_splits, 'fold_seed': CFG.fold_seed, 'cnn_seeds': (42, 43, 44, 45, 46), 'lgb_seeds_oof': (42, 43, 44, 45, 46), 'lgb_seeds_sub': (42, 43, 44, 45, 46, 47, 48, 49), 'knn_k': int(os.environ.get('KNN_K', CFG.knn_k)), 'knn_metric': os.environ.get('KNN_METRIC', CFG.knn_metric), 'knn_weighting': os.environ.get('KNN_WEIGHTING', CFG.knn_weighting), 'knn_p': (float(os.environ['KNN_P']) if os.environ.get('KNN_P') else CFG.knn_p)}
FORCE_CNN_TRAIN = False
CNN_SEEDS = list(HYPERPARAMS['cnn_seeds'])
from modeling.encoders.cnn import device, normalize_per_fold, train_cnn, predict_cnn
from modeling.evaluation.cv import subject_blocked_temporal_kfold
from modeling.features.wb import WithinBetweenDiagnoser, SubdayCrossBucketBuilder, DayLevelInteractionBuilder
from modeling.features.tabular import PanelFXConfig, load_features, run_oof_per_fold, OOFRunner
from modeling.blend.neighbors import knn_borrow, random_knn_borrow
from modeling.blend.combine import bg_alpha
from modeling.blend.prior import lag1_autocorr, temporal_prior, oof_temporal_prior, blend_with_prior
from modeling.evaluation.metrics import macro_logloss
_HR_COVERAGE_FLOOR = 0.1


def kept_channels():
    """Sleep-tensor channels with at least the floor coverage.

    Loaded when training the CNN rather than at import, so the module imports before
    preprocess/sleep_tensor.py has built the tensor."""
    mask = np.load(SENSOR_TENSOR_PATH, allow_pickle=True)['mask']
    cov = mask.mean(axis=(0, 2))
    kc = [c for c in range(mask.shape[1]) if cov[c] >= _HR_COVERAGE_FLOOR]
    print(f'CNN channels kept (>={_HR_COVERAGE_FLOOR * 100:.0f}% coverage): {kc}')
    return kc


def get_cnn_predictions(force_train: bool = False):
    """Load cached CNN OOF + sub predictions, or train fresh if cache missing / forced.

    Returns:
        (cnn_oof_per_seed, cnn_sub_per_seed): shapes (n_seeds, n_train, 7), (n_seeds, n_sub, 7)
    """
    cache_path = CNN_TEMPORAL_CACHE_PATH if BASE_TEMPORAL else CNN_OOF_CACHE_PATH
    if cache_path.exists() and not force_train:
        cnn_z = np.load(cache_path, allow_pickle=True)
        oof, sub = cnn_z['oof'], cnn_z['sub']
        print(f'Loaded CNN cache ({"temporal" if BASE_TEMPORAL else "shuffle"}): oof={oof.shape}, sub={sub.shape}')
        return oof, sub
    print(f'Training CNN {len(CNN_SEEDS)}-seed × {HYPERPARAMS["n_splits"]}-fold (device={device}, ~1min on GPU)...')
    KC = kept_channels()
    npz = np.load(SENSOR_TENSOR_PATH, allow_pickle=True)
    X_tensor = npz['X'][:, KC, :].astype(np.float32)
    M_tensor = npz['mask'][:, KC, :].astype(np.float32)
    tensor_subj = npz['subject_id']
    tensor_date = pd.to_datetime(npz['lifelog_date']).normalize()
    feat0 = pd.read_parquet(FEATURES_PATH)
    feat0['lifelog_date'] = pd.to_datetime(feat0['lifelog_date']).dt.normalize()
    feat0 = feat0.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
    tr_df0 = feat0[feat0.split == 'train'].reset_index(drop=True)
    sb_df0 = feat0[feat0.split == 'sub'].reset_index(drop=True)
    tlk = pd.DataFrame({'subject_id': tensor_subj, 'lifelog_date': tensor_date, 'tensor_idx': np.arange(len(tensor_subj))})
    tr_idx_t = tr_df0[['subject_id', 'lifelog_date']].merge(tlk, on=['subject_id', 'lifelog_date']).tensor_idx.values.astype(int)
    sub_idx_t = sb_df0[['subject_id', 'lifelog_date']].merge(tlk, on=['subject_id', 'lifelog_date']).tensor_idx.values.astype(int)
    assert len(tr_idx_t) == len(tr_df0), f'Train dataset row mismatch! Tabular: {len(tr_df0)}, Matched Tensor: {len(tr_idx_t)}'
    assert len(sub_idx_t) == len(sb_df0), f'Sub dataset row mismatch! Tabular: {len(sb_df0)}, Matched Tensor: {len(sub_idx_t)}'
    Xtr_full = X_tensor[tr_idx_t]
    Mtr_full = M_tensor[tr_idx_t]
    Xsub_full = X_tensor[sub_idx_t]
    Msub_full = M_tensor[sub_idx_t]
    subj_unique = sorted(tr_df0.subject_id.unique())
    s2i = {s: i for (i, s) in enumerate(subj_unique)}
    str_full = np.array([s2i[s] for s in tr_df0.subject_id], dtype=np.int64)
    ssub_full = np.array([s2i[s] for s in sb_df0.subject_id], dtype=np.int64)
    n_tr0 = len(tr_df0)
    n_sb0 = len(sb_df0)
    y_cnn = tr_df0[list(LABELS)].to_numpy().astype(float)
    n_ch = len(KC)
    n_subj = len(subj_unique)
    cnn_oof_per_seed = np.zeros((len(CNN_SEEDS), n_tr0, len(LABELS)))
    cnn_sub_per_seed = np.zeros((len(CNN_SEEDS), n_sb0, len(LABELS)))
    if BASE_TEMPORAL:
        cnn_fold_splits = subject_blocked_temporal_kfold(
            tr_df0.subject_id.to_numpy(), tr_df0.lifelog_date.to_numpy(), HYPERPARAMS['n_splits'])
        print('  CNN folds: subject_blocked_temporal (Stage-C clean)')
    else:
        kf = KFold(n_splits=HYPERPARAMS['n_splits'], shuffle=True, random_state=HYPERPARAMS['fold_seed'])
        cnn_fold_splits = list(kf.split(np.arange(n_tr0)))
    t_total = time.time()
    for (si, seed) in enumerate(CNN_SEEDS):
        sub_acc = np.zeros((n_sb0, len(LABELS)))
        for (fold, (tr_, va_)) in enumerate(cnn_fold_splits):
            (Xtr, Mtr) = (Xtr_full[tr_], Mtr_full[tr_])
            (Xva, Mva) = (Xtr_full[va_], Mtr_full[va_])
            ytr_ = y_cnn[tr_].astype(np.float32)
            (str_, sva) = (str_full[tr_], str_full[va_])
            (Xtr_n, Xva_n, Xsub_n) = normalize_per_fold(Xtr, Mtr, Xva, Xsub_full)
            np.random.seed(seed)
            perm = np.random.permutation(len(tr_))
            v_n = max(20, len(tr_) // 10)
            (ts, vs) = (perm[v_n:], perm[:v_n])
            model = train_cnn(Xtr_n[ts], Mtr[ts], str_[ts], ytr_[ts], Xtr_n[vs], Mtr[vs], str_[vs], ytr_[vs], n_ch=n_ch, n_subj=n_subj, seed=seed)
            cnn_oof_per_seed[si, va_] = predict_cnn(model, Xva_n, Mva, sva)
            sub_acc += predict_cnn(model, Xsub_n, Msub_full, ssub_full)
        cnn_sub_per_seed[si] = sub_acc / HYPERPARAMS['n_splits']
        print(f'  seed {seed} done ({time.time() - t_total:.1f}s)')
    os.makedirs(cache_path.parent, exist_ok=True)
    np.savez(cache_path, oof=cnn_oof_per_seed, sub=cnn_sub_per_seed, seeds=np.array(CNN_SEEDS))
    print(f'Saved: {cache_path}')
    return cnn_oof_per_seed, cnn_sub_per_seed


def main():
    """Run the full LGBM + BG + temporal prior pipeline; writes submission CSV."""
    cnn_oof_per_seed, cnn_sub_per_seed = get_cnn_predictions(FORCE_CNN_TRAIN)

    cfg = PanelFXConfig.default_derived()
    cfg.seeds_oof = HYPERPARAMS['lgb_seeds_oof']
    cfg.seeds_sub = HYPERPARAMS['lgb_seeds_sub']
    if 'subday' in ABLATE: cfg.use_subday = False
    if 'daylevel' in ABLATE: cfg.use_day_level = False
    if ABLATE: print(f'[ABLATE] removed: {sorted(ABLATE)}')

    data = load_features(DATA_DIR, labels=cfg.labels)
    df = data['df']
    base_feat_cols = data['base_feat_cols']

    diag = WithinBetweenDiagnoser(
        labels=cfg.labels, min_raw_r=cfg.min_raw_r,
        day_rich_ratio=cfg.day_rich_ratio, between_only_ratio=cfg.between_only_ratio
    ).fit(data['train_df'], base_feat_cols)

    if cfg.use_day_level:
        day_level_b = DayLevelInteractionBuilder(
            top_k_dayrich=cfg.day_level_top_k_pool,
            top_k_inter=cfg.day_level_top_k_per_label,
            labels=cfg.labels, min_raw_r=cfg.min_raw_r,
            day_rich_ratio=cfg.day_rich_ratio,
            between_only_ratio=cfg.between_only_ratio,
        )
        day_level_cols, df = day_level_b.fit_transform(data['train_df'], df, diag=diag)
    else:
        day_level_cols = []

    if cfg.use_subday:
        subday_b = SubdayCrossBucketBuilder(
            mode=cfg.subday_selection_mode, top_k=cfg.subday_top_k, pinned=cfg.subday_pinned,
        )
        subday_cols, df = subday_b.fit_transform(df, diag=diag)
    else:
        subday_cols = []

    train_df = df[df.split == 'train'].reset_index(drop=True)
    sub_df = df[df.split == 'sub'].reset_index(drop=True)
    train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date'])
    sub_df['lifelog_date'] = pd.to_datetime(sub_df['lifelog_date'])

    day_rich = (diag.df[diag.df.category == 'day_rich']
                .assign(abs_r=lambda d: d.raw_r.abs())
                .sort_values('abs_r', ascending=False).feature.unique().tolist())
    enrich_feats = [f for f in day_rich if f in train_df.columns][:cfg.wb_enrichment_top_k]
    if os.environ.get('WB_FULL'):
        enrich_feats = [c for c in base_feat_cols if c in train_df.columns and np.issubdtype(train_df[c].dtype, np.number)
                        and train_df[c].nunique() > 5 and not (('_lag1' in c) or c.startswith('cal_')
                        or ('_subj_' in c) or ('_missing' in c) or ('_is_' in c) or c.endswith('is_weekend'))]

    sleep_cols = [f'cnn_{lab}' for lab in LABELS]
    for li, c in enumerate(sleep_cols):
        train_df[c] = cnn_oof_per_seed.mean(axis=0)[:, li]
        sub_df[c] = cnn_sub_per_seed.mean(axis=0)[:, li]

    base_cols = list(base_feat_cols + day_level_cols) + sleep_cols
    y = train_df[list(LABELS)].to_numpy().astype(np.float32)

    train_df_raw = data['train_df'].reset_index(drop=True)
    train_df_raw['lifelog_date'] = pd.to_datetime(train_df_raw['lifelog_date'])
    cnn_oof_mean = cnn_oof_per_seed.mean(axis=0).astype(np.float32)
    print('Stage 1 OOF (per-fold leak-safe)...')
    runner = OOFRunner(cfg, enrich_feats=enrich_feats)
    if os.environ.get('CNN_BLEND', 'on' if CFG.cnn_blend else 'off') != 'off':
        base_cols_nc = [c for c in base_cols if c not in sleep_cols]

        def _tabular_for(backend):
            """Stage-1 features-only tabular view for a given learner backend (deployed feature pipeline)."""
            cfg.model_backend = backend
            saved = (cfg.seeds_oof, cfg.seeds_sub)
            if backend == 'linear':                          # elastic-net is deterministic -> 1 seed suffices
                cfg.seeds_oof, cfg.seeds_sub = (42,), (42,)
            o = run_oof_per_fold(train_df_raw, list(base_feat_cols), np.zeros_like(cnn_oof_mean), cfg, LABELS, verbose=False)
            s = OOFRunner(cfg, enrich_feats=enrich_feats).predict_sub(
                train_df, sub_df, base_cols=base_cols_nc, subday_cols=subday_cols, labels=LABELS, verbose=False)
            cfg.seeds_oof, cfg.seeds_sub = saved
            return o.clip(EPS, 1 - EPS).astype(np.float32), s.clip(EPS, 1 - EPS).astype(np.float32)

        # TABULAR_LEARNER: 'lgbm' (deployed default) | 'linear' (elastic-net) | 'bg' (bounded blend of both)
        if TABULAR_LEARNER == 'bg':
            o_l, s_l = _tabular_for('lgbm')
            o_n, s_n = _tabular_for('linear')
            al = np.array([bg_alpha(o_l[:, i], o_n[:, i], y[:, i]) for i in range(len(LABELS))])
            print('  [base] BG(LGBM, linear)  alpha_linear: ' + ' '.join(f'{L}={a:.2f}' for L, a in zip(LABELS, al)))
            oof_tabular = np.stack([(1 - al[i]) * o_l[:, i] + al[i] * o_n[:, i] for i in range(len(LABELS))], 1).clip(EPS, 1 - EPS)
            sub_tabular = np.stack([(1 - al[i]) * s_l[:, i] + al[i] * s_n[:, i] for i in range(len(LABELS))], 1).clip(EPS, 1 - EPS)
        else:
            oof_tabular, sub_tabular = _tabular_for(TABULAR_LEARNER)
        print(f'  [base] features-only tabular view learner = {TABULAR_LEARNER} (CNN pulled out → late sleep-fusion)')
    else:
        oof_tabular = run_oof_per_fold(train_df_raw, list(base_feat_cols), cnn_oof_mean, cfg, LABELS, verbose=False)
        print('Stage 1 sub (full-train features, leak-safe for sub)...')
        sub_tabular = runner.predict_sub(train_df, sub_df, base_cols=base_cols, subday_cols=subday_cols, labels=LABELS, verbose=False)

    num_cols = [c for c in base_feat_cols if np.issubdtype(train_df[c].dtype, np.number)]
    F_tr_nan = train_df[num_cols].to_numpy().astype(np.float32)
    F_sb_nan = sub_df[num_cols].to_numpy().astype(np.float32)
    if BASE_TEMPORAL:
        fold_splits = subject_blocked_temporal_kfold(
            train_df.subject_id.to_numpy(), train_df.lifelog_date.to_numpy(), HYPERPARAMS['n_splits'])
        print('  kNN/prior folds: subject_blocked_temporal (Stage-C clean)')
    else:
        fold_splits = list(KFold(n_splits=HYPERPARAMS['n_splits'], shuffle=True,
                                 random_state=HYPERPARAMS['fold_seed']).split(np.arange(len(train_df))))
    if os.environ.get('RANDKNN') == '1':
        rk_M = int(os.environ.get('RANDKNN_M', 50)); rk_frac = float(os.environ.get('RANDKNN_FRAC', 0.8)); rk_k = int(os.environ.get('RANDKNN_K', 2))
        oof_hybrid, sub_hybrid, bg_alpha_sub = random_knn_borrow(
            oof_tabular, sub_tabular, F_tr_nan, F_sb_nan, y, fold_splits,
            M=rk_M, frac=rk_frac, k=rk_k, metric=HYPERPARAMS['knn_metric'], eps=EPS)
        print(f'  Stage 2: randKNN ensemble (M={rk_M}, frac={rk_frac}, k={rk_k})')
    else:
        oof_hybrid, sub_hybrid, bg_alpha_sub = knn_borrow(
            oof_tabular, sub_tabular, F_tr_nan, F_sb_nan, y, fold_splits,
            k=HYPERPARAMS['knn_k'], metric=HYPERPARAMS['knn_metric'], weighting=HYPERPARAMS['knn_weighting'], p=HYPERPARAMS['knn_p'], eps=EPS)
    print('Bates-Granger alpha per label (full-OOF estimate): '
          + ', '.join(f'{L}={a:.3f}' for L, a in zip(LABELS, bg_alpha_sub)))

    max_days = int(os.environ.get('PRIOR_WINDOW', CFG.prior_window))
    autocorr_raw = lag1_autocorr(train_df, y, LABELS)
    prior_oof, _ = oof_temporal_prior(train_df, y, fold_splits,
                                      max_days=max_days, kernel='power', power=CFG.prior_power)
    prior_sub, _ = temporal_prior(sub_df['subject_id'].to_numpy(),
                                  sub_df['lifelog_date'].to_numpy(),
                                  train_df['subject_id'].to_numpy(),
                                  train_df['lifelog_date'].to_numpy(),
                                  y, max_days=max_days, kernel='power', power=CFG.prior_power)
    oof_final = blend_with_prior(oof_hybrid, prior_oof, autocorr_raw)
    sub_final = blend_with_prior(sub_hybrid, prior_sub, autocorr_raw)

    if os.environ.get('RESEARCH_DUMP') or CFG.research_dump:
        np.savez(HERE / 'cache/gate_inputs.npz',
                 oof_hybrid=oof_hybrid, sub_hybrid=sub_hybrid, w=autocorr_raw, y=y,
                 oof_tabular=oof_tabular, sub_tabular=sub_tabular,
                 F_tr=F_tr_nan, F_sb=F_sb_nan,
                 tr_subj=train_df['subject_id'].to_numpy(),
                 tr_date=train_df['lifelog_date'].to_numpy().astype('datetime64[ns]'),
                 sb_subj=sub_df['subject_id'].to_numpy(),
                 sb_date=sub_df['lifelog_date'].to_numpy().astype('datetime64[ns]'))
        print('  RESEARCH_DUMP: wrote cache/gate_inputs.npz (pre-prior hybrid + dates)')


    per_label_w = ', '.join((f'{L}={w:.2f}' for (L, w) in zip(LABELS, autocorr_raw)))
    print(f'Temporal prior weights (lag-1 autocorr): {per_label_w}')
    print('\nOOF mean log-loss:')
    print(f'  Stage 1 base                : {macro_logloss(oof_tabular, y, EPS):.5f}')
    print(f'  Stage 2 Bates-Granger blend : {macro_logloss(oof_hybrid, y, EPS):.5f}')
    print(f'  Stage 3 + temporal prior    : {macro_logloss(oof_final, y, EPS):.5f}')

    lr_tag = '' if TABULAR_LEARNER == 'lgbm' else f'_{TABULAR_LEARNER}'
    suffix = lr_tag + ('_temporal' if BASE_TEMPORAL else '')
    out_path = TABULAR_ARTIFACT_PATH if suffix == '' else TABULAR_ARTIFACT_PATH.parent / f'tabular_oof{suffix}.npz'
    os.makedirs(out_path.parent, exist_ok=True)
    subj_arr = train_df['subject_id'].to_numpy()
    np.savez(out_path, oof=oof_final, sub=sub_final, y=y, subj=subj_arr)
    print(f'Saved base artifact: {out_path}')


if __name__ == "__main__":
    main()
