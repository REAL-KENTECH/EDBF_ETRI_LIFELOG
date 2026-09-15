"""tabular.py — per-label LGBM (PanelFX): configuration, two-stage feature selection, full-train fit,
and the WB-enriched per-fold OOF runner.

Imports the CV splits (evaluation/cv.py) and the Within-Between feature builders (features/wb.py), and
reads cache/features.parquet. Produces the per-label LGBM OOF/sub predictions.
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass, field
from typing import Sequence, Literal
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from modeling.evaluation.cv import subject_blocked_temporal_kfold
from modeling.features.wb import WithinBetweenDiagnoser, WithinBetweenInputBuilder, SubdayCrossBucketBuilder, DayLevelInteractionBuilder


def _clean_mat(A):
    return np.nan_to_num(np.asarray(A, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

BASE_TEMPORAL = bool(os.environ.get('BASE_TEMPORAL'))
ABLATE = set(c.strip() for c in os.environ.get('ABLATE', '').split(',') if c.strip())
FEATURES_PATH = Path(__file__).resolve().parents[2] / 'cache/features.parquet'
PINNED_LB_REPRODUCER = ('m_charging_bucket_evening_ratio', 'm_screen_use_bucket_overnight_ratio', 'w_light_bucket_evening_mean_log1p', 'w_light_bucket_evening_mean', 'usage_total_time_bucket_overnight_ms', 'm_screen_use_bucket_afternoon_ratio', 'm_light_bucket_morning_mean', 'm_light_bucket_morning_mean_log1p', 'm_light_bucket_afternoon_mean', 'w_light_bucket_overnight_mean', 'whr_bucket_afternoon_mean', 'whr_bucket_morning_mean', 'm_activity_active_bucket_evening_ratio', 'm_charging_bucket_afternoon_ratio', 'm_screen_use_bucket_evening_ratio')


def _default_lgb_full() -> dict:
    """Default LightGBM params for the full training run (slow, deep)."""
    d = dict(objective='binary', metric='binary_logloss', num_leaves=15, learning_rate=0.02, n_estimators=2000, min_child_samples=15, reg_lambda=1.0, reg_alpha=0.1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, verbose=-1)
    for k, env in [('num_leaves', 'LGB_NL'), ('reg_lambda', 'LGB_RL'), ('reg_alpha', 'LGB_RA'), ('min_child_samples', 'LGB_MCS'), ('feature_fraction', 'LGB_FF'), ('learning_rate', 'LGB_LR')]:
        if os.environ.get(env):
            d[k] = (int if k in ('num_leaves', 'min_child_samples') else float)(os.environ[env])
    return d

def _default_lgb_fast() -> dict:
    """Default LightGBM params for the fast feature-selection run."""
    return dict(objective='binary', metric='binary_logloss', num_leaves=15, learning_rate=0.05, n_estimators=200, min_child_samples=15, reg_lambda=1.0, reg_alpha=0.1, verbose=-1)

@dataclass

class PanelFXConfig:
    """Pipeline configuration (seeds, K-fold settings, feature options, LGB params)."""
    labels: tuple = ('Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4')
    q_labels: tuple = ('Q1', 'Q2', 'Q3')
    s_labels: tuple = ('S1', 'S2', 'S3', 'S4')
    use_day_level: bool = False
    use_subday: bool = False
    min_raw_r: float = 0.1
    day_rich_ratio: float = 0.7
    between_only_ratio: float = 0.3
    day_level_top_k_pool: int = 30
    day_level_top_k_per_label: int = 5
    subday_selection_mode: Literal['derived', 'pinned'] = 'derived'
    subday_top_k: int = 15
    subday_pinned: tuple = PINNED_LB_REPRODUCER
    subday_top_k_per_label: int = 10
    model_backend: Literal['lgbm', 'linear', 'extratrees'] = 'lgbm'
    wb_enrichment_top_k: int = 30
    lgb_full: dict = field(default_factory=_default_lgb_full)
    lgb_fast: dict = field(default_factory=_default_lgb_fast)
    n_splits: int = 5
    early_stop: int = 100
    seeds_oof: tuple = (42, 43, 44)
    seeds_sub: tuple = (42, 43, 44, 45, 46, 47, 48, 49)

    @classmethod
    def default_derived(cls, **overrides) -> 'PanelFXConfig':
        return cls(subday_selection_mode='derived', subday_pinned=PINNED_LB_REPRODUCER, **overrides)

META_COLS = ('subject_id', 'lifelog_date', 'sleep_date', 'split')

def load_features(data_dir, labels=('Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4')):
    """Load the day-level feature table (this folder's cache/features.parquet), split train/sub.
    Uses the module FEATURES_PATH — data_dir kept only for signature compatibility."""
    feat_path = FEATURES_PATH
    df = pd.read_parquet(feat_path)
    df['lifelog_date'] = pd.to_datetime(df['lifelog_date'])
    df = df.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
    base_feat_cols = [c for c in df.columns if c not in (*META_COLS, *labels)]
    bucket_cols = [c for c in base_feat_cols if 'bucket' in c.lower() or '_b_' in c]
    return {
        'df': df,
        'train_df': df[df.split == 'train'].reset_index(drop=True),
        'sub_df': df[df.split == 'sub'].reset_index(drop=True),
        'base_feat_cols': base_feat_cols,
        'bucket_cols': bucket_cols,
        'feat_path': feat_path,
    }


class SubdayStandaloneLGB:
    """Two-stage LGB: fast feature selection then full train on base + top-K subday."""

    def __init__(self, params_full: dict, params_fast: dict, top_k_per_label: int=10, early_stop: int=100):
        self.params_full = dict(params_full)
        self.params_fast = dict(params_fast)
        self.top_k_per_label = top_k_per_label
        self.early_stop = early_stop
        self.selection_log_: dict[tuple, list[str]] = {}
        self.models_: dict[tuple, lgb.LGBMClassifier] = {}

    @staticmethod
    def _carve_inner_val(n_tr: int, seed: int, frac: float = 0.15, min_inner: int = 20):
        """Return (inner_tr_idx, inner_va_idx) — random split of [0, n_tr) at given seed."""
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_tr)
        n_iv = max(min_inner, int(n_tr * frac))
        return perm[n_iv:], perm[:n_iv]

    def select_top_subday(self, X_tr: pd.DataFrame, y_tr: pd.Series, subday_cols: list[str], seed: int, sample_weight: np.ndarray=None) -> list[str]:
        params = dict(self.params_fast)
        params['random_state'] = seed
        it, iv = self._carve_inner_val(len(X_tr), seed)
        sw_it = sample_weight[it] if sample_weight is not None else None
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr.iloc[it], y_tr.iloc[it], sample_weight=sw_it,
              eval_set=[(X_tr.iloc[iv], y_tr.iloc[iv])],
              callbacks=[lgb.early_stopping(self.early_stop, verbose=False)])
        imp = pd.Series(m.feature_importances_, index=X_tr.columns)
        subday_imp = imp[imp.index.isin(subday_cols)].sort_values(ascending=False)
        return subday_imp.head(self.top_k_per_label).index.tolist()

    def fit_fold(self, X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, y_te: pd.Series, base_cols: list[str], subday_cols: list[str], label: str, fold_id: int, seed: int, sample_weight: np.ndarray=None) -> tuple[list[str], lgb.LGBMClassifier]:
        all_cols = base_cols + subday_cols
        top_k = self.select_top_subday(X_tr[all_cols], y_tr, subday_cols, seed=seed, sample_weight=sample_weight)
        active = base_cols + top_k
        params = dict(self.params_full)
        params['random_state'] = seed
        it, iv = self._carve_inner_val(len(X_tr), seed + 1)
        sw_it = sample_weight[it] if sample_weight is not None else None
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr[active].iloc[it], y_tr.iloc[it], sample_weight=sw_it,
              eval_set=[(X_tr[active].iloc[iv], y_tr.iloc[iv])],
              callbacks=[lgb.early_stopping(self.early_stop, verbose=False)])
        key = (label, fold_id, seed)
        self.selection_log_[key] = top_k
        self.models_[key] = m
        return (top_k, m)

    def predict_fold(self, X: pd.DataFrame, base_cols: list[str], label: str, fold_id: int, seed: int) -> np.ndarray:
        key = (label, fold_id, seed)
        top_k = self.selection_log_[key]
        active = base_cols + top_k
        return self.models_[key].predict_proba(X[active])[:, 1]

    def fit(self, *args, **kwargs):
        raise NotImplementedError('use fit_fold per (label, fold) instead')

    def predict_proba(self, *args, **kwargs):
        raise NotImplementedError('use predict_fold per (label, fold) instead')

class SubdayStandaloneLinear:
    """Elastic-net logistic analogue of SubdayStandaloneLGB (same fit_fold / predict_fold interface).

    Median-imputes and standardizes per fold, fit on the train fold only. Selected by
    model_backend='linear'."""

    def __init__(self, top_k_per_label: int = 10, C: float = 0.2, l1_ratio: float = 0.5):
        self.top_k_per_label = top_k_per_label
        self.C = float(os.environ.get('LIN_C', C))      # deployed value; override with LIN_C
        self.l1_ratio = l1_ratio
        self.rs = 42                                     # fixed saga seed -> deterministic (byte-reproducible)
        self.selection_log_: dict = {}
        self.models_: dict = {}

    def _fit_prep(self, X: pd.DataFrame):
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(X)
        sc = StandardScaler().fit(_clean_mat(imp.transform(X)))
        return imp, sc

    def _apply(self, X, imp, sc):
        return _clean_mat(sc.transform(_clean_mat(imp.transform(X))))

    def select_top_subday(self, X_tr, y_tr, subday_cols, seed, sample_weight=None):
        if not subday_cols:
            return []
        imp, sc = self._fit_prep(X_tr)
        m = LogisticRegression(penalty='l1', solver='saga', C=self.C, max_iter=2000,
                               random_state=self.rs).fit(self._apply(X_tr, imp, sc), y_tr)
        coef = pd.Series(np.abs(m.coef_[0]), index=X_tr.columns)
        return coef[coef.index.isin(subday_cols)].sort_values(ascending=False).head(self.top_k_per_label).index.tolist()

    def fit_fold(self, X_tr, y_tr, X_te, y_te, base_cols, subday_cols, label, fold_id, seed, sample_weight=None):
        top_k = self.select_top_subday(X_tr[base_cols + subday_cols], y_tr, subday_cols, seed)
        active = base_cols + top_k
        imp, sc = self._fit_prep(X_tr[active])
        m = LogisticRegression(penalty='elasticnet', l1_ratio=self.l1_ratio, solver='saga', C=self.C,
                               max_iter=3000, random_state=self.rs).fit(self._apply(X_tr[active], imp, sc), y_tr)
        key = (label, fold_id, seed)
        self.selection_log_[key] = (top_k, imp, sc)
        self.models_[key] = m
        return top_k, m

    def predict_fold(self, X, base_cols, label, fold_id, seed):
        top_k, imp, sc = self.selection_log_[(label, fold_id, seed)]
        return self.models_[(label, fold_id, seed)].predict_proba(self._apply(X[base_cols + top_k], imp, sc))[:, 1]

    def fit(self, *a, **k): raise NotImplementedError('use fit_fold per (label, fold)')
    def predict_proba(self, *a, **k): raise NotImplementedError('use predict_fold per (label, fold)')


class SubdayStandaloneET:
    """Extra-Trees analogue of SubdayStandaloneLGB (same fit_fold / predict_fold interface).

    Median-imputes per fold, since ET cannot ingest NaN. Not part of the deployed build; selected by
    model_backend='extratrees'."""

    def __init__(self, top_k_per_label: int = 10, n_estimators: int = 400):
        self.top_k_per_label = top_k_per_label
        self.n_estimators = int(os.environ.get('ET_TREES', n_estimators))
        self.selection_log_: dict = {}
        self.models_: dict = {}

    def _et(self, seed):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=self.n_estimators, max_features='sqrt',
                                    min_samples_leaf=3, n_jobs=-1, random_state=seed)

    def select_top_subday(self, X_tr, y_tr, subday_cols, seed, sample_weight=None):
        if not subday_cols:
            return []
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(X_tr)
        m = self._et(seed).fit(_clean_mat(imp.transform(X_tr)), y_tr)
        coef = pd.Series(m.feature_importances_, index=X_tr.columns)
        return coef[coef.index.isin(subday_cols)].sort_values(ascending=False).head(self.top_k_per_label).index.tolist()

    def fit_fold(self, X_tr, y_tr, X_te, y_te, base_cols, subday_cols, label, fold_id, seed, sample_weight=None):
        top_k = self.select_top_subday(X_tr[base_cols + subday_cols], y_tr, subday_cols, seed)
        active = base_cols + top_k
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(X_tr[active])
        m = self._et(seed).fit(_clean_mat(imp.transform(X_tr[active])), y_tr)
        key = (label, fold_id, seed)
        self.selection_log_[key] = (top_k, imp)
        self.models_[key] = m
        return top_k, m

    def predict_fold(self, X, base_cols, label, fold_id, seed):
        top_k, imp = self.selection_log_[(label, fold_id, seed)]
        return self.models_[(label, fold_id, seed)].predict_proba(_clean_mat(imp.transform(X[base_cols + top_k])))[:, 1]

    def fit(self, *a, **k): raise NotImplementedError('use fit_fold per (label, fold)')
    def predict_proba(self, *a, **k): raise NotImplementedError('use predict_fold per (label, fold)')


def make_subday_standalone(backend: str, cfg):
    """Backend factory: 'lgbm' and 'linear' are the two deployed bases; 'extratrees' is off-path."""
    if backend == 'lgbm':
        return SubdayStandaloneLGB(params_full=cfg.lgb_full, params_fast=cfg.lgb_fast, top_k_per_label=cfg.subday_top_k_per_label, early_stop=cfg.early_stop)
    if backend == 'linear':
        return SubdayStandaloneLinear(top_k_per_label=cfg.subday_top_k_per_label)
    if backend == 'extratrees':
        return SubdayStandaloneET(top_k_per_label=cfg.subday_top_k_per_label)
    raise ValueError(f"unknown backend: {backend!r}. Use 'lgbm', 'linear', or 'extratrees'.")


class OOFRunner:
    """Multi-seed K-fold OOF + multi-seed sub prediction with per-fold WB enrichment."""

    def __init__(self, cfg: PanelFXConfig, enrich_feats: list[str] | None=None, kfold_drop_regex: str | None='_bidir\\d+$'):
        self.cfg = cfg
        self.enrich_feats = enrich_feats
        self.kfold_drop_regex = kfold_drop_regex
        self.selection_log_: dict = {}
        self.oof_log_: dict = {}

    def _filter_kfold_cols(self, cols: Sequence[str]) -> list[str]:
        if not self.kfold_drop_regex:
            return list(cols)
        pat = re.compile(self.kfold_drop_regex)
        return [c for c in cols if not pat.search(c)]

    def _enrich_fold(self, tr_df: pd.DataFrame, other_dfs: list[pd.DataFrame], base_cols: list[str]) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
        if not self.enrich_feats:
            return (tr_df, other_dfs, base_cols)
        builder = WithinBetweenInputBuilder()
        tr_enriched = builder.fit_transform(tr_df, self.enrich_feats)
        others_enriched = [builder.transform(df) for df in other_dfs]
        enriched_base = base_cols + builder.added_cols_
        return (tr_enriched, others_enriched, enriched_base)


    def predict_sub(self, train_df: pd.DataFrame, sub_df: pd.DataFrame, base_cols: Sequence[str], subday_cols: Sequence[str], labels: Sequence[str], verbose: bool=True) -> np.ndarray:
        cfg = self.cfg
        labels = list(labels)
        n_sub = len(sub_df)
        n_train = len(train_df)
        all_preds = np.zeros((n_sub, len(labels)))
        n_models = 0
        for (si, seed) in enumerate(cfg.seeds_sub):
            if verbose:
                print(f'  [seed {seed}] sub-prediction K-fold')
            kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed)
            model = make_subday_standalone(cfg.model_backend, cfg)
            for (fold, (tr_idx, va_idx)) in enumerate(kf.split(np.arange(n_train))):
                tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
                va_df = train_df.iloc[va_idx].reset_index(drop=True)
                (tr_df, [va_df, sub_enriched], fold_base) = self._enrich_fold(tr_df, [va_df, sub_df], list(base_cols))
                for lab in labels:
                    model.fit_fold(tr_df, tr_df[lab], va_df, va_df[lab], fold_base, list(subday_cols), label=lab, fold_id=fold, seed=seed)
                    pred = model.predict_fold(sub_enriched, fold_base, label=lab, fold_id=fold, seed=seed)
                    all_preds[:, labels.index(lab)] += pred
                n_models += 1
        n_passes = len(cfg.seeds_sub) * cfg.n_splits
        return all_preds / n_passes


def _per_fold_augment(tr_df_raw: pd.DataFrame, va_df_raw: pd.DataFrame,
                       base_feat_cols: list, cfg: 'PanelFXConfig'):
    """Fit diag + builders on tr_df_raw only; transform both tr/va.

    Returns (tr_aug, va_aug, base_cols_aug, subday_cols, enrich_feats, fold_diag).
    """
    fold_diag = WithinBetweenDiagnoser(
        labels=cfg.labels, min_raw_r=cfg.min_raw_r,
        day_rich_ratio=cfg.day_rich_ratio, between_only_ratio=cfg.between_only_ratio,
    ).fit(tr_df_raw, base_feat_cols)

    if cfg.use_day_level:
        dlb = DayLevelInteractionBuilder(
            top_k_dayrich=cfg.day_level_top_k_pool,
            top_k_inter=cfg.day_level_top_k_per_label,
            labels=cfg.labels, min_raw_r=cfg.min_raw_r,
            day_rich_ratio=cfg.day_rich_ratio,
            between_only_ratio=cfg.between_only_ratio,
        ).fit(tr_df_raw, fold_diag)
        day_level_cols, tr_aug = dlb.transform(tr_df_raw)
        _, va_aug = dlb.transform(va_df_raw)
    else:
        day_level_cols, tr_aug, va_aug = [], tr_df_raw, va_df_raw

    if cfg.use_subday:
        sub_b = SubdayCrossBucketBuilder(
            mode=cfg.subday_selection_mode, top_k=cfg.subday_top_k, pinned=cfg.subday_pinned,
        ).fit(tr_aug, fold_diag)
        subday_cols, tr_aug = sub_b.transform(tr_aug)
        _, va_aug = sub_b.transform(va_aug)
    else:
        subday_cols = []

    day_rich = (fold_diag.df[fold_diag.df.category == 'day_rich']
                .assign(abs_r=lambda d: d.raw_r.abs())
                .sort_values('abs_r', ascending=False).feature.unique().tolist())
    enrich_feats = [f for f in day_rich if f in tr_aug.columns][:cfg.wb_enrichment_top_k]
    if os.environ.get('WB_FULL'):
        enrich_feats = [c for c in base_feat_cols if c in tr_aug.columns and np.issubdtype(tr_aug[c].dtype, np.number)
                        and tr_aug[c].nunique() > 5 and not (('_lag1' in c) or c.startswith('cal_')
                        or ('_subj_' in c) or ('_missing' in c) or ('_is_' in c) or c.endswith('is_weekend'))]

    return tr_aug, va_aug, base_feat_cols + day_level_cols, subday_cols, enrich_feats, fold_diag


def run_oof_per_fold(train_df_raw: pd.DataFrame, base_feat_cols: list,
                      cnn_oof_features: np.ndarray, cfg: 'PanelFXConfig',
                      labels: Sequence[str], verbose: bool = False) -> np.ndarray:
    """Per-fold leak-safe OOF. cnn_oof_features shape (n_train, n_labels), already mean-pooled across CNN seeds."""
    n_train = len(train_df_raw)
    L = len(labels)
    oof_seeds = np.zeros((len(cfg.seeds_oof), n_train, L), dtype=np.float32)
    sleep_cols = [f'cnn_{lab}' for lab in labels]
    for si, seed in enumerate(cfg.seeds_oof):
        if verbose:
            print(f'  [seed {seed}] per-fold OOF ({cfg.n_splits} folds)')
        if BASE_TEMPORAL:
            fold_splits_pf = subject_blocked_temporal_kfold(
                train_df_raw.subject_id.to_numpy(), train_df_raw.lifelog_date.to_numpy(), cfg.n_splits)
        else:
            fold_splits_pf = list(KFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed).split(np.arange(n_train)))
        model = make_subday_standalone(cfg.model_backend, cfg)
        for fold, (tr_idx, va_idx) in enumerate(fold_splits_pf):
            tr_df_raw = train_df_raw.iloc[tr_idx].reset_index(drop=True)
            va_df_raw = train_df_raw.iloc[va_idx].reset_index(drop=True)
            tr_aug, va_aug, base_cols, subday_cols, enrich_feats, _ = _per_fold_augment(
                tr_df_raw, va_df_raw, base_feat_cols, cfg,
            )
            if 'cnn' not in ABLATE:
                for li, c in enumerate(sleep_cols):
                    tr_aug[c] = cnn_oof_features[tr_idx, li]
                    va_aug[c] = cnn_oof_features[va_idx, li]
                base_cols = base_cols + sleep_cols
            if 'wb' in ABLATE:
                enrich_feats = []
            if enrich_feats:
                wb_b = WithinBetweenInputBuilder()
                tr_aug = wb_b.fit_transform(tr_aug, enrich_feats)
                va_aug = wb_b.transform(va_aug)
                base_cols = base_cols + wb_b.added_cols_
            for lab in labels:
                top_k, _ = model.fit_fold(
                    tr_aug, tr_aug[lab], va_aug, va_aug[lab],
                    base_cols, list(subday_cols),
                    label=lab, fold_id=fold, seed=seed,
                )
                pred = model.predict_fold(va_aug, base_cols, label=lab, fold_id=fold, seed=seed)
                oof_seeds[si, va_idx, labels.index(lab)] = pred
    return oof_seeds.mean(axis=0)

