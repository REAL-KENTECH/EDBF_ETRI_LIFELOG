"""wb.py — Within-Between (Mundlak) feature engineering.

Imported by features/tabular.py and stages/tabular_model.py to enrich the day-level features with
subject-mean and deviation columns before the per-label LGBM fits, plus the subday cross-bucket and
day-level interaction builders and the bucket utilities.
"""
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from dataclasses import dataclass, field
from typing import Sequence
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


def _vec_pearson(X_ranks: np.ndarray, y_ranks: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation r[i] = corr(X_ranks[:, i], y_ranks).

    With rank-transformed inputs this is equivalent to Spearman ρ, vectorized over feature columns.
    Returns NaN where the denominator is zero (constant column or constant y).
    """
    Xc = X_ranks - X_ranks.mean(axis=0)
    yc = y_ranks - y_ranks.mean()
    num = (Xc * yc[:, None]).sum(axis=0)
    den = np.sqrt((Xc * Xc).sum(axis=0) * (yc * yc).sum())
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


@dataclass

class WithinBetweenResult:
    """Output of WB diagnosis: per-feature category + within/between/raw correlations."""
    df: pd.DataFrame
    n_features: int
    n_labels: int
    config: dict = field(default_factory=dict)

class WithinBetweenDiagnoser:
    """Within-Between variance decomposition + per-label correlation diagnostics."""

    def __init__(self, labels: Sequence[str]=('Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4'), min_raw_r: float=0.1, day_rich_ratio: float=0.7, between_only_ratio: float=0.3, impute_strategy: str='median'):
        self.labels = tuple(labels)
        self.min_raw_r = min_raw_r
        self.day_rich_ratio = day_rich_ratio
        self.between_only_ratio = between_only_ratio
        self.impute_strategy = impute_strategy

    def fit(self, df: pd.DataFrame, feature_cols: Sequence[str]) -> WithinBetweenResult:
        feature_cols = list(feature_cols)
        self._check_inputs(df, feature_cols)
        raw_df = df[feature_cols].fillna(df[feature_cols].median()).fillna(0)
        sm_df = df.groupby('subject_id')[feature_cols].transform('mean')
        df_imp = df[feature_cols].fillna(df[feature_cols].median())
        dem_df = (df_imp - sm_df).fillna(0)
        raw_ranks = raw_df.rank().to_numpy()
        dem_ranks = dem_df.rank().to_numpy()
        rows = []
        for lab in self.labels:
            y = df[lab].astype(float)
            y_dem = y - df.groupby('subject_id')[lab].transform('mean')
            r_raw_all = _vec_pearson(raw_ranks, y.rank().to_numpy())
            r_wit_all = _vec_pearson(dem_ranks, y_dem.rank().to_numpy())
            for i, c in enumerate(feature_cols):
                r_raw, r_wit = r_raw_all[i], r_wit_all[i]
                if np.isnan(r_raw) or np.isnan(r_wit):
                    continue
                ratio = r_wit / r_raw if abs(r_raw) > 0.001 else 0.0
                cat = self._classify(r_raw, r_wit)
                rows.append({'feature': c, 'label': lab, 'raw_r': r_raw, 'within_r': r_wit, 'ratio': ratio, 'category': cat})
        return WithinBetweenResult(df=pd.DataFrame(rows), n_features=len(feature_cols), n_labels=len(self.labels), config={'min_raw_r': self.min_raw_r, 'day_rich_ratio': self.day_rich_ratio, 'between_only_ratio': self.between_only_ratio, 'impute_strategy': self.impute_strategy, 'labels': list(self.labels)})

    def _classify(self, r_raw: float, r_wit: float) -> str:
        if abs(r_raw) < self.min_raw_r:
            return 'mixed'
        if abs(r_wit) >= self.day_rich_ratio * abs(r_raw):
            return 'day_rich'
        if abs(r_wit) <= self.between_only_ratio * abs(r_raw):
            return 'between_only'
        return 'mixed'

    def _check_inputs(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        if 'subject_id' not in df.columns:
            raise KeyError("df must contain 'subject_id' column for within-subject demeaning")
        missing_labels = [l for l in self.labels if l not in df.columns]
        if missing_labels:
            raise KeyError(f'missing labels in df: {missing_labels}')
        missing_feats = [f for f in feature_cols if f not in df.columns]
        if missing_feats:
            raise KeyError(f'missing features in df: {missing_feats[:5]}' + (f'... ({len(missing_feats)} total)' if len(missing_feats) > 5 else ''))
BUCKETS: tuple[str, ...] = ('overnight', 'morning', 'afternoon', 'evening')

def is_bucket_feature(name: str) -> bool:
    """Return True if column name follows the subday bucket convention."""
    n = name.lower()
    return any((f'_bucket_{b}_' in n or f'_b_{b}_' in n for b in BUCKETS))

def strip_bucket(name: str) -> str:
    """Strip the trailing bucket suffix from a column name."""
    s = name
    for b in BUCKETS:
        s = s.replace(f'_bucket_{b}_', '_BKT_')
        s = s.replace(f'_b_{b}_', '_BKT_')
    return s.replace('_log1p', '')

def feature_bucket(name: str) -> str | None:
    """Return the bucket name from a feature column, or None if not a bucket."""
    n = name.lower()
    for b in BUCKETS:
        if b in n:
            return b
    return None

def derive_subday_dayrich(diag: WithinBetweenResult, top_k: int=15, per_label_top: int=20) -> list[str]:
    """Pick top-K data-driven subday features from WB diagnosis day-rich category."""
    feat_n_labels: Counter = Counter()
    feat_max_abs_r: dict[str, float] = {}
    for lab in diag.df.label.unique():
        sub = diag.df[(diag.df.label == lab) & (diag.df.category == 'day_rich') & diag.df.feature.apply(is_bucket_feature)].copy()
        sub['abs_r'] = sub.raw_r.abs()
        sub = sub.sort_values('abs_r', ascending=False).head(per_label_top)
        for (_, row) in sub.iterrows():
            feat_n_labels[row.feature] += 1
            feat_max_abs_r[row.feature] = max(feat_max_abs_r.get(row.feature, 0.0), float(row.abs_r))
    return sorted(feat_n_labels.keys(), key=lambda f: (-feat_n_labels[f], -feat_max_abs_r[f]))[:top_k]


class SubdayCrossBucketBuilder:
    """Build sub-day cross-bucket interaction features."""

    def __init__(self, mode: str='derived', top_k: int=15, pinned: Sequence[str]=()):
        if mode not in ('derived', 'pinned'):
            raise ValueError(f"mode must be 'derived' or 'pinned', got {mode!r}")
        self.mode = mode
        self.top_k = top_k
        self.pinned = list(pinned)
        self.derived_: list[str] = []
        self.subday_dayrich_: list[str] = []
        self.trace_: dict = {}

    def fit(self, df: pd.DataFrame, diag: WithinBetweenResult | None=None) -> 'SubdayCrossBucketBuilder':
        if self.mode == 'derived':
            if diag is None:
                raise ValueError("diag required for mode='derived'")
            self.derived_ = derive_subday_dayrich(diag, top_k=self.top_k)
            self.subday_dayrich_ = list(self.derived_)
        else:
            self.derived_ = []
            self.subday_dayrich_ = list(self.pinned)
        missing = [f for f in self.subday_dayrich_ if f not in df.columns]
        if missing:
            raise ValueError(f'selected subday_dayrich missing from df: {missing}')
        return self

    def transform(self, df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
        if not self.subday_dayrich_:
            raise RuntimeError('call fit() before transform()')
        bucket_cols = [c for c in df.columns if is_bucket_feature(c)]
        new_cols: dict[str, pd.Series] = {}
        for (f1, f2) in combinations(self.subday_dayrich_, 2):
            b1 = feature_bucket(f1)
            b2 = feature_bucket(f2)
            if b1 and b2 and (b1 != b2):
                new_cols[f'cross_mul_{f1}_X_{f2}'] = df[f1] * df[f2]
        n_mul = len(new_cols)
        sensor_groups: dict[str, list[str]] = {}
        for f in bucket_cols:
            sensor_groups.setdefault(strip_bucket(f), []).append(f)
        for related in sensor_groups.values():
            if len(related) < 2:
                continue
            for (f1, f2) in combinations(related, 2):
                new_cols[f'cross_diff_{f1}_MINUS_{f2}'] = df[f1] - df[f2]
        n_diff = len(new_cols) - n_mul
        df_aug = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_feat_names = list(new_cols.keys())
        (derived_set, pinned_set) = (set(self.derived_), set(self.pinned))
        overlap_jacc = len(derived_set & pinned_set) / max(1, len(derived_set | pinned_set)) if derived_set and pinned_set else None
        self.trace_ = {'mode': self.mode, 'subday_top_k': self.top_k, 'pinned': list(self.pinned), 'derived': list(self.derived_), 'used_subday_dayrich': list(self.subday_dayrich_), 'overlap_jaccard': overlap_jacc, 'derived_only': sorted(derived_set - pinned_set) if self.derived_ else [], 'pinned_only': sorted(pinned_set - derived_set) if self.derived_ else [], 'cross_mul_count': n_mul, 'cross_diff_count': n_diff, 'bucket_cols_count': len(bucket_cols), 'sensor_groups_count': len(sensor_groups)}
        return (new_feat_names, df_aug)

    def fit_transform(self, df: pd.DataFrame, diag: WithinBetweenResult | None=None) -> tuple[list[str], pd.DataFrame]:
        return self.fit(df, diag).transform(df)

class DayLevelInteractionBuilder:
    """Build day-level multivariate pairwise interaction features."""

    def __init__(self, top_k_dayrich: int=30, top_k_inter: int=5, labels: Sequence[str]=('Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4'), min_raw_r: float=0.1, day_rich_ratio: float=0.7, between_only_ratio: float=0.3):
        self.top_k_dayrich = top_k_dayrich
        self.top_k_inter = top_k_inter
        self.labels = tuple(labels)
        self.min_raw_r = min_raw_r
        self.day_rich_ratio = day_rich_ratio
        self.between_only_ratio = between_only_ratio
        self.pooled_: list[str] = []
        self.selected_: list[str] = []
        self.scaler_: StandardScaler | None = None
        self.impute_median_: pd.Series | None = None
        self.trace_: dict = {}

    def fit(self, train_df: pd.DataFrame, diag: WithinBetweenResult) -> 'DayLevelInteractionBuilder':
        top_per_label = {}
        for lab in self.labels:
            sub = diag.df[(diag.df.label == lab) & (diag.df.category == 'day_rich')].copy()
            sub['abs_raw'] = sub.raw_r.abs()
            top_per_label[lab] = sub.sort_values('abs_raw', ascending=False).head(20).feature.tolist()
        feat_count: Counter = Counter()
        for feats in top_per_label.values():
            for f in feats:
                feat_count[f] += 1
        self.pooled_ = [f for (f, _) in feat_count.most_common(self.top_k_dayrich)]
        self.impute_median_ = train_df[self.pooled_].median()
        X_tr = train_df[self.pooled_].fillna(self.impute_median_).fillna(0)
        self.scaler_ = StandardScaler().fit(X_tr)
        X_tr_std = pd.DataFrame(self.scaler_.transform(X_tr), columns=self.pooled_, index=train_df.index)
        pairs = list(combinations(self.pooled_, 2))
        inter_data = {f'{f1}__x__{f2}': X_tr_std[f1] * X_tr_std[f2] for (f1, f2) in pairs}
        inter_tr = pd.DataFrame(inter_data, index=train_df.index)
        inter_dem = inter_tr.copy()
        for c in inter_dem.columns:
            sm = inter_dem.groupby(train_df.subject_id)[c].transform('mean')
            inter_dem[c] = inter_tr[c] - sm
        rows = []
        for lab in self.labels:
            y = train_df[lab].astype(float)
            y_dem = y - train_df.groupby('subject_id')[lab].transform('mean')
            for c in inter_tr.columns:
                (r_raw, _) = spearmanr(inter_tr[c], y)
                (r_wit, _) = spearmanr(inter_dem[c], y_dem)
                if np.isnan(r_raw) or np.isnan(r_wit):
                    continue
                cat = 'mixed'
                if abs(r_raw) >= self.min_raw_r:
                    if abs(r_wit) >= self.day_rich_ratio * abs(r_raw):
                        cat = 'day_rich'
                    elif abs(r_wit) <= self.between_only_ratio * abs(r_raw):
                        cat = 'between_only'
                rows.append({'interaction': c, 'label': lab, 'raw_r': r_raw, 'within_r': r_wit, 'category': cat})
        day_level_b_diag = pd.DataFrame(rows)
        selected: set[str] = set()
        for lab in self.labels:
            sub = day_level_b_diag[(day_level_b_diag.label == lab) & (day_level_b_diag.category == 'day_rich')].copy()
            sub['abs_raw'] = sub.raw_r.abs()
            sub = sub.sort_values('abs_raw', ascending=False).head(self.top_k_inter)
            selected.update(sub['interaction'].tolist())
        self.selected_ = list(selected)
        self.trace_ = {'top_k_dayrich': self.top_k_dayrich, 'top_k_inter': self.top_k_inter, 'n_pooled': len(self.pooled_), 'n_pairs': len(pairs), 'n_diagnosed': len(day_level_b_diag), 'n_selected': len(self.selected_), 'category_counts': day_level_b_diag.category.value_counts().to_dict(), 'pooled': self.pooled_, 'selected': self.selected_}
        return self

    def transform(self, df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
        if self.scaler_ is None:
            raise RuntimeError('call fit() before transform()')
        X_full = df[self.pooled_].fillna(self.impute_median_).fillna(0)
        X_full_std = pd.DataFrame(self.scaler_.transform(X_full), columns=self.pooled_, index=df.index)
        inter_data = {c: X_full_std[c.split('__x__')[0]] * X_full_std[c.split('__x__')[1]] for c in self.selected_}
        inter_full = pd.DataFrame(inter_data, index=df.index)
        return (list(inter_full.columns), pd.concat([df, inter_full], axis=1))

    def fit_transform(self, train_df: pd.DataFrame, df_full: pd.DataFrame, diag: WithinBetweenResult) -> tuple[list[str], pd.DataFrame]:
        self.fit(train_df, diag)
        return self.transform(df_full)

class WithinBetweenInputBuilder:
    """Add within/between component columns for given features (leak-safe via fit/transform)."""

    def __init__(self, suffix_within: str='_w', suffix_between: str='_b', subject_col: str='subject_id', keep_original: bool=True, fallback_to_global: bool=True):
        self.suffix_within = suffix_within
        self.suffix_between = suffix_between
        self.subject_col = subject_col
        self.keep_original = keep_original
        self.fallback_to_global = fallback_to_global
        self.subject_means_: pd.DataFrame | None = None
        self.global_means_: pd.Series | None = None
        self.feat_cols_: list[str] = []
        self.added_cols_: list[str] = []

    def fit(self, train_df: pd.DataFrame, feat_cols: Sequence[str]) -> 'WithinBetweenInputBuilder':
        self.feat_cols_ = list(feat_cols)
        X = train_df[self.feat_cols_]
        self.global_means_ = X.median().fillna(0.0)
        X_imp = X.fillna(self.global_means_).fillna(0.0)
        df_imp = train_df[[self.subject_col]].assign(**{c: X_imp[c] for c in self.feat_cols_})
        self.subject_means_ = df_imp.groupby(self.subject_col)[self.feat_cols_].mean()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.subject_means_ is None:
            raise RuntimeError('call fit() before transform()')
        subj = df[self.subject_col]
        per_row_means = pd.DataFrame(index=df.index, columns=self.feat_cols_, dtype=float)
        for c in self.feat_cols_:
            per_row_means[c] = subj.map(self.subject_means_[c])
            if self.fallback_to_global:
                per_row_means[c] = per_row_means[c].fillna(self.global_means_[c])
        X = df[self.feat_cols_].copy()
        for c in self.feat_cols_:
            X[c] = X[c].fillna(per_row_means[c])
        within = X - per_row_means.values
        within.columns = [f'{c}{self.suffix_within}' for c in self.feat_cols_]
        between = per_row_means.copy()
        between.columns = [f'{c}{self.suffix_between}' for c in self.feat_cols_]
        self.added_cols_ = list(within.columns) + list(between.columns)
        if self.keep_original:
            return pd.concat([df, within, between], axis=1)
        else:
            df_drop = df.drop(columns=self.feat_cols_)
            return pd.concat([df_drop, within, between], axis=1)

    def fit_transform(self, train_df: pd.DataFrame, feat_cols: Sequence[str]) -> pd.DataFrame:
        self.fit(train_df, feat_cols)
        return self.transform(train_df)
