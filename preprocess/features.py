"""features.py — raw competition sensors (/data, or $ETRI_DATA) to model caches.

Outputs (cache/):
  features.parquet            day-level feature table plus the 7 labels
  sleep_window_tensor.npz     6-channel 00-09h sleep tensor

Read by modeling/stages/tabular_model.py and the encoders.

Run: python preprocess/features.py

PREPROCESS_VERBOSE=1 additionally prints the per-block missing-rate and distribution tables.
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 220)

DATA_DIR  = Path(os.environ.get('ETRI_DATA') or ('/data' if Path('/data/ch2026_submission_sample.csv').exists() else Path(__file__).resolve().parents[2] / 'data'))
ITEMS_DIR = DATA_DIR / 'ch2025_data_items'
OUT_DIR   = Path(__file__).resolve().parent.parent / 'cache'
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_SUFFIX = os.environ.get('RAW_OUT_SUFFIX', '')

# Diagnostic tables (missing rates, distributions, samples) are printed only when
# PREPROCESS_VERBOSE=1. `log` always prints, and carries the progress lines.
VERBOSE = os.environ.get('PREPROCESS_VERBOSE', '0') == '1'
log = print


def print(*args, **kwargs):
    if VERBOSE:
        log(*args, **kwargs)


assert DATA_DIR.exists()
assert ITEMS_DIR.exists()
print('DATA_DIR  =', DATA_DIR)
print('ITEMS_DIR =', ITEMS_DIR)
print('OUT_DIR   =', OUT_DIR)


train = pd.read_csv(DATA_DIR/'ch2026_metrics_train.csv', parse_dates=['sleep_date','lifelog_date'])
sub   = pd.read_csv(DATA_DIR/'ch2026_submission_sample.csv', parse_dates=['sleep_date','lifelog_date'])

train['split'] = 'train'
sub['split']   = 'sub'

all_keys = pd.concat([
    train[['subject_id','sleep_date','lifelog_date','split']],
    sub[['subject_id','sleep_date','lifelog_date','split']],
], ignore_index=True).sort_values(['subject_id','lifelog_date']).reset_index(drop=True)

assert not all_keys[['subject_id','lifelog_date']].duplicated().any(), '키 중복!'

print(f'train: {len(train)},  sub: {len(sub)},  total keys: {len(all_keys)}')
print(f'subjects: {sorted(all_keys.subject_id.unique())}')
print(f'lifelog_date: {all_keys.lifelog_date.min().date()} → {all_keys.lifelog_date.max().date()}')
all_keys.head(3)


TIME_BUCKETS = [
    ('overnight',  0,  6),
    ('morning',    6, 12),
    ('afternoon', 12, 18),
    ('evening',   18, 24),
]

def add_bucket(df, ts_col='timestamp'):
    """timestamp의 시(hour)를 보고 bucket 컬럼 반환. np.select로 명시적."""
    h = df[ts_col].dt.hour
    conditions = [h.between(lo, hi - 1) for _, lo, hi in TIME_BUCKETS]
    choices    = [name for name, _, _ in TIME_BUCKETS]
    return pd.Series(np.select(conditions, choices, default='unknown'),
                     index=df.index)

def restrict_to_lifelog_day(sensor_df, key_col='subject_id', ts_col='timestamp'):
    """
    센서 df를 라벨이 있는 (sid, lifelog_date) 키에 맞게 잘라내고
    'lifelog_date' + 'bucket' 컬럼을 붙인다.
    """
    df = sensor_df.copy()
    df['lifelog_date'] = df[ts_col].dt.normalize()
    keys = all_keys[['subject_id','lifelog_date']]
    df = df.merge(keys, on=['subject_id','lifelog_date'], how='inner')
    df['bucket'] = add_bucket(df, ts_col)
    return df

print('TIME_BUCKETS:', TIME_BUCKETS)
_t = pd.DataFrame({'timestamp': pd.to_datetime([
    '2024-06-26 03:00','2024-06-26 09:00','2024-06-26 15:00','2024-06-26 21:00'])})
_t['bucket'] = add_bucket(_t)
print(_t)


KEY = ['subject_id','lifelog_date']

def reindex_to_keys(feat_df):
    """피처 df를 all_keys 순서로 정렬·정합. 없는 (sid,day)는 NaN."""
    out = all_keys[KEY].merge(feat_df, on=KEY, how='left')
    return out

feature_blocks = {}

print('KEY columns =', KEY)
print('feature_blocks 초기화 (현재 0개)')




def scalar_day_stats(df, value_col, prefix):
    """
    df: restrict_to_lifelog_day() 결과 (subject_id, lifelog_date, bucket, value_col 포함)
    반환: (sid, lifelog_date)별 일일통계 + 버킷별 평균 wide 테이블
    """
    g = df.groupby(KEY)[value_col]
    daily = g.agg(
        mean='mean', std='std', min='min', max='max',
        p25=lambda s: s.quantile(0.25),
        p50='median',
        p75=lambda s: s.quantile(0.75),
    )
    daily.columns = [f'{prefix}_{c}' for c in daily.columns]

    bucket_mean = (df.groupby(KEY + ['bucket'])[value_col].mean()
                     .unstack('bucket'))
    for b, _, _ in TIME_BUCKETS:
        if b not in bucket_mean.columns:
            bucket_mean[b] = np.nan
    bucket_mean = bucket_mean[[b for b,_,_ in TIME_BUCKETS]]
    bucket_mean.columns = [f'{prefix}_bucket_{b}_mean' for b in bucket_mean.columns]

    out = daily.join(bucket_mean, how='outer').reset_index()
    return out

def missing_flag(df, prefix):
    """df는 restrict_to_lifelog_day 후. 키별 row 수를 보고 0인 키를 missing=1로."""
    has = df.groupby(KEY).size().rename(f'{prefix}_n_rows').reset_index()
    out = all_keys[KEY].merge(has, on=KEY, how='left')
    out[f'{prefix}_missing'] = out[f'{prefix}_n_rows'].isna().astype(int)
    return out[KEY + [f'{prefix}_missing']]

print('헬퍼 정의 완료: scalar_day_stats, missing_flag')


mlight_raw = pd.read_parquet(ITEMS_DIR/'ch2025_mLight.parquet')
wlight_raw = pd.read_parquet(ITEMS_DIR/'ch2025_wLight.parquet')
print(f'm_light raw: {len(mlight_raw):,}  w_light raw: {len(wlight_raw):,}')

mlight = restrict_to_lifelog_day(mlight_raw)
wlight = restrict_to_lifelog_day(wlight_raw)
print(f'after restrict to label keys → m_light: {len(mlight):,}  w_light: {len(wlight):,}')

m_stats = scalar_day_stats(mlight, 'm_light', 'm_light')
w_stats = scalar_day_stats(wlight, 'w_light', 'w_light')

w_miss = missing_flag(wlight, 'w_light')

A_block = (reindex_to_keys(m_stats)
             .merge(reindex_to_keys(w_stats), on=KEY, how='left')
             .merge(w_miss, on=KEY, how='left'))

feature_blocks['A_light'] = A_block
print(f'A_block shape: {A_block.shape}  (expected 700 × 25 = 23 features + 2 keys)')
print(f'columns: {[c for c in A_block.columns if c not in KEY]}')

print('=== 결측 비율 ===')
miss_rate = A_block.drop(columns=KEY).isna().mean().sort_values(ascending=False)
print(miss_rate.round(3).to_string())
print()

print('=== 분포 (요약) ===')
print(A_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(1).T[['count','mean','std','min','50%','95%','max']])
print()

print('=== sample (id01 첫 3일) ===')
print(A_block[A_block.subject_id=='id01'].head(3).to_string())




def binary_day_ratios(df, value_col, prefix):
    """
    df: restrict_to_lifelog_day 결과
    반환: (sid, lifelog_date)별 일일 ratio + 버킷별 ratio
    """
    daily = df.groupby(KEY)[value_col].mean().rename(f'{prefix}_ratio').reset_index()

    bucket_ratio = (df.groupby(KEY + ['bucket'])[value_col].mean()
                      .unstack('bucket'))
    for b, _, _ in TIME_BUCKETS:
        if b not in bucket_ratio.columns:
            bucket_ratio[b] = np.nan
    bucket_ratio = bucket_ratio[[b for b,_,_ in TIME_BUCKETS]]
    bucket_ratio.columns = [f'{prefix}_bucket_{b}_ratio' for b in bucket_ratio.columns]

    out = daily.merge(bucket_ratio.reset_index(), on=KEY, how='outer')
    return out

print('헬퍼 정의 완료: binary_day_ratios')


mchg_raw = pd.read_parquet(ITEMS_DIR/'ch2025_mACStatus.parquet')
mscr_raw = pd.read_parquet(ITEMS_DIR/'ch2025_mScreenStatus.parquet')

mchg = restrict_to_lifelog_day(mchg_raw)
mscr = restrict_to_lifelog_day(mscr_raw)

chg_feat = binary_day_ratios(mchg, 'm_charging',   'm_charging')
scr_feat = binary_day_ratios(mscr, 'm_screen_use', 'm_screen_use')

print('m_charging 피처:', [c for c in chg_feat.columns if c not in KEY])
print('m_screen_use 피처:', [c for c in scr_feat.columns if c not in KEY])
print()
print('샘플 (id01 첫 3일, screen):')
print(scr_feat[scr_feat.subject_id=='id01'].head(3).to_string())


ACTIVITY_CODES = {
    3: 'still',
    4: 'unknown',
    0: 'vehicle',
    7: 'walking',
}

mact_raw = pd.read_parquet(ITEMS_DIR/'ch2025_mActivity.parquet')
mact = restrict_to_lifelog_day(mact_raw)

mact['code_label'] = mact.m_activity.map(ACTIVITY_CODES).fillna('other')
code_ratio = (mact.groupby(KEY)['code_label'].value_counts(normalize=True)
                  .unstack('code_label').fillna(0))
wanted = list(ACTIVITY_CODES.values())
for c in wanted:
    if c not in code_ratio.columns:
        code_ratio[c] = 0.0
code_ratio = code_ratio[wanted].rename(columns={c: f'm_activity_{c}_ratio' for c in wanted})

mact['active'] = (mact.m_activity != 3).astype(int)
active_bucket = (mact.groupby(KEY + ['bucket'])['active'].mean()
                 .unstack('bucket'))
for b, _, _ in TIME_BUCKETS:
    if b not in active_bucket.columns:
        active_bucket[b] = np.nan
active_bucket = active_bucket[[b for b,_,_ in TIME_BUCKETS]]
active_bucket.columns = [f'm_activity_active_bucket_{b}_ratio' for b in active_bucket.columns]

act_feat = code_ratio.join(active_bucket, how='outer').reset_index()

print('m_activity 피처:', [c for c in act_feat.columns if c not in KEY])
print()
print('샘플 (id01 첫 3일):')
print(act_feat[act_feat.subject_id=='id01'].head(3).to_string())

B_block = (reindex_to_keys(chg_feat)
             .merge(reindex_to_keys(scr_feat), on=KEY, how='left')
             .merge(reindex_to_keys(act_feat), on=KEY, how='left'))

feature_blocks['B_phone_state'] = B_block

print(f'B_block shape: {B_block.shape}  (expected 700 × 20 = 18 features + 2 keys)')
print()
print('=== 결측 비율 ===')
print(B_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).to_string())
print()
print('=== 분포 ===')
pd.set_option('display.width', 220)
print(B_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(3).T[['count','mean','std','min','50%','95%','max']])




wpedo_raw = pd.read_parquet(ITEMS_DIR/'ch2025_wPedo.parquet')
print(f'wPedo raw: {len(wpedo_raw):,}')
print(f'  running_step max = {wpedo_raw.running_step.max()}, '
      f'walking_step max = {wpedo_raw.walking_step.max()}  (둘 다 0 → 제외)')

wpedo = restrict_to_lifelog_day(wpedo_raw)

sum_cols = ['step', 'distance', 'burned_calories']
daily_sum = wpedo.groupby(KEY)[sum_cols].sum()
daily_sum.columns = ['wpedo_step_sum', 'wpedo_distance_sum', 'wpedo_cal_sum']

intensity = wpedo.groupby(KEY).agg(
    wpedo_step_freq_mean=('step_frequency', 'mean'),
    wpedo_step_freq_max =('step_frequency', 'max'),
    wpedo_speed_mean    =('speed', 'mean'),
    wpedo_speed_max     =('speed', 'max'),
)

wpedo['active'] = (wpedo.step > 0).astype(int)
active_min = wpedo.groupby(KEY)['active'].sum().rename('wpedo_active_min').to_frame()

step_bucket = (wpedo.groupby(KEY + ['bucket'])['step'].sum().unstack('bucket'))
for b, _, _ in TIME_BUCKETS:
    if b not in step_bucket.columns:
        step_bucket[b] = np.nan
step_bucket = step_bucket[[b for b,_,_ in TIME_BUCKETS]]
step_bucket.columns = [f'wpedo_step_bucket_{b}_sum' for b in step_bucket.columns]

wpedo_miss = missing_flag(wpedo, 'wpedo')

wpedo_feat = (daily_sum
                .join(intensity)
                .join(active_min)
                .join(step_bucket)
                .reset_index())

C_block = (reindex_to_keys(wpedo_feat)
             .merge(wpedo_miss, on=KEY, how='left'))
feature_blocks['C_wpedo'] = C_block

print(f'C_block shape: {C_block.shape}  (expected 700 × 15 = 13 features + 2 keys)')
print()
print('=== 결측 비율 ===')
print(C_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).to_string())
print()
print('=== 분포 ===')
print(C_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(2).T[['count','mean','std','min','50%','95%','max']])
print()
print('=== id01 첫 3일 ===')
print(C_block[C_block.subject_id=='id01'].head(3).to_string())




HR_MIN_PHYSIO, HR_MAX_PHYSIO = 30, 200

def _clean_hr_arr(arr):
    """raw heart_rate list → ndarray of physiologically plausible BPM only."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return a
    return a[(a >= HR_MIN_PHYSIO) & (a <= HR_MAX_PHYSIO)]

def _clean_hr_mean(arr):
    """분단위 평균 — outlier 제외 후 mean. n<3이면 NaN."""
    c = _clean_hr_arr(arr)
    return float(np.mean(c)) if c.size >= 3 else np.nan

whr_raw = pd.read_parquet(ITEMS_DIR/'ch2025_wHr.parquet')
print(f'wHr raw rows: {len(whr_raw):,}')

whr = restrict_to_lifelog_day(whr_raw)
print(f'restricted: {len(whr):,}')

whr['hr_min_mean'] = whr.heart_rate.map(_clean_hr_mean)

def rmssd(s):
    """successive 차이의 RMS. n<2면 NaN. NaN 값(필터 결과)은 제외 후 계산."""
    arr = np.asarray(s, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.nan
    return float(np.sqrt(np.mean(np.diff(arr)**2)))

whr_sorted = whr.sort_values(['subject_id','timestamp'])
hrv = (whr_sorted.groupby(KEY)['hr_min_mean']
        .apply(rmssd).rename('whr_rmssd_minute').to_frame())

n_min = whr.groupby(KEY).size().rename('whr_n_min').to_frame()

bucket_mean = (whr.groupby(KEY + ['bucket'])['hr_min_mean'].mean().unstack('bucket'))
for b, _, _ in TIME_BUCKETS:
    if b not in bucket_mean.columns:
        bucket_mean[b] = np.nan
bucket_mean = bucket_mean[[b for b,_,_ in TIME_BUCKETS]]
bucket_mean.columns = [f'whr_bucket_{b}_mean' for b in bucket_mean.columns]

print('explode 중...')
expl = whr[KEY + ['heart_rate']].explode('heart_rate')
expl['heart_rate'] = pd.to_numeric(expl['heart_rate'], errors='coerce')
n_before = len(expl)
expl = expl.dropna(subset=['heart_rate'])
expl = expl[(expl['heart_rate'] >= HR_MIN_PHYSIO) &
            (expl['heart_rate'] <= HR_MAX_PHYSIO)]
n_after = len(expl)
n_dropped_outlier = n_before - n_after
print(f'expanded rows: {n_before:,}  →  physio-filtered: {n_after:,}  '
      f'(outlier dropped: {n_dropped_outlier:,}, '
      f'{100*n_dropped_outlier/max(n_before,1):.2f}%)')

g = expl.groupby(KEY)['heart_rate']
daily = g.agg(
    whr_mean='mean', whr_std='std', whr_min='min', whr_max='max',
    whr_p25=lambda s: s.quantile(0.25),
    whr_p75=lambda s: s.quantile(0.75),
)

whr_miss = missing_flag(whr, 'whr')

whr_feat = (daily.join(bucket_mean, how='outer')
                 .join(hrv, how='outer')
                 .join(n_min, how='outer')
                 .reset_index())

D_block = (reindex_to_keys(whr_feat)
             .merge(whr_miss, on=KEY, how='left'))
feature_blocks['D_heart_rate'] = D_block

print(f'\nD_block shape: {D_block.shape}  (expected 700 × 15 = 13 features + 2 keys)')
print()
print('=== 결측 ===')
print(D_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).to_string())
print()
print('=== 분포 (physio-filtered [30, 200]) ===')
print(D_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(2).T[['count','mean','std','min','50%','95%','max']])
print()
print('=== id01 첫 3일 ===')
print(D_block[D_block.subject_id=='id01'].head(3).to_string())
print()
print(f'note: HR_MIN_PHYSIO={HR_MIN_PHYSIO}, HR_MAX_PHYSIO={HR_MAX_PHYSIO} — [S2] 텐서와 동일 기준')




amb_raw = pd.read_parquet(ITEMS_DIR/'ch2025_mAmbience.parquet')
print(f'mAmbience raw: {len(amb_raw):,}')

amb = restrict_to_lifelog_day(amb_raw)
print(f'restricted: {len(amb):,}')

GROUPS = {
    'silence':     ['Silence'],
    'speech':      ['Speech','Conversation','Narration, monologue','Child speech, kid speaking',
                    'Babbling','Speech synthesizer','Shout','Whispering',
                    'Female speech, woman speaking','Male speech, man speaking'],
    'music':       ['Music','Musical instrument','Song','Singing'],
    'media':       ['Television','Radio'],
    'vehicle':     ['Vehicle','Car','Motor vehicle (road)','Engine','Bicycle','Bus','Motorcycle','Train',
                    'Motorboat, speedboat','Heavy engine (low frequency)','Medium engine (mid frequency)',
                    'Accelerating, revving, vroom'],
    'indoor':      ['Inside, small room','Inside, large room or hall','Inside, public space'],
    'sleep_sound': ['Snoring','Breathing','Sleep'],
    'appliance':   ['Mechanical fan','Air conditioning','Vacuum cleaner','Water tap, faucet',
                    'Electric shaver, electric razor','Hair dryer','Spray','Sliding door',
                    'Microwave oven','Blender'],
    'noise':       ['White noise','Buzz','Hiss','Rustle','Crackle','Static'],
}
CLASS_TO_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}

expl = amb[KEY + ['timestamp','m_ambience']].explode('m_ambience').reset_index(drop=True)
expl = expl[expl.m_ambience.notna()]
expl['class'] = expl.m_ambience.map(lambda x: str(x[0]))
expl['score'] = expl.m_ambience.map(lambda x: float(x[1]))
expl['group'] = expl['class'].map(CLASS_TO_GROUP)
expl = expl[expl.group.notna()]
hh = expl.timestamp.dt.hour
expl['is_sleep']   = hh < 9
expl['is_pre_bed'] = hh.between(21, 23)
print(f'expanded (group 매칭만): {len(expl):,}')

def per_row_group_score(df, prefix):
    """row × group → max score per (row, group). row에 group 없으면 0.
    그 후 daily mean = "평균 row당 group 강도"."""
    row_max = (df.groupby(KEY + ['timestamp', 'group'])['score'].max()
                 .unstack('group').fillna(0))
    daily = row_max.groupby(level=KEY).mean()
    daily.columns = [f'{prefix}_{g}_score' for g in daily.columns]
    return daily

day_score    = per_row_group_score(expl,                    'amb_day')
sleep_score  = per_row_group_score(expl[expl.is_sleep],     'amb_sleep')
prebed_score = per_row_group_score(expl[expl.is_pre_bed],   'amb_prebed')

amb_t1 = amb.copy()
amb_t1['top1'] = amb_t1.m_ambience.map(lambda x: str(x[0][0]) if len(x) > 0 else None)
def _ent(s):
    p = s.value_counts(normalize=True)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0
top1_ent = amb_t1.groupby(KEY)['top1'].apply(_ent).rename('amb_top1_entropy')

KEEP_DAY    = ['silence','speech','music','indoor','vehicle']
KEEP_SLEEP  = ['silence','speech','sleep_sound','indoor','noise','appliance']
KEEP_PREBED = ['silence','speech','music','media']

day_cols    = [f'amb_day_{g}_score'    for g in KEEP_DAY    if f'amb_day_{g}_score'    in day_score.columns]
sleep_cols  = [f'amb_sleep_{g}_score'  for g in KEEP_SLEEP  if f'amb_sleep_{g}_score'  in sleep_score.columns]
prebed_cols = [f'amb_prebed_{g}_score' for g in KEEP_PREBED if f'amb_prebed_{g}_score' in prebed_score.columns]

amb_feat = pd.concat([
    day_score[day_cols],
    sleep_score[sleep_cols],
    prebed_score[prebed_cols],
    top1_ent,
], axis=1).reset_index()

E_block = reindex_to_keys(amb_feat)
feature_blocks['E_ambience'] = E_block

print(f'\nE_block shape: {E_block.shape}  (이전 6 → 신규 16 features + 2 keys)')
print()
print('=== 결측 ===')
print(E_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).to_string())
print()
print('=== 분포 ===')
print(E_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(3).T[['count','mean','std','min','50%','95%','max']])




KEEP_GROUPS_BUCKET = ['silence', 'speech', 'music', 'vehicle', 'indoor', 'sleep_sound', 'appliance']

amb_e2 = restrict_to_lifelog_day(amb_raw)
expl_e2 = (amb_e2[KEY + ['timestamp', 'bucket', 'm_ambience']]
              .explode('m_ambience').reset_index(drop=True))
expl_e2 = expl_e2[expl_e2.m_ambience.notna()]
expl_e2['cls']   = expl_e2.m_ambience.map(lambda x: str(x[0]))
expl_e2['score'] = expl_e2.m_ambience.map(lambda x: float(x[1]))
expl_e2['group'] = expl_e2['cls'].map(CLASS_TO_GROUP)
expl_e2 = expl_e2[expl_e2.group.notna()]
print(f'expanded (group 매칭): {len(expl_e2):,}')

row_max_e2 = (expl_e2.groupby(KEY + ['timestamp', 'bucket', 'group'])['score'].max()
                      .unstack('group').fillna(0))
bucket_score = row_max_e2.groupby(level=KEY + ['bucket']).mean()
keep = [g for g in KEEP_GROUPS_BUCKET if g in bucket_score.columns]
bucket_score = bucket_score[keep]

wide = bucket_score.unstack('bucket')
ordered_pairs = []
for b, _, _ in TIME_BUCKETS:
    for g in keep:
        if (g, b) in wide.columns:
            ordered_pairs.append((g, b))
wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered_pairs, names=['group', 'bucket']))
wide.columns = [f'amb_b_{b}_{g}_score' for g, b in wide.columns]
wide = wide.reset_index()

E2_block = reindex_to_keys(wide)
feature_blocks['E2_ambience_bucket'] = E2_block

print(f'\nE2_block shape: {E2_block.shape}')
print()
print('=== 결측 (top 12) ===')
print(E2_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).head(12).to_string())
print()
print('=== 분포 (top 12) ===')
print(E2_block.drop(columns=KEY).describe(percentiles=[.5, .95]).round(3).T[
    ['count', 'mean', 'std', 'min', '50%', '95%', 'max']].head(12))
print()
print('의미:')
print('  overnight (00-06h): silence/sleep_sound 우세 → 수면 환경 quality')
print('  morning   (06-12h): speech/vehicle → 출근 / 활동 시작')
print('  afternoon (12-18h): music/indoor → 낮 활동 패턴')
print('  evening   (18-24h): silence/speech/music → 취침 직전 자극 정도')




import time as _t

gps_raw = pd.read_parquet(ITEMS_DIR/'ch2025_mGps.parquet')
print(f'mGps raw: {len(gps_raw):,}')

gps = restrict_to_lifelog_day(gps_raw)
print(f'restricted: {len(gps):,}')

t0 = _t.time()
expl = gps[KEY + ['timestamp', 'm_gps']].explode('m_gps').reset_index(drop=True)
expl = expl.dropna(subset=['m_gps'])
gps_pts = pd.json_normalize(expl['m_gps'].tolist())
for _c in [*KEY, 'timestamp']:
    gps_pts[_c] = expl[_c].reset_index(drop=True).values
print(f'explode + normalize: {_t.time()-t0:.1f}s, points={len(gps_pts):,}')

g = gps_pts.groupby(KEY)
gps_basic = g.agg(
    gps_speed_mean=('speed', 'mean'),
    gps_speed_max =('speed', 'max'),
    gps_speed_std =('speed', 'std'),
    gps_lat_std   =('latitude',  'std'),
    gps_lon_std   =('longitude', 'std'),
    gps_alt_std   =('altitude',  'std'),
    gps_n_points  =('speed', 'size'),
)
stationary = (gps_pts.assign(stat=(gps_pts.speed < 0.1).astype(int))
                     .groupby(KEY)['stat'].mean()
                     .rename('gps_stationary_ratio')
                     .to_frame())

GRID = 0.001
gps_pts['lat_bin'] = (gps_pts['latitude']  / GRID).round().astype(int)
gps_pts['lon_bin'] = (gps_pts['longitude'] / GRID).round().astype(int)
gps_pts['hour']    = gps_pts['timestamp'].dt.hour

gps_pts_split = gps_pts.merge(all_keys[KEY + ['split']], on=KEY, how='left')
night = gps_pts_split[gps_pts_split.hour < 6]

def _gps_anchor(df):
    return (df.groupby(['subject_id','lat_bin','lon_bin']).size()
              .reset_index(name='_n')
              .sort_values(['subject_id','_n'], ascending=[True, False])
              .drop_duplicates('subject_id', keep='first')
              [['subject_id','lat_bin','lon_bin']])

anchor_train = _gps_anchor(night[night.split == 'train'])
anchor_full  = _gps_anchor(night)
audit = anchor_train.merge(anchor_full, on='subject_id', suffixes=('_train','_full'))
diff_n = ((audit['lat_bin_train'] != audit['lat_bin_full']) |
          (audit['lon_bin_train'] != audit['lon_bin_full'])).sum()
print(f'[gps] home anchor audit: {diff_n}/{len(audit)} subjects differ (train-only vs full)')

home_anchor = anchor_train.rename(columns={'lat_bin':'home_lat_bin','lon_bin':'home_lon_bin'})
print(f'home anchor (train-only): {len(home_anchor)} subjects')

gps_pts = gps_pts.merge(home_anchor, on='subject_id', how='left')
gps_pts['at_home'] = ((gps_pts.lat_bin == gps_pts.home_lat_bin) &
                      (gps_pts.lon_bin == gps_pts.home_lon_bin)).astype(int)

home_daily   = gps_pts.groupby(KEY)['at_home'].mean().rename('gps_home_ratio')
home_evening = (gps_pts[gps_pts.hour.between(18, 23)]
                  .groupby(KEY)['at_home'].mean().rename('gps_evening_home_ratio'))
home_overnight = (gps_pts[gps_pts.hour < 6]
                  .groupby(KEY)['at_home'].mean().rename('gps_overnight_home_ratio'))

def _loc_ent(d):
    if len(d) == 0: return 0.0
    bins = pd.Series(list(zip(d['lat_bin'], d['lon_bin']))).value_counts(normalize=True)
    bins = bins[bins > 0]
    return float(-(bins * np.log(bins)).sum())
loc_ent = (gps_pts.groupby(KEY)[['lat_bin','lon_bin']]
                  .apply(_loc_ent).rename('gps_location_entropy'))

n_locs = (gps_pts.groupby(KEY)[['lat_bin','lon_bin']]
                  .apply(lambda d: int(d.drop_duplicates().shape[0]))
                  .rename('gps_unique_locations'))

gps_feat = pd.concat([
    gps_basic, stationary,
    home_daily, home_evening, home_overnight,
    loc_ent, n_locs,
], axis=1).reset_index()

gps_miss = missing_flag(gps, 'gps')
F_block = (reindex_to_keys(gps_feat)
             .merge(gps_miss, on=KEY, how='left'))
feature_blocks['F_gps'] = F_block

print(f'\nF_block shape: {F_block.shape}')
print('\n=== 결측 ===')
print(F_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).to_string())
print('\n=== 분포 ===')
print(F_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(2).T[['count','mean','std','min','50%','95%','max']])




wifi_raw = pd.read_parquet(ITEMS_DIR/'ch2025_mWifi.parquet')
print(f'mWifi raw: {len(wifi_raw):,}')

wifi = restrict_to_lifelog_day(wifi_raw)
print(f'restricted: {len(wifi):,}')

wifi['n_bssid'] = wifi.m_wifi.map(len)
n_scans         = wifi.groupby(KEY).size().rename('wifi_n_scans')
bssids_per_scan = wifi.groupby(KEY)['n_bssid'].mean().rename('wifi_bssids_per_scan_mean')

expl = wifi[KEY + ['timestamp', 'm_wifi']].explode('m_wifi').reset_index(drop=True)
expl = expl.dropna(subset=['m_wifi'])
pts = pd.json_normalize(expl.m_wifi.tolist())
for _c in [*KEY, 'timestamp']:
    pts[_c] = expl[_c].reset_index(drop=True).values
print(f'expanded BSSID rows: {len(pts):,}')

n_rssi_zero = int((pts['rssi'] == 0).sum())
pts.loc[pts['rssi'] == 0, 'rssi'] = np.nan
print(f'  rssi==0 sentinel → NaN: {n_rssi_zero:,} rows '
      f'({100*n_rssi_zero/max(len(pts),1):.2f}%)')

unique_bssid = pts.groupby(KEY)['bssid'].nunique().rename('wifi_unique_bssid_total')
rssi_stats = pts.groupby(KEY)['rssi'].agg(
    wifi_rssi_mean='mean', wifi_rssi_max='max', wifi_rssi_std='std',
)
top_bssid_count = pts.groupby(KEY)['bssid'].agg(
    lambda s: s.value_counts().iloc[0] if len(s) else 0
).rename('top_bssid_count')
top_ratio = (top_bssid_count / n_scans).rename('wifi_top_bssid_ratio')

pts['hour'] = pts['timestamp'].dt.hour
pts_split = pts.merge(all_keys[KEY + ['split']], on=KEY, how='left')
night = pts_split[pts_split.hour < 6]

def _wifi_anchor(df):
    return (df.groupby(['subject_id','bssid']).size()
              .reset_index(name='_n')
              .sort_values(['subject_id','_n'], ascending=[True, False])
              .drop_duplicates('subject_id', keep='first')
              [['subject_id','bssid']])

anchor_train = _wifi_anchor(night[night.split == 'train'])
anchor_full  = _wifi_anchor(night)
audit = anchor_train.merge(anchor_full, on='subject_id', suffixes=('_train','_full'))
diff_n = (audit['bssid_train'] != audit['bssid_full']).sum()
print(f'[wifi] home anchor audit: {diff_n}/{len(audit)} subjects differ (train-only vs full)')

home_bssid = anchor_train.rename(columns={'bssid':'home_bssid'})
print(f'home BSSID (train-only): {len(home_bssid)} subjects')

pts2 = pts.merge(home_bssid, on='subject_id', how='left')
pts2['at_home'] = (pts2.bssid == pts2.home_bssid).astype(int)

scan_home = pts2.groupby([*KEY, 'timestamp']).agg(
    at_home=('at_home', 'max'),
    hour=('hour', 'first'),
).reset_index()

home_daily = scan_home.groupby(KEY)['at_home'].mean().rename('wifi_home_scan_ratio')
home_evening = (scan_home[scan_home.hour.between(18, 23)]
                  .groupby(KEY)['at_home'].mean()
                  .rename('wifi_home_evening_ratio'))
home_overnight = (scan_home[scan_home.hour < 6]
                  .groupby(KEY)['at_home'].mean()
                  .rename('wifi_home_overnight_ratio'))

wifi_feat = (pd.concat([n_scans, unique_bssid, bssids_per_scan, rssi_stats, top_ratio,
                         home_daily, home_evening, home_overnight], axis=1)
                .reset_index())

wifi_miss = missing_flag(wifi, 'wifi')

G_block = (reindex_to_keys(wifi_feat)
             .merge(wifi_miss, on=KEY, how='left'))
feature_blocks['G_wifi'] = G_block

print(f'\nG_block shape: {G_block.shape}')
print('\n=== 결측 ===')
print(G_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).to_string())
print('\n=== 분포 ===')
print(G_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(2).T[['count','mean','std','min','50%','95%','max']])




ble_raw = pd.read_parquet(ITEMS_DIR/'ch2025_mBle.parquet')
print(f'mBle raw: {len(ble_raw):,}')

ble = restrict_to_lifelog_day(ble_raw)
print(f'restricted: {len(ble):,}')

ble['n_dev'] = ble.m_ble.map(len)
n_scans          = ble.groupby(KEY).size().rename('ble_n_scans')
devices_per_scan = ble.groupby(KEY)['n_dev'].mean().rename('ble_devices_per_scan_mean')

expl = ble[KEY + ['timestamp', 'm_ble']].explode('m_ble').reset_index(drop=True)
expl = expl.dropna(subset=['m_ble'])
expl = expl[expl.m_ble.map(lambda x: isinstance(x, dict) or hasattr(x, '__getitem__'))]
pts = pd.json_normalize(expl.m_ble.tolist())
for _c in [*KEY, 'timestamp']:
    pts[_c] = expl[_c].reset_index(drop=True).values
print(f'expanded device rows: {len(pts):,}')

unique_addr = pts.groupby(KEY)['address'].nunique().rename('ble_unique_addr_total')
rssi_mean   = pts.groupby(KEY)['rssi'].mean().rename('ble_rssi_mean')

pts['hour'] = pts['timestamp'].dt.hour
pts_split = pts.merge(all_keys[KEY + ['split']], on=KEY, how='left')
night = pts_split[pts_split.hour < 6]

def _ble_anchor(df):
    return (df.groupby(['subject_id','address']).size()
              .reset_index(name='_n')
              .sort_values(['subject_id','_n'], ascending=[True, False])
              .drop_duplicates('subject_id', keep='first')
              [['subject_id','address']])

anchor_train = _ble_anchor(night[night.split == 'train'])
anchor_full  = _ble_anchor(night)
audit = anchor_train.merge(anchor_full, on='subject_id', suffixes=('_train','_full'))
diff_n = (audit['address_train'] != audit['address_full']).sum()
print(f'[ble] home anchor audit: {diff_n}/{len(audit)} subjects differ (train-only vs full)')

home_ble = anchor_train.rename(columns={'address':'home_ble_addr'})
print(f'home BLE addr (train-only): {len(home_ble)} subjects')

pts2 = pts.merge(home_ble, on='subject_id', how='left')
pts2['at_home'] = ((pts2.address == pts2.home_ble_addr) &
                    pts2.home_ble_addr.notna()).astype(int)

scan_home = pts2.groupby([*KEY, 'timestamp']).agg(
    at_home=('at_home','max'),
    hour=('hour','first'),
).reset_index()

home_daily   = scan_home.groupby(KEY)['at_home'].mean().rename('ble_home_scan_ratio')
home_evening = (scan_home[scan_home.hour.between(18, 23)]
                  .groupby(KEY)['at_home'].mean()
                  .rename('ble_home_evening_ratio'))
home_overnight = (scan_home[scan_home.hour < 6]
                  .groupby(KEY)['at_home'].mean()
                  .rename('ble_home_overnight_ratio'))

ble_feat = pd.concat([n_scans, unique_addr, devices_per_scan, rssi_mean,
                      home_daily, home_evening, home_overnight], axis=1).reset_index()

ble_miss = missing_flag(ble, 'ble')

H_block = (reindex_to_keys(ble_feat)
             .merge(ble_miss, on=KEY, how='left'))
feature_blocks['H_ble'] = H_block

print(f'\nH_block shape: {H_block.shape}')
print('\n=== 결측 ===')
print(H_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).to_string())
print('\n=== 분포 ===')
print(H_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(2).T[['count','mean','std','min','50%','95%','max']])




usg_raw = pd.read_parquet(ITEMS_DIR/'ch2025_mUsageStats.parquet')
print(f'mUsageStats raw: {len(usg_raw):,}')

usg = restrict_to_lifelog_day(usg_raw)
print(f'restricted: {len(usg):,}')

def _sum_total(lst):
    return sum(d.get('total_time', 0) for d in lst) if lst is not None else 0
usg['row_sum_ms'] = usg.m_usage_stats.map(_sum_total)
h_ts = usg.timestamp.dt.hour
usg['is_pre_bed']   = h_ts.between(21, 23)
usg['is_late']      = h_ts.between(22, 23)

expl = usg[KEY + ['m_usage_stats','bucket','is_pre_bed','is_late']].explode('m_usage_stats').reset_index(drop=True)
expl = expl.dropna(subset=['m_usage_stats'])
pts = pd.json_normalize(expl.m_usage_stats.tolist())
for _c in [*KEY, 'bucket', 'is_pre_bed', 'is_late']:
    pts[_c] = expl[_c].reset_index(drop=True).values
print(f'expanded app-level rows: {len(pts):,}')

daily_total = usg.groupby(KEY)['row_sum_ms'].sum().rename('usage_total_time_ms')
n_records   = usg.groupby(KEY).size().rename('usage_n_records')
unique_apps = pts.groupby(KEY)['app_name'].nunique().rename('usage_unique_apps')

bucket_total = (usg[usg.bucket.isin(['overnight','evening'])]
                  .groupby(KEY + ['bucket'])['row_sum_ms'].sum()
                  .unstack('bucket'))
for b in ['overnight','evening']:
    if b not in bucket_total.columns:
        bucket_total[b] = np.nan
bucket_total = bucket_total[['overnight','evening']]
bucket_total.columns = [f'usage_total_time_bucket_{b}_ms' for b in bucket_total.columns]

pre_bed_total = (usg[usg.is_pre_bed].groupby(KEY)['row_sum_ms'].sum()
                  .rename('usage_pre_bed_time_ms'))
pre_bed_apps  = (pts[pts.is_pre_bed].groupby(KEY)['app_name'].nunique()
                  .rename('usage_pre_bed_unique_apps'))

late_total = (usg[usg.is_late].groupby(KEY)['row_sum_ms'].sum()
                .rename('usage_late_night_time_ms'))

day_app_tt = pts.groupby(KEY + ['app_name'])['total_time'].sum().rename('_app_tt')

def _top_share(s):
    tot = s.sum()
    return float(s.max() / tot) if tot > 0 else 0.0

def _shannon(s):
    tot = s.sum()
    if tot <= 0:
        return 0.0
    p = s / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

day_grp = day_app_tt.groupby(KEY)
top_app_share = day_grp.apply(_top_share).rename('usage_top_app_share')
app_entropy   = day_grp.apply(_shannon).rename('usage_app_entropy')

usg_feat = pd.concat([
    daily_total, n_records, unique_apps,
    bucket_total,
    pre_bed_total, pre_bed_apps, late_total,
    top_app_share, app_entropy,
], axis=1).reset_index()

usg_miss = missing_flag(usg, 'usage')

I_block = (reindex_to_keys(usg_feat)
             .merge(usg_miss, on=KEY, how='left'))
feature_blocks['I_usage'] = I_block

print(f'\nI_block shape: {I_block.shape}  (이전 8 → 신규 11 features + 2 keys)')
print()
print('=== 결측 ===')
print(I_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).to_string())
print()
print('=== 분포 ===')
print(I_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(2).T[['count','mean','std','min','50%','95%','max']])
print()
print('=== 단위 안내 ===')
print('  *_ms  = milliseconds. 1 hour = 3,600,000 ms.')
print('  pre_bed (21-23h) / late_night (22-23h) = S 라벨 sleep 직접 신호 후보')
print('  top_app_share, app_entropy = 폰 사용 다양성 (낮은 entropy = 집중적 한 앱 의존)')




def evening_split_stats(sensor_df, value_col, prefix):
    """
    각 sensor의 18-20, 21-23 두 창에서 mean/max 4개 stat 계산.
    sensor_df: raw parquet (timestamp 컬럼 있음)
    """
    s = restrict_to_lifelog_day(sensor_df)
    h = s['timestamp'].dt.hour
    early = s[h.between(18, 20)]
    pre   = s[h.between(21, 23)]

    out = all_keys[KEY].copy()
    if len(early) > 0:
        e = early.groupby(KEY)[value_col].agg(
            **{f'{prefix}_early_evening_mean': 'mean',
               f'{prefix}_early_evening_max': 'max'})
        out = out.merge(e.reset_index(), on=KEY, how='left')
    else:
        out[f'{prefix}_early_evening_mean'] = np.nan
        out[f'{prefix}_early_evening_max']  = np.nan

    if len(pre) > 0:
        p = pre.groupby(KEY)[value_col].agg(
            **{f'{prefix}_pre_bed_mean': 'mean',
               f'{prefix}_pre_bed_max': 'max'})
        out = out.merge(p.reset_index(), on=KEY, how='left')
    else:
        out[f'{prefix}_pre_bed_mean'] = np.nan
        out[f'{prefix}_pre_bed_max']  = np.nan
    return out

J_block = all_keys[KEY].copy()
for raw, value_col, prefix in [
    (mlight_raw, 'm_light',      'm_light'),
    (wlight_raw, 'w_light',      'w_light'),
    (mscr_raw,   'm_screen_use', 'm_screen_use'),
    (mchg_raw,   'm_charging',   'm_charging'),
]:
    block = evening_split_stats(raw, value_col, prefix)
    J_block = J_block.merge(block, on=KEY, how='left')

feature_blocks['J_pre_bed'] = J_block

print(f'J_block shape: {J_block.shape}  (expected 700 × 18 = 16 features + 2 keys)')
print()
print('=== 결측 ===')
print(J_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).head(8).to_string())
print()
print('=== 분포 (요약) ===')
pd.set_option('display.width', 220)
print(J_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(2).T[['count','mean','min','50%','95%','max']])
print()
print('=== id01 첫 3일 (subset) ===')
show_cols = ['m_light_early_evening_mean','m_light_pre_bed_mean',
             'm_screen_use_early_evening_mean','m_screen_use_pre_bed_mean']
print(J_block[J_block.subject_id=='id01'][KEY + show_cols].head(3).to_string())




K_block = all_keys[KEY].copy()
ld = pd.to_datetime(K_block['lifelog_date'])
K_block['cal_dow']        = ld.dt.dayofweek.astype('int8')
K_block['cal_is_weekend'] = (ld.dt.dayofweek >= 5).astype('int8')
K_block['cal_month']      = ld.dt.month.astype('int8')

feature_blocks['K_calendar'] = K_block

print(f'K_block shape: {K_block.shape}  (expected 700 × 5 = 3 features + 2 keys)')
print()
print('=== 분포 ===')
print(K_block.drop(columns=KEY).describe().round(1).T[['count','mean','min','50%','max']])



whr_raw['hr_mean_per_min'] = whr_raw.heart_rate.map(np.mean)
resting_hr = whr_raw.groupby('subject_id')['hr_mean_per_min'].quantile(0.10).to_dict()
print('Subject resting HR (p10):')
for sid, hr in sorted(resting_hr.items()): print(f'  {sid}: {hr:.1f}')

def detect_sleep_onset(sid, ldate, rest_hr):
    ev_start = pd.Timestamp(ldate) + pd.Timedelta(hours=18)
    ev_end   = pd.Timestamp(ldate) + pd.Timedelta(hours=24)
    minutes = pd.date_range(ev_start, ev_end - pd.Timedelta(seconds=1), freq='1min')
    grid = pd.DataFrame({'ts': minutes})
    grid['min_of_day'] = grid['ts'].dt.hour * 60 + grid['ts'].dt.minute

    h = whr_raw[(whr_raw.subject_id==sid) & (whr_raw.timestamp>=ev_start) & (whr_raw.timestamp<ev_end)]
    grid['hr'] = grid['ts'].map(h.set_index(h.timestamp.dt.floor('min'))['hr_mean_per_min'].groupby(level=0).mean()) if len(h)>0 else np.nan

    a = mact_raw[(mact_raw.subject_id==sid) & (mact_raw.timestamp>=ev_start) & (mact_raw.timestamp<ev_end)]
    grid['act'] = grid['ts'].map(a.set_index('timestamp')['m_activity'].groupby(level=0).first()) if len(a)>0 else np.nan

    s = mscr_raw[(mscr_raw.subject_id==sid) & (mscr_raw.timestamp>=ev_start) & (mscr_raw.timestamp<ev_end)]
    grid['scr'] = grid['ts'].map(s.set_index('timestamp')['m_screen_use'].groupby(level=0).first()) if len(s)>0 else np.nan

    l = mlight_raw[(mlight_raw.subject_id==sid) & (mlight_raw.timestamp>=ev_start-pd.Timedelta(minutes=15)) & (mlight_raw.timestamp<ev_end)]
    if len(l)>0:
        l_idx = l.sort_values('timestamp').set_index('timestamp')['m_light']
        grid['light'] = grid['ts'].map(l_idx.reindex(grid['ts'], method='ffill'))
    else: grid['light'] = np.nan

    grid['low_hr']=(grid['hr']<rest_hr+5).fillna(False).astype(int)
    grid['still']=(grid['act']==3).fillna(False).astype(int)
    grid['no_screen']=(grid['scr']==0).fillna(False).astype(int)
    grid['dark']=(grid['light']<10).fillna(False).astype(int)
    grid['score']=grid['low_hr']+grid['still']+grid['no_screen']+grid['dark']
    grid['asleep']=(grid['score']>=3).astype(int)

    runs=[]; s_idx=None; l_run=0
    for i,v in enumerate(grid['asleep'].values):
        if v==1:
            if s_idx is None: s_idx,l_run=i,1
            else: l_run+=1
        else:
            if s_idx is not None and l_run>=5: runs.append((s_idx,l_run))
            s_idx,l_run=None,0
    if s_idx is not None and l_run>=5: runs.append((s_idx,l_run))

    out = {
        'sleep_onset_min_of_day': np.nan, 'sleep_onset_hour_after_18': np.nan,
        'sleep_has_onset': 0, 'sleep_num_attempts': len(runs),
        'sleep_longest_run_min': max((r[1] for r in runs), default=0),
        'evening_dark_ratio': float(grid['dark'].mean()),
        'evening_screen_ratio': float((grid['scr']==1).mean()),
        'evening_still_ratio': float(grid['still'].mean()),
        'evening_low_hr_ratio': float(grid['low_hr'].mean()),
    }
    if runs:
        first_min = grid['min_of_day'].iloc[runs[0][0]]
        out['sleep_onset_min_of_day'] = first_min
        out['sleep_onset_hour_after_18'] = (first_min - 18*60) / 60
        out['sleep_has_onset'] = 1
    return out

import time as _t
t0 = _t.time()
O_results = []
for _, row in all_keys.iterrows():
    res = detect_sleep_onset(row.subject_id, row.lifelog_date, resting_hr[row.subject_id])
    res.update({'subject_id': row.subject_id, 'lifelog_date': row.lifelog_date})
    O_results.append(res)
print(f'Sleep onset detection: {_t.time()-t0:.1f}s')

O_block = pd.DataFrame(O_results)
O_block = O_block[KEY + [c for c in O_block.columns if c not in KEY]]
feature_blocks['O_sleep_onset'] = O_block
print(f'O_block: {O_block.shape}')
print(O_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(2).T[['count','mean','min','50%','95%','max']])



from scipy.signal import welch as scipy_welch

def hrv_freq_for_day(sid, ldate, hour_start=18, hour_end=24):
    ev_s = pd.Timestamp(ldate) + pd.Timedelta(hours=hour_start)
    ev_e = pd.Timestamp(ldate) + pd.Timedelta(hours=hour_end)
    h = whr_raw[(whr_raw.subject_id==sid) & (whr_raw.timestamp>=ev_s) & (whr_raw.timestamp<ev_e)].sort_values('timestamp')
    if len(h) < 5:
        return {'hrv_lf': np.nan, 'hrv_hf': np.nan, 'hrv_lf_hf_ratio': np.nan, 'hrv_total_power': np.nan}
    all_hr = []
    for hr_list in h.heart_rate: all_hr.extend(hr_list)
    if len(all_hr) < 60:
        return {'hrv_lf': np.nan, 'hrv_hf': np.nan, 'hrv_lf_hf_ratio': np.nan, 'hrv_total_power': np.nan}
    arr = np.array(all_hr, dtype=float); arr = arr - arr.mean()
    nperseg = min(120, len(arr))
    freqs, psd = scipy_welch(arr, fs=1.0, nperseg=nperseg)
    LF = float(psd[(freqs>=0.04) & (freqs<0.15)].sum())
    HF = float(psd[(freqs>=0.15) & (freqs<0.4)].sum())
    return {
        'hrv_lf': float(np.log1p(LF)),
        'hrv_hf': float(np.log1p(HF)),
        'hrv_lf_hf_ratio': float(LF / (HF + 1e-9)),
        'hrv_total_power': float(np.log1p(LF + HF)),
    }

t0 = _t.time()
P_results = []
for _, row in all_keys.iterrows():
    res = hrv_freq_for_day(row.subject_id, row.lifelog_date)
    res.update({'subject_id': row.subject_id, 'lifelog_date': row.lifelog_date})
    P_results.append(res)
print(f'HRV freq: {_t.time()-t0:.1f}s')
P_block = pd.DataFrame(P_results)
P_block = P_block[KEY + [c for c in P_block.columns if c not in KEY]]
feature_blocks['P_hrv_freq'] = P_block
print(f'P_block: {P_block.shape}')




import time as _t

def _sampen(x, m=2, r_factor=0.2):
    """Sample Entropy. O(N²) but N≤360 per bucket → fast."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < m + 2:
        return np.nan
    sd = float(np.std(x, ddof=0))
    if sd <= 0:
        return np.nan
    r = r_factor * sd
    def _count(mm):
        if N < mm + 1: return 0
        T = np.array([x[i:i+mm] for i in range(N - mm + 1)])
        n = len(T)
        if n < 2: return 0
        d = np.max(np.abs(T[:, None, :] - T[None, :, :]), axis=2)
        np.fill_diagonal(d, np.inf)
        return int((d <= r).sum() // 2)
    B = _count(m)
    A = _count(m + 1)
    if A == 0 or B == 0:
        return np.nan
    return float(-np.log(A / B))

def _dfa_alpha1(x, scale_min=4, scale_max=16):
    """DFA short-term scaling exponent α1 (boxes 4-16)."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < scale_max * 2 or np.std(x) <= 0:
        return np.nan
    y = np.cumsum(x - np.mean(x))
    scales = np.unique(np.round(np.logspace(
        np.log10(scale_min), np.log10(min(scale_max, N // 4)), 6
    )).astype(int))
    F = []
    for n in scales:
        n = int(n)
        if n < 2 or N // n < 2:
            continue
        n_segs = N // n
        rms = []
        t_axis = np.arange(n)
        for i in range(n_segs):
            seg = y[i*n:(i+1)*n]
            p = np.polyfit(t_axis, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(p, t_axis)) ** 2)))
        F.append((n, float(np.mean(rms))))
    if len(F) < 3:
        return np.nan
    F = np.array(F)
    return float(np.polyfit(np.log10(F[:, 0]), np.log10(F[:, 1]), 1)[0])

whr_p2 = restrict_to_lifelog_day(whr_raw)
whr_p2['hr_min_mean'] = whr_p2.heart_rate.map(np.mean)
whr_p2 = whr_p2.sort_values(KEY + ['timestamp'])

t0 = _t.time()
records = []
for (sid, day, bucket), g in whr_p2.groupby(KEY + ['bucket'], sort=False):
    arr = g['hr_min_mean'].dropna().to_numpy()
    n = len(arr)
    if n < 4:
        continue
    sdnn  = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    rmssd = float(np.sqrt(np.mean(np.diff(arr) ** 2))) if n > 1 else np.nan
    arg   = (2 * sdnn ** 2 - 0.5 * rmssd ** 2) if (
        sdnn is not None and rmssd is not None
        and not np.isnan(sdnn) and not np.isnan(rmssd)
    ) else None
    sd2    = float(np.sqrt(arg)) if (arg is not None and arg > 0) else np.nan
    sampen = _sampen(arr)
    dfa    = _dfa_alpha1(arr) if n >= 12 else np.nan
    records.append({
        'subject_id': sid, 'lifelog_date': day, 'bucket': bucket,
        'sdnn': sdnn, 'rmssd_b': rmssd, 'sd2': sd2,
        'sampen': sampen, 'dfa_a1': dfa,
    })
print(f'HRV nonlinear: {_t.time()-t0:.1f}s, {len(records)} (sid,day,bucket) entries')

p2_long = pd.DataFrame(records)
metrics = ['sdnn', 'rmssd_b', 'sd2', 'sampen', 'dfa_a1']
p2_wide = p2_long.pivot_table(index=KEY, columns='bucket', values=metrics)
p2_wide.columns = [f'whr_b_{b}_{m}' for m, b in p2_wide.columns]

ordered_cols = [f'whr_b_{b}_{m}' for b, _, _ in TIME_BUCKETS for m in metrics]
ordered_cols = [c for c in ordered_cols if c in p2_wide.columns]
p2_wide = p2_wide.reindex(columns=ordered_cols).reset_index()

P2_block = reindex_to_keys(p2_wide)
feature_blocks['P2_hrv_nonlinear'] = P2_block

print(f'\nP2_block shape: {P2_block.shape}')
print()
print('=== 결측 (워치 작동 부족 + 짧은 bucket) ===')
print(P2_block.drop(columns=KEY).isna().mean().sort_values(ascending=False).round(3).head(12).to_string())
print()
print('=== 분포 (top 12) ===')
print(P2_block.drop(columns=KEY).describe(percentiles=[.5, .95]).round(3).T[
    ['count', 'mean', 'std', 'min', '50%', '95%', 'max']].head(12))
print()
print('의미:')
print('  whr_b_<bucket>_sdnn      overall HR variability (분단위)')
print('  whr_b_<bucket>_rmssd_b   parasympathetic activity proxy')
print('  whr_b_<bucket>_sd2       Poincaré long-term variability (sd1=rmssd/√2 제거됨)')
print('  whr_b_<bucket>_sampen    signal complexity (낮을수록 regular)')
print('  whr_b_<bucket>_dfa_a1    autonomic regulation (~1.0 healthy)')
print()
print('shape note: 4 bucket × 5 metric = 20 features (이전 24 → 20, sd1 4개 제거)')
print('overnight sparsity caveat: overnight 4 metrics 결측 75-82% — modeling 단계에서')
print('  COLUMN_GROUPS["hrv_overnight_sparse"] drop contract 적용 권장')




def cross_sensor_for_day(sid, ldate):
    out = {}
    for hour_start, hour_end, name in [(18, 24, 'evening'), (0, 6, 'overnight')]:
        ts = pd.Timestamp(ldate) + pd.Timedelta(hours=hour_start)
        te = pd.Timestamp(ldate) + pd.Timedelta(hours=hour_end)
        s = mscr_raw[(mscr_raw.subject_id==sid) & (mscr_raw.timestamp>=ts) & (mscr_raw.timestamp<te)]
        l = mlight_raw[(mlight_raw.subject_id==sid) & (mlight_raw.timestamp>=ts-pd.Timedelta(minutes=15)) & (mlight_raw.timestamp<te)]
        if len(s) < 10 or len(l) < 1:
            out[f'phone_in_dark_{name}_ratio'] = np.nan
            continue
        l_idx = l.sort_values('timestamp').set_index('timestamp')['m_light']
        s_sorted = s.sort_values('timestamp').reset_index(drop=True)
        light_at_s = l_idx.reindex(s_sorted.timestamp, method='ffill').values
        phone_in_dark = float(((s_sorted.m_screen_use==1) & (light_at_s<10)).mean())
        out[f'phone_in_dark_{name}_ratio'] = phone_in_dark

    ts = pd.Timestamp(ldate) + pd.Timedelta(hours=18)
    te = pd.Timestamp(ldate) + pd.Timedelta(hours=24)
    h = whr_raw[(whr_raw.subject_id==sid) & (whr_raw.timestamp>=ts) & (whr_raw.timestamp<te)]
    a = mact_raw[(mact_raw.subject_id==sid) & (mact_raw.timestamp>=ts) & (mact_raw.timestamp<te)]
    out['hr_during_still_evening_mean'] = np.nan
    out['hr_during_still_evening_std'] = np.nan
    if len(h) > 5 and len(a) > 5:
        h2 = h.copy(); h2['min'] = h2.timestamp.dt.floor('min')
        a2 = a.copy(); a2['min'] = a2.timestamp.dt.floor('min')
        merged = h2[['min','hr_mean_per_min']].merge(a2[['min','m_activity']], on='min', how='inner')
        if len(merged) > 5:
            still_hr = merged[merged.m_activity==3]['hr_mean_per_min']
            if len(still_hr) > 3:
                out['hr_during_still_evening_mean'] = float(still_hr.mean())
                out['hr_during_still_evening_std'] = float(still_hr.std())
    return out

t0 = _t.time()
Q_results = []
for _, row in all_keys.iterrows():
    res = cross_sensor_for_day(row.subject_id, row.lifelog_date)
    res.update({'subject_id': row.subject_id, 'lifelog_date': row.lifelog_date})
    Q_results.append(res)
print(f'Cross-sensor: {_t.time()-t0:.1f}s')
Q_block = pd.DataFrame(Q_results)
Q_block = Q_block[KEY + [c for c in Q_block.columns if c not in KEY]]
feature_blocks['Q_cross_sensor'] = Q_block
print(f'Q_block: {Q_block.shape}')


_qfeat = Q_block.drop(columns=KEY)
print()
print('=== 결측 ===')
print(_qfeat.isna().mean().sort_values(ascending=False).round(3).to_string())
print()
print('=== 분포 ===')
print(_qfeat.describe(percentiles=[.5, .95]).round(3).T[
    ['count', 'mean', 'std', 'min', '50%', '95%', 'max']])
print()
print('=== sparse 의심 (median 0 또는 unique<5) ===')
for c in _qfeat.columns:
    nonna = _qfeat[c].dropna()
    if len(nonna) == 0:
        continue
    med = float(nonna.median())
    nu = nonna.nunique()
    flag = []
    if med == 0: flag.append('median=0')
    if nu < 5:   flag.append(f'unique={nu}')
    if flag:
        print(f'  {c}: {", ".join(flag)}  (n_nonna={len(nonna)}, '
              f'p95={float(nonna.quantile(0.95)):.3f})')

print()
print('=== hr_during_still / phone_in_dark coverage 진단 (S3 r=-0.24 신뢰성) ===')
for c in ['hr_during_still_evening_mean', 'hr_during_still_evening_std',
          'phone_in_dark_evening_ratio', 'phone_in_dark_overnight_ratio']:
    if c in _qfeat.columns:
        n_valid = int(_qfeat[c].notna().sum())
        miss = float(_qfeat[c].isna().mean())
        print(f'  {c:38s} valid={n_valid:4d}/700 ({100*(1-miss):.1f}%)  '
              f'결측 {miss:.1%}')
print('  → 마크다운 r=-0.24는 valid n 위에서만 계산됨 — sanity의 sign_stable로')
print('     이 sparse coverage가 5-fold에서 일관된지 확인 (인사이트 5)')




def activity_sequences_for_day(sid, ldate):
    ds = pd.Timestamp(ldate)
    de = ds + pd.Timedelta(days=1)
    a = mact_raw[(mact_raw.subject_id==sid) & (mact_raw.timestamp>=ds) & (mact_raw.timestamp<de)].sort_values('timestamp')
    if len(a) < 10:
        return {'act_n_transitions': np.nan, 'act_transitions_per_hour': np.nan,
                'act_still_bout_max': np.nan, 'act_still_bout_median': np.nan,
                'act_active_bout_max': np.nan, 'act_active_bout_median': np.nan}
    codes = a['m_activity'].values
    n_trans = int(np.sum(codes[1:] != codes[:-1]))
    trans_per_hour = float(n_trans / (len(codes) / 60.0))
    runs = []; cur, cur_len = codes[0], 1
    for c in codes[1:]:
        if c == cur: cur_len += 1
        else: runs.append((cur, cur_len)); cur, cur_len = c, 1
    runs.append((cur, cur_len))
    still_runs = [r[1] for r in runs if r[0] == 3]
    active_runs = [r[1] for r in runs if r[0] != 3]
    return {
        'act_n_transitions': n_trans,
        'act_transitions_per_hour': trans_per_hour,
        'act_still_bout_max': max(still_runs) if still_runs else 0,
        'act_still_bout_median': float(np.median(still_runs)) if still_runs else 0,
        'act_active_bout_max': max(active_runs) if active_runs else 0,
        'act_active_bout_median': float(np.median(active_runs)) if active_runs else 0,
    }

t0 = _t.time()
R_results = []
for _, row in all_keys.iterrows():
    res = activity_sequences_for_day(row.subject_id, row.lifelog_date)
    res.update({'subject_id': row.subject_id, 'lifelog_date': row.lifelog_date})
    R_results.append(res)
print(f'Activity sequences: {_t.time()-t0:.1f}s')
R_block = pd.DataFrame(R_results)
R_block = R_block[KEY + [c for c in R_block.columns if c not in KEY]]
feature_blocks['R_act_sequence'] = R_block
print(f'R_block: {R_block.shape}')


_rfeat = R_block.drop(columns=KEY)
print()
print('=== 결측 ===')
print(_rfeat.isna().mean().sort_values(ascending=False).round(3).to_string())
print()
print('=== 분포 ===')
print(_rfeat.describe(percentiles=[.5, .95]).round(3).T[
    ['count', 'mean', 'std', 'min', '50%', '95%', 'max']])


_n_raw_lt10 = 0
for _, _r in all_keys.iterrows():
    _ds = _r.lifelog_date; _de = _ds + pd.Timedelta(days=1)
    _a = mact_raw[(mact_raw.subject_id==_r.subject_id) &
                  (mact_raw.timestamp>=_ds) & (mact_raw.timestamp<_de)]
    if len(_a) < 10:
        _n_raw_lt10 += 1

print()
print('=== [R] raw<10 guard 효과 + bout 1440 분석 ===')
print(f'  raw<10 케이스: {_n_raw_lt10}/{len(all_keys)} '
      f'({100*_n_raw_lt10/max(len(all_keys),1):.1f}%) → NaN 정책 적용 대상')
if _n_raw_lt10 == 0:
    print('  → 0건: NaN 정책은 dead branch (future-proof guard로 유지).')
    print('     결측 0%는 측정실패 위장이 아니라 실제 측정실패가 없는 것.')
    print('  → bout_max=1440은 측정 실패가 아니라 실제 monotonic activity day:')
    print('     mActivity는 1분 boundary로 dense (median raw=1440 = 24h 전부 잡힘),')
    print('     일부 날은 한 활동코드가 24h 동안 85-100% 점유 (24h 거의 가만히, 또는 일정 패턴).')
else:
    print(f'  → {_n_raw_lt10}건 적용. 결측 % 컬럼 확인 → 위 결측 표에 반영됨.')
for c in _rfeat.columns:
    nonna = _rfeat[c].dropna()
    if len(nonna) == 0:
        continue
    n_zero = int((nonna == 0).sum())
    pct_zero = n_zero / len(nonna)
    if pct_zero > 0.05:
        print(f'  {c:35s} 0-value rate: {pct_zero:.1%}  ({n_zero}/{len(nonna)} nonna rows)')
print('  → 마크다운 Q2 r=-0.19 / S4 r=-0.16는 진짜 분포 위 값 (오염 가설 기각).')
print('     [R] 정정으로는 변하지 않음 — sanity의 5-fold sign_stable에서')
print('     일관된 부호를 보이는지가 신뢰성의 전부 (인사이트 5).')

print()
print('=== bout_max 꼬리 분포 (outlier 확인) ===')
for c in ['act_still_bout_max', 'act_active_bout_max']:
    if c in _rfeat.columns:
        s = _rfeat[c].dropna()
        if len(s):
            print(f'  {c}: '
                  f'median={float(s.median()):.0f}  '
                  f'p95={float(s.quantile(0.95)):.0f}  '
                  f'max={float(s.max()):.0f}  '
                  f'(max/median ratio={float(s.max())/max(float(s.median()),1):.1f})')
print('  → ratio가 큰 경우 outlier 점검 ([D] HR=1, [F] speed=888, [G] rssi=0과 같은 sentinel 패턴)')




if 'hr_mean_per_min' not in whr_raw.columns:
    whr_raw['hr_mean_per_min'] = whr_raw.heart_rate.map(np.mean)


def sleep_window_features(sid, ldate):
    sd = pd.Timestamp(ldate) + pd.Timedelta(days=1)
    s_start = sd
    s_end = sd + pd.Timedelta(hours=9)
    out = {}

    h = whr_raw[(whr_raw.subject_id==sid) & (whr_raw.timestamp>=s_start) & (whr_raw.timestamp<s_end)]
    h = h[h.hr_mean_per_min.between(30, 200)]
    if len(h) > 5:
        out['whr_sleep_mean'] = float(h.hr_mean_per_min.mean())
        out['whr_sleep_std']  = float(h.hr_mean_per_min.std())
        out['whr_sleep_min']  = float(h.hr_mean_per_min.min())
        out['whr_sleep_max']  = float(h.hr_mean_per_min.max())
        out['whr_sleep_n_min'] = int(len(h))
    else:
        out.update({'whr_sleep_mean': np.nan, 'whr_sleep_std': np.nan,
                    'whr_sleep_min': np.nan, 'whr_sleep_max': np.nan, 'whr_sleep_n_min': 0})

    a = mact_raw[(mact_raw.subject_id==sid) & (mact_raw.timestamp>=s_start) & (mact_raw.timestamp<s_end)]
    out['mact_sleep_still_ratio'] = float((a.m_activity == 3).mean()) if len(a) > 5 else np.nan
    out['mact_sleep_n_min'] = int(len(a))

    s = mscr_raw[(mscr_raw.subject_id==sid) & (mscr_raw.timestamp>=s_start) & (mscr_raw.timestamp<s_end)]
    out['mscr_sleep_screen_ratio'] = float(s.m_screen_use.mean()) if len(s) > 5 else np.nan

    l = mlight_raw[(mlight_raw.subject_id==sid) & (mlight_raw.timestamp>=s_start) & (mlight_raw.timestamp<s_end)]
    out['mlight_sleep_mean'] = float(l.m_light.mean()) if len(l) > 0 else np.nan
    out['mlight_sleep_max']  = float(l.m_light.max()) if len(l) > 0 else np.nan

    return out

import time as _t
t0 = _t.time()
S_results = []
for _, row in all_keys.iterrows():
    res = sleep_window_features(row.subject_id, row.lifelog_date)
    res.update({'subject_id': row.subject_id, 'lifelog_date': row.lifelog_date})
    S_results.append(res)
print(f'Sleep window features: {_t.time()-t0:.1f}s')

S_block = pd.DataFrame(S_results)
S_block = S_block[KEY + [c for c in S_block.columns if c not in KEY]]
feature_blocks['S_sleep_window'] = S_block
print(f'S_block: {S_block.shape}')
print(S_block.drop(columns=KEY).describe(percentiles=[.5,.95]).round(2).T[['count','mean','min','50%','95%','max']])




LABELS_T = ['Q1','Q2','Q3','S1','S2','S3','S4']

labels_train = train.set_index(KEY)[LABELS_T]
combined = (all_keys[KEY].copy()
            .set_index(KEY)
            .join(labels_train, how='left')
            .reset_index()
            .sort_values(KEY)
            .reset_index(drop=True))

T_block = combined[KEY].copy()

print('Causal encoding: lag1 only for 7 labels (enc windows dropped as redundant)...')
for lab in LABELS_T:
    g = combined.groupby('subject_id')[lab]
    T_block[f'{lab}_lag1'] = g.shift(1)


feature_blocks['T_target_encoding'] = T_block
print(f'T_block shape: {T_block.shape}')
print(f'features: {T_block.shape[1] - 2} = 7 lag1  (이전 35 → 7, 28 redundant 제거)')

first_day_per_subj = combined.groupby('subject_id').head(1)
print(f'lag1 NaN check (should = 10): '
      f'{T_block.loc[first_day_per_subj.index, "Q1_lag1"].isna().sum()}')




EXPANDING_SOURCE_COLS = [
    'wpedo_step_sum', 'wpedo_active_min', 'wpedo_distance_sum',
    'm_charging_ratio', 'm_screen_use_ratio',
    'm_light_p50', 'w_light_p50',
    'whr_mean', 'whr_rmssd_minute',
    'amb_day_silence_score',
]

src = all_keys[KEY].copy()
for col in EXPANDING_SOURCE_COLS:
    found = False
    for name, block in feature_blocks.items():
        if col in block.columns:
            src = src.merge(block[KEY + [col]], on=KEY, how='left')
            found = True
            break
    if not found:
        raise ValueError(f'{col!r} not found in feature_blocks')

src = src.sort_values(KEY).reset_index(drop=True)

EPS = 1e-6
L_block = src[KEY].copy()
for col in EXPANDING_SOURCE_COLS:
    g = src.groupby('subject_id')[col]
    exp_mean_today = g.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    exp_std_today  = g.expanding(min_periods=1).std().reset_index(level=0, drop=True)
    exp_mean = exp_mean_today.groupby(src['subject_id']).shift(1)
    exp_std  = exp_std_today.groupby(src['subject_id']).shift(1)

    L_block[f'{col}_subj_exp_mean'] = exp_mean.values
    L_block[f'{col}_subj_zscore']   = ((src[col] - exp_mean) / (exp_std + EPS)).values

feature_blocks['L_subj_expanding'] = L_block

print(f'L_block shape: {L_block.shape}  (expected 700 × 22 = 20 features + 2 keys)')
print()
print('=== 결측 비율 (첫 날 NaN 영향) ===')
miss = L_block.drop(columns=KEY).isna().mean().sort_values(ascending=False)
print(miss.round(3).head(8).to_string())
print(f'  ... (전체 {len(miss)}개 컬럼)')
print()
print('=== id01 첫 5일 (예: step expanding) ===')
print(L_block[L_block.subject_id=='id01'][KEY + ['wpedo_step_sum_subj_exp_mean','wpedo_step_sum_subj_zscore']].head(5).to_string())




ROLLING_SOURCE_COLS = [
    'wpedo_step_sum', 'm_screen_use_ratio',
    'whr_mean', 'm_light_p50', 'amb_day_silence_score',
]

M_block = src[KEY].copy()
for col in ROLLING_SOURCE_COLS:
    g = src.groupby('subject_id')[col]
    roll_today = g.rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    roll = roll_today.groupby(src['subject_id']).shift(1)
    M_block[f'{col}_subj_roll7d_mean'] = roll.values

feature_blocks['M_roll7d'] = M_block

print(f'M_block shape: {M_block.shape}  (expected 700 × 7 = 5 features + 2 keys)')
print()
print('=== id01 첫 10일 (step rolling 7d 변화 보기) ===')
ex = M_block[M_block.subject_id=='id01'][KEY + ['wpedo_step_sum_subj_roll7d_mean']].head(10)
print(ex.to_string())




def select_light_cols_to_log(block, exclude_substrs=('std','min','missing','max')):
    """block에서 'light' 들어가고 exclude_substrs 안 들어간 numeric 컬럼만."""
    out = []
    for col in block.columns:
        if 'light' not in col: continue
        if any(sub in col for sub in exclude_substrs): continue
        out.append(col)
    return out

A_log_cols = select_light_cols_to_log(feature_blocks['A_light'], exclude_substrs=('std','min','missing'))
print(f'A_block log1p 대상 ({len(A_log_cols)}): {A_log_cols}')

A_log_cols_with_max = A_log_cols + [c for c in feature_blocks['A_light'].columns
                                     if 'light' in c and 'max' in c and 'missing' not in c]
print(f'  max 포함 ({len(A_log_cols_with_max)})')

for col in A_log_cols_with_max:
    feature_blocks['A_light'][f'{col}_log1p'] = np.log1p(feature_blocks['A_light'][col])

J_log_cols = [c for c in feature_blocks['J_pre_bed'].columns
              if 'light' in c and 'mean' in c]
print(f'J_block log1p 대상 ({len(J_log_cols)}): {J_log_cols}')
for col in J_log_cols:
    feature_blocks['J_pre_bed'][f'{col}_log1p'] = np.log1p(feature_blocks['J_pre_bed'][col])

print(f'\nA_light shape after log1p: {feature_blocks["A_light"].shape}')
print(f'J_pre_bed shape after log1p: {feature_blocks["J_pre_bed"].shape}')

import scipy.stats as ss
ml_raw = feature_blocks['A_light']['m_light_p50']
wl_raw = feature_blocks['A_light']['w_light_p50']
ml_log = feature_blocks['A_light']['m_light_p50_log1p']
wl_log = feature_blocks['A_light']['w_light_p50_log1p']
m = ~(ml_raw.isna() | wl_raw.isna())
print('\nm_light_p50 vs w_light_p50:')
print(f'  Pearson  raw  : {ss.pearsonr(ml_raw[m], wl_raw[m])[0]:.3f}')
print(f'  Pearson  log1p: {ss.pearsonr(ml_log[m], wl_log[m])[0]:.3f}')
print(f'  Spearman raw  : {ss.spearmanr(ml_raw[m], wl_raw[m])[0]:.3f}  (변환 무관, 같음)')




features = all_keys[KEY + ['sleep_date','split']].copy()

for name, block in feature_blocks.items():
    features = features.merge(block, on=KEY, how='left')
    print(f'  + {name:18s} → shape: {features.shape}')

LABELS = ['Q1','Q2','Q3','S1','S2','S3','S4']
labels = pd.concat([
    train[KEY + LABELS],
    sub[KEY].assign(**{lab: np.nan for lab in LABELS}),
], ignore_index=True)
features = features.merge(labels, on=KEY, how='left')

print(f'\n최종 shape: {features.shape}')
print(f'  train rows: {(features.split=="train").sum()} (라벨 있음)')
print(f'  sub   rows: {(features.split=="sub").sum()} (라벨 NaN)')
print(f'  키 중복: {features[KEY].duplicated().sum()}')
print(f'  컬럼명 중복: {features.columns.duplicated().sum()}')
print(f'  feature 컬럼 수: {features.shape[1] - len(KEY) - 2 - len(LABELS)}'
      f' (= total - 2 keys - sleep_date - split - 7 labels)')


import json

S_PREFIXES = (
    'whr_', 'hrv_',
    'mact_sleep_', 'mscr_sleep_', 'mlight_sleep_',
    'sleep_', 'longest_sleep', 'num_sleep_attempts', 'has_sleep',
    'evening_low_hr', 'evening_still', 'evening_screen', 'evening_dark',
    'hr_during_still',
    'act_',
    'wpedo_',
    'm_light_', 'w_light_',
)
Q_PREFIXES = (
    'amb_',
    'wifi_', 'ble_', 'gps_',
    'phone_in_dark',
    'usage_',
    'm_screen_use', 'm_charging', 'm_activity',
)
SHARED_PREFIXES = ('cal_',)
SHARED_SUFFIXES = ('_subj_exp_mean', '_subj_zscore', '_subj_roll7d_mean',
                   '_missing')

def classify_column(col):
    if any(col.startswith(p) for p in S_PREFIXES):
        return 'S_relevant'
    if any(col.startswith(p) for p in Q_PREFIXES):
        return 'Q_relevant'
    if any(col.startswith(p) for p in SHARED_PREFIXES):
        return 'shared'
    if any(col.endswith(s) for s in SHARED_SUFFIXES):
        return 'shared'
    if col.startswith(('Q1_','Q2_','Q3_')):
        return 'Q_relevant'
    if col.startswith(('S1_','S2_','S3_','S4_')):
        return 'S_relevant'
    return 'unclassified'

def in_Q_zscore_only(col):
    """True iff col이 within-subject component만 (LOSO-safe for Q labels)."""
    if col.endswith('_subj_zscore'):
        return True
    if col.startswith(('Q1_lag', 'Q2_lag', 'Q3_lag')):
        return True
    if col.startswith('cal_'):
        return True
    return False

feat_cols_only = [c for c in features.columns
                  if c not in KEY + ['sleep_date','split'] + LABELS]
groups = {c: classify_column(c) for c in feat_cols_only}
q_zscore_only_cols = sorted([c for c in feat_cols_only if in_Q_zscore_only(c)])

AMBIENCE_REDUNDANT_PAIRS = [
    ('amb_sleep_silence',         'amb_b_overnight_silence'),
    ('amb_sleep_speech',          'amb_b_overnight_speech'),
    ('amb_sleep_music',           'amb_b_overnight_music_score'),
    ('amb_prebed_silence',        'amb_b_evening_silence_score'),
    ('amb_prebed_speech',         'amb_b_evening_speech_score'),
    ('amb_prebed_music',          'amb_b_evening_music_score'),
]
present_pairs = [(a, b) for a, b in AMBIENCE_REDUNDANT_PAIRS
                 if a in feat_cols_only and b in feat_cols_only]

hrv_overnight_sparse_cols = sorted(
    [c for c in feat_cols_only if c.startswith('whr_b_overnight_')]
)

LOCATION_REDUNDANT_PAIRS = [
    ('gps_home_ratio',            'wifi_home_scan_ratio'),
    ('gps_evening_home_ratio',    'wifi_home_evening_ratio'),
    ('gps_overnight_home_ratio',  'wifi_home_overnight_ratio'),
]
present_location_pairs = [(a, b) for a, b in LOCATION_REDUNDANT_PAIRS
                           if a in feat_cols_only and b in feat_cols_only]

COLUMN_GROUPS = {
    'S_relevant':   sorted([c for c,g in groups.items() if g=='S_relevant']),
    'Q_relevant':   sorted([c for c,g in groups.items() if g=='Q_relevant']),
    'shared':       sorted([c for c,g in groups.items() if g=='shared']),
    'unclassified': sorted([c for c,g in groups.items() if g=='unclassified']),
    'Q_zscore_only': q_zscore_only_cols,
    'ambience_redundant_pairs': [list(p) for p in present_pairs],
    'location_redundant_pairs': [list(p) for p in present_location_pairs],
    'hrv_overnight_sparse': hrv_overnight_sparse_cols,
}

print('Column groups:')
for g, cols in COLUMN_GROUPS.items():
    print(f'  {g:15s}: {len(cols):3d} cols')
if COLUMN_GROUPS['unclassified']:
    print('\n Unclassified (prefix 추가 필요):')
    for c in COLUMN_GROUPS['unclassified']:
        print(f'  {c}')

if present_pairs:
    print(f'\nambience_redundant_pairs: {len(present_pairs)} pair(s) — '
         f'modeling은 sign-stability 진단으로 한쪽 select')
    for a, b in present_pairs:
        print(f'  {a:30s}  ↔  {b}')

if present_location_pairs:
    print(f'\nlocation_redundant_pairs: {len(present_location_pairs)} pair(s) — '
         f'GPS vs wifi home 절단, modeling이 sign-stability 진단으로 select')
    for a, b in present_location_pairs:
        print(f'  {a:30s}  ↔  {b}')

if hrv_overnight_sparse_cols:
    print(f'\nhrv_overnight_sparse: {len(hrv_overnight_sparse_cols)} col(s) — '
          f'overnight HRV 결측 75-82%, modeling 단계 drop 후보')

print('\nQ_zscore_only (Q 라벨 학습 시 권장, LOSO-safe):')
print(f'  {len(q_zscore_only_cols)} cols / {len(feat_cols_only)} total '
      f'({100*len(q_zscore_only_cols)/len(feat_cols_only):.1f}%)')
print('  (in_Q_zscore_only: _subj_zscore | Q*_lag | cal_*)')
print('  rationale: Q 라벨이 within-subject deviation으로 binarize되므로 절대 피처는')
print('             정의상 직교. LOSO에서 노이즈, random K-fold에서 subject ID 누설.')

groups_path = OUT_DIR / f'feature_column_groups{RAW_SUFFIX}.json'
with open(groups_path, 'w') as f:
    json.dump(COLUMN_GROUPS, f, indent=2)
print(f'\n저장: {groups_path}')


feat_cols_only = [c for c in features.columns
                  if c not in KEY + ['sleep_date','split'] + LABELS]

print(f'전체 피처 컬럼 수: {len(feat_cols_only)}')
print()

def prefix_of(col):
    for p in ['m_light','w_light','m_charging','m_screen_use','m_activity',
              'wpedo','whr','amb','gps','wifi','ble','usage']:
        if col.startswith(p):
            return p
    return 'other'

prefix_map = {c: prefix_of(c) for c in feat_cols_only}

miss_long = features[KEY + feat_cols_only].melt(id_vars=KEY, var_name='col', value_name='v')
miss_long['prefix'] = miss_long.col.map(prefix_map)
miss_long['is_na']  = miss_long.v.isna().astype(int)

heat = (miss_long.groupby(['subject_id','prefix'])['is_na']
        .mean().unstack('prefix').fillna(0).round(3))
print('=== sid × block 결측 비율 ===')
print(heat)
print()

print('=== 결측 큰 피처 top 10 ===')
print(features[feat_cols_only].isna().mean().sort_values(ascending=False).head(10).round(3).to_string())


# Per-label feature correlation audit: prints only, no effect on the saved features.
if VERBOSE:
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold

    tr = features[features.split == 'train'].copy()
    print(f'train rows for correlation: {len(tr)}')
    print(f'unique subjects: {tr.subject_id.nunique()}')

    gkf = GroupKFold(n_splits=5)
    groups = tr.subject_id.to_numpy()
    fold_indices = list(gkf.split(tr, groups=groups))

    corr_rows = []
    for lab in LABELS:
        y_all = tr[lab].astype(float).to_numpy()
        for c in feat_cols_only:
            x_all = tr[c].astype(float).to_numpy()
            fold_rhos = []
            for _, val_idx in fold_indices:
                xv = x_all[val_idx]; yv = y_all[val_idx]
                m = ~(pd.isna(xv) | pd.isna(yv))
                if m.sum() < 20:
                    continue
                rho, _ = spearmanr(xv[m], yv[m])
                if not np.isnan(rho):
                    fold_rhos.append(float(rho))
            if len(fold_rhos) < 3:
                continue
            fold_rhos = np.array(fold_rhos)
            n_pos = int((fold_rhos > 0).sum()); n_neg = int((fold_rhos < 0).sum())
            sign_stable = max(n_pos, n_neg) / len(fold_rhos)
            corr_rows.append({
                'label': lab, 'feature': c,
                'median_r': float(np.median(fold_rhos)),
                'mean_r':   float(np.mean(fold_rhos)),
                'std_r':    float(np.std(fold_rhos, ddof=1)),
                'min_r':    float(np.min(fold_rhos)),
                'max_r':    float(np.max(fold_rhos)),
                'sign_stable': sign_stable,
                'n_folds':  len(fold_rhos),
            })

    corr_df = pd.DataFrame(corr_rows)

    print()
    print('=== 라벨별 |median_r| top-5 피처 (5-fold subject-cluster spearman) ===')
    print('   sign_stable = 5 fold 중 같은 부호 비율 (1.0 = robust; <0.6 = noise)')
    print()
    for lab in LABELS:
        sub_corr = corr_df[corr_df.label==lab].copy()
        sub_corr['abs'] = sub_corr.median_r.abs()
        top = sub_corr.nlargest(5, 'abs')[
            ['feature','median_r','std_r','min_r','max_r','sign_stable','n_folds']
        ]
        print(f'[{lab}]')
        print(top.to_string(index=False, float_format=lambda v: f'{v:+.3f}' if isinstance(v,float) else str(v)))
        print()

    edge_threshold = 0.20
    unstable_threshold = 0.8
    edge = corr_df[corr_df.median_r.abs() < edge_threshold].copy()
    unstable_edge = edge[edge.sign_stable < unstable_threshold]
    print('=== 인사이트 5 진단: noise-floor edge sign instability ===')
    print(f'  |median_r| < {edge_threshold}: {len(edge)} (feature, label) pairs')
    print(f'    그 중 sign_stable < {unstable_threshold} (부호 5-fold에서 불안정): '
          f'{len(unstable_edge)} ({100*len(unstable_edge)/max(len(edge),1):.1f}%)')
    print('  → sign instability가 noise-floor 가장자리에서 큰 비율로 발견되면')
    print('    단발 r 채택은 인사이트 5의 collapse risk를 정량 입증')


    ambience_pairs = [
        ('amb_sleep_silence',         'amb_b_overnight_silence'),
        ('amb_sleep_speech',          'amb_b_overnight_speech'),
        ('amb_prebed_silence',        'amb_b_evening_silence_score'),
        ('amb_prebed_speech',         'amb_b_evening_speech_score'),
    ]
    pairs_present = [(a, b) for a, b in ambience_pairs
                     if a in feat_cols_only and b in feat_cols_only]
    if pairs_present:
        print()
        print('=== [E] ↔ [E2] ambience 절단 sign-stability 비교 ===')
        print('  같은 신호의 두 시간 절단 → label별 어느 쪽 부호가 더 안정적인지')
        print(f"  {'pair':50s} {'label':6s} {'E_med':>7s} {'E_stab':>7s} {'E2_med':>7s} {'E2_stab':>7s} {'verdict':>10s}")
        for a, b in pairs_present:
            for lab in LABELS:
                rA = corr_df[(corr_df.label==lab) & (corr_df.feature==a)]
                rB = corr_df[(corr_df.label==lab) & (corr_df.feature==b)]
                if rA.empty or rB.empty:
                    continue
                mA, sA = float(rA.median_r.iloc[0]), float(rA.sign_stable.iloc[0])
                mB, sB = float(rB.median_r.iloc[0]), float(rB.sign_stable.iloc[0])
                if sA == sB:
                    verdict = 'tie'
                else:
                    verdict = '[E] preferred' if sA > sB else '[E2] preferred'
                if max(sA, sB) >= 0.8 or max(abs(mA), abs(mB)) >= 0.10:
                    pair_label = f'{a[:24]} ↔ {b[:24]}'
                    print(f'  {pair_label:50s} {lab:6s} {mA:+7.3f} {sA:7.2f} {mB:+7.3f} {sB:7.2f} {verdict:>10s}')
        print('  → modeling: 라벨별로 sign_stable이 더 높은 절단을 select.')
        print('    sA, sB 둘 다 < 0.8이면 둘 다 noise — 둘 다 drop 권장.')


    location_pairs = [
        ('gps_home_ratio',            'wifi_home_scan_ratio'),
        ('gps_evening_home_ratio',    'wifi_home_evening_ratio'),
        ('gps_overnight_home_ratio',  'wifi_home_overnight_ratio'),
    ]
    loc_present = [(a, b) for a, b in location_pairs
                   if a in feat_cols_only and b in feat_cols_only]
    if loc_present:
        print()
        print('=== [F] GPS ↔ [G] wifi location 절단 sign-stability 비교 ===')
        print('  같은 "어디에 있었나"를 두 modality로 → label별 어느 쪽이 더 안정적인지')
        print(f"  {'pair':50s} {'label':6s} {'F_med':>7s} {'F_stab':>7s} {'G_med':>7s} {'G_stab':>7s} {'verdict':>10s}")
        for a, b in loc_present:
            for lab in LABELS:
                rA = corr_df[(corr_df.label==lab) & (corr_df.feature==a)]
                rB = corr_df[(corr_df.label==lab) & (corr_df.feature==b)]
                if rA.empty or rB.empty:
                    continue
                mA, sA = float(rA.median_r.iloc[0]), float(rA.sign_stable.iloc[0])
                mB, sB = float(rB.median_r.iloc[0]), float(rB.sign_stable.iloc[0])
                if sA == sB:
                    verdict = 'tie'
                else:
                    verdict = '[F] preferred' if sA > sB else '[G] preferred'
                if max(sA, sB) >= 0.8 or max(abs(mA), abs(mB)) >= 0.10:
                    pair_label = f'{a[:24]} ↔ {b[:24]}'
                    print(f'  {pair_label:50s} {lab:6s} {mA:+7.3f} {sA:7.2f} {mB:+7.3f} {sB:7.2f} {verdict:>10s}')
        print('  → modeling: 라벨별로 sign_stable이 더 높은 modality를 select.')
        print('    둘 다 < 0.8이면 둘 다 drop (location signal noise floor 안).')


OUT_PATH = OUT_DIR / f'features{RAW_SUFFIX}.parquet'
features.to_parquet(OUT_PATH, index=False)
log(f'saved {OUT_PATH}  shape={features.shape}')
print(f'  파일 크기: {OUT_PATH.stat().st_size / 1024:.1f} KB')

chk = pd.read_parquet(OUT_PATH)
assert chk.shape == features.shape
assert (chk[KEY] == features[KEY]).all().all()
print('\n읽기 검증 OK')

print('\n--- 사용법 ---')
print("import pandas as pd")
print("df = pd.read_parquet('cache/features.parquet')")
print("train_X = df[df.split=='train'].drop(columns=['split','sleep_date','Q1','Q2','Q3','S1','S2','S3','S4'])")
print("train_y = df[df.split=='train'][['Q1','Q2','Q3','S1','S2','S3','S4']]")
print("sub_X   = df[df.split=='sub']  .drop(columns=['split','sleep_date','Q1','Q2','Q3','S1','S2','S3','S4'])")




SLEEP_HOURS = 9
BIN_MIN     = 5
N_BINS      = SLEEP_HOURS * 60 // BIN_MIN
CHANNELS    = ['hr', 'light_log1p', 'screen', 'activity_still', 'step_log1p', 'ambience_silence']
HR_MIN, HR_MAX = 30, 200

def bin_sleep_window(raw_df, value_col, agg='mean'):
    """sensor raw → groupby (sid, lifelog_date, bin_idx)[value_col].agg.

    sleep window = sleep_date 00:00~09:00 = label row의 (lifelog_date+1) 새벽.
    """
    df = raw_df[['subject_id', 'timestamp', value_col]].copy()
    df = df[df.timestamp.dt.hour < SLEEP_HOURS]
    df['sleep_date']   = df.timestamp.dt.normalize()
    df['lifelog_date'] = df.sleep_date - pd.Timedelta(days=1)
    df['bin_idx']      = ((df.timestamp.dt.hour * 60 + df.timestamp.dt.minute) // BIN_MIN).astype(np.int16)
    df = df.merge(all_keys[KEY], on=KEY, how='inner')
    return df.groupby(KEY + ['bin_idx'])[value_col].agg(agg)

def to_dense(agg_series):
    """groupby Series → (N_ROWS, N_BINS) dense + mask, all_keys 순서."""
    wide = agg_series.unstack('bin_idx')
    for i in range(N_BINS):
        if i not in wide.columns:
            wide[i] = np.nan
    wide = wide[list(range(N_BINS))]
    wide = wide.reindex(all_keys.set_index(KEY).index)
    mask = (~wide.isna()).astype(np.int8).values
    arr  = wide.fillna(0.0).astype(np.float32).values
    return arr, mask

import time as _t
t0 = _t.time()

if 'hr_mean_per_min' not in whr_raw.columns:
    whr_raw['hr_mean_per_min'] = whr_raw.heart_rate.map(np.mean)
whr_clean = whr_raw[whr_raw.hr_mean_per_min.between(HR_MIN, HR_MAX)]
print(f'HR filter: {len(whr_clean):,} / {len(whr_raw):,} minutes kept')
hr_arr, hr_mask = to_dense(bin_sleep_window(whr_clean, 'hr_mean_per_min', 'mean'))

ml_arr, ml_mask = to_dense(bin_sleep_window(mlight_raw, 'm_light', 'mean'))
ml_arr = np.log1p(np.maximum(ml_arr, 0.0)).astype(np.float32)

sc_arr, sc_mask = to_dense(bin_sleep_window(mscr_raw, 'm_screen_use', 'mean'))

ma = mact_raw[['subject_id', 'timestamp', 'm_activity']].copy()
ma['is_still'] = (ma.m_activity == 3).astype(np.int8)
st_arr, st_mask = to_dense(bin_sleep_window(ma, 'is_still', 'mean'))

step_arr, step_mask = to_dense(bin_sleep_window(wpedo_raw, 'step', 'sum'))
step_arr = np.log1p(np.maximum(step_arr, 0.0)).astype(np.float32)

amb_for_tensor = amb_raw[['subject_id', 'timestamp', 'm_ambience']].copy()
amb_for_tensor['is_silence'] = amb_for_tensor.m_ambience.map(
    lambda x: 1.0 if (len(x) > 0 and str(x[0][0]) == 'Silence') else 0.0
).astype(np.float32)
am_arr, am_mask = to_dense(bin_sleep_window(amb_for_tensor, 'is_silence', 'mean'))

X    = np.stack([hr_arr, ml_arr, sc_arr, st_arr, step_arr, am_arr], axis=1).astype(np.float32)
mask = np.stack([hr_mask, ml_mask, sc_mask, st_mask, step_mask, am_mask], axis=1).astype(np.int8)

print(f'tensor build: {_t.time()-t0:.1f}s')
print(f'X    shape={X.shape}    dtype={X.dtype}')
print(f'mask shape={mask.shape} dtype={mask.dtype}')

SLEEP_TENSOR_PATH = OUT_DIR / f'sleep_window_tensor{RAW_SUFFIX}.npz'
np.savez_compressed(
    SLEEP_TENSOR_PATH,
    X=X, mask=mask,
    subject_id   = all_keys.subject_id.values.astype(str),
    lifelog_date = pd.to_datetime(all_keys.lifelog_date).dt.strftime('%Y-%m-%d').values,
    sleep_date   = pd.to_datetime(all_keys.sleep_date).dt.strftime('%Y-%m-%d').values,
    split        = all_keys.split.values.astype(str),
    channels     = np.array(CHANNELS),
)
log(f'saved {SLEEP_TENSOR_PATH}  X={X.shape} mask={mask.shape}')

chk = np.load(SLEEP_TENSOR_PATH, allow_pickle=False)
assert chk['X'].shape == X.shape and chk['mask'].shape == mask.shape
print('reload OK')

print('\n=== Channel별 row coverage (per-row mask sum / 108) ===')
print(f'  {"channel":>18}  {"mean":>6} {"p50":>6} {"p95":>6} {"empty":>10}')
for c, name in enumerate(CHANNELS):
    cov = mask[:, c].mean(1)
    n_empty = int((cov == 0).sum())
    print(f'  {name:>18}  {cov.mean():>6.3f} {np.median(cov):>6.3f} {np.quantile(cov, 0.95):>6.3f} {n_empty:>4}/{len(cov)}')

print('\n=== Channel별 값 분포 (mask=1 bin만) ===')
print(f'  {"channel":>18}  {"n_bins":>8} {"mean":>8} {"std":>8} {"min":>8} {"max":>8}')
for c, name in enumerate(CHANNELS):
    has = mask[:, c].astype(bool)
    v = X[:, c][has]
    print(f'  {name:>18}  {has.sum():>8} {v.mean():>8.3f} {v.std():>8.3f} {v.min():>8.3f} {v.max():>8.3f}')

print('\n=== train vs sub coverage 비교 ===')
is_train = (all_keys.split == 'train').values
is_sub   = (all_keys.split == 'sub').values
for c, name in enumerate(CHANNELS):
    tr = mask[is_train, c].mean()
    sb = mask[is_sub, c].mean()
    print(f'  {name:>18}  train={tr:.3f}  sub={sb:.3f}  diff={tr-sb:+.3f}')

print('\n--- 사용법 ---')
print("d = np.load('cache/sleep_window_tensor.npz', allow_pickle=False)")
print("X    = d['X']            # (700, 6, 108) float32")
print("mask = d['mask']         # (700, 6, 108) int8")

