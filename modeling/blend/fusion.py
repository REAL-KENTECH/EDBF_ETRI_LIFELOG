"""fusion.py — the deployed EDBF fusion tree.

`deployed_pipeline(...)` wires the convex-weight primitives from combine.py (`bg_alpha` 2-way,
`global_bg_weights` N-way Ledoit-Wolf) into

    final = GlobalBG-LW( Platt( BG(base, BG(BG(cnn, mr), timing)) ), context )

together with the per-fold (`bg_oof`), full-train (`bg_sub`) and assembly (`platt_full`) helpers.
"""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from modeling.blend.combine import bg_alpha, global_bg_weights

EPS = 1e-12


def bg_oof(m, n, y, folds):
    """Leak-safe per-fold bounded BG blend of two OOF prediction sets (alpha fit on the other folds)."""
    fo = np.empty(len(y), int)
    for k, (_, va) in enumerate(folds):
        fo[va] = k
    out = np.zeros_like(m)
    for l in range(m.shape[1]):
        for k, (_, va) in enumerate(folds):
            o = np.where(fo != k)[0]
            a = float(bg_alpha(m[o, l], n[o, l], y[o, l]))
            out[va, l] = (1 - a) * m[va, l] + a * n[va, l]
    return out.clip(EPS, 1 - EPS)


def bg_sub(m_oof, n_oof, y, m_sub, n_sub):
    """Sub-set bounded BG blend: full-train per-label alpha (from OOF), applied to the sub predictions."""
    a = bg_alpha(m_oof, n_oof, y)
    return np.stack([(1 - a[l]) * m_sub[:, l] + a[l] * n_sub[:, l] for l in range(m_oof.shape[1])], 1).clip(EPS, 1 - EPS)


def platt_full(oof, sub, y):
    """Full-train per-label Platt (the deployed assembly calibration), applied to both oof and sub."""
    o = np.zeros_like(oof); s = np.zeros_like(sub)
    for l in range(oof.shape[1]):
        zo = logit(np.clip(oof[:, l], EPS, 1 - EPS)).reshape(-1, 1)
        zs = logit(np.clip(sub[:, l], EPS, 1 - EPS)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(zo, y[:, l])
        o[:, l] = lr.predict_proba(zo)[:, 1]
        s[:, l] = lr.predict_proba(zs)[:, 1]
    return o.clip(EPS, 1 - EPS), s.clip(EPS, 1 - EPS)


def deployed_pipeline(tab_o, tab_s, scnn_o, scnn_s, mr_o, mr_s,
                      evt_lgbm_o, evt_lgbm_s, evt_lin_o, evt_lin_s, ctx_o, ctx_s, y, folds, use_mr=True):
    """The deployed best as one call: GlobalBG-LW( Platt( BG(base, BG(BG(cnn,mr), timing)) ), context ).

    timing = BG(timing-LGBM, timing-linear); base is the already-BG(LGBM, elastic-net) base. `folds`
    sets the per-fold alpha on the OOF path. With use_mr=False the sleep-tensor MiniRocket co-view is
    pruned, leaving the CNN as the sole sleep encoder; that is the deployed build (md5 3b0f2da7).
    Returns (oof, sub)."""
    evt_o = bg_oof(evt_lgbm_o, evt_lin_o, y, folds); evt_s = bg_sub(evt_lgbm_o, evt_lin_o, y, evt_lgbm_s, evt_lin_s)
    if use_mr:
        to = bg_oof(scnn_o, mr_o, y, folds); ts = bg_sub(scnn_o, mr_o, y, scnn_s, mr_s)
    else:
        to = np.clip(scnn_o, EPS, 1 - EPS); ts = np.clip(scnn_s, EPS, 1 - EPS)
    so = bg_oof(to, evt_o, y, folds); ss = bg_sub(to, evt_o, y, ts, evt_s)
    fo = bg_oof(tab_o, so, y, folds); fs = bg_sub(tab_o, so, y, tab_s, ss)
    fo_c, fs_c = platt_full(fo, fs, y)
    out_o = np.zeros_like(fo_c); out_s = np.zeros_like(fs_c)
    for l in range(fo_c.shape[1]):
        q = global_bg_weights(np.stack([fo_c[:, l], ctx_o[:, l]]), y[:, l])
        out_o[:, l] = q[0] * fo_c[:, l] + q[1] * ctx_o[:, l]
        out_s[:, l] = q[0] * fs_c[:, l] + q[1] * ctx_s[:, l]
    return out_o.clip(EPS, 1 - EPS), out_s.clip(EPS, 1 - EPS)
