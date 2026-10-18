"""Section 8: walk-forward training of UP / DOWN models on the daily panel.

Folds = common_eval.walkforward.blocks (half-year tests 2022-01 .. 2025-06). For a test block starting at ts:
  train rows : day + H_NEW bars < ts  (label window ends before the test)  AND  day < ts - EMBARGO
  validation : the last VALID_FRAC of the train period (by date); fit rows end H_NEW + EMBARGO before it (purged)
Models: rule baseline (form fixed from the section-5 profile, thresholds re-fit per fold on train rows only),
logistic regression (median-impute + missing flags), XGBoost (early stopping on the validation slice).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config                      # noqa: E402
from common_eval.walkforward import blocks                       # noqa: E402

CFG = load_config()
BAR = pd.Timedelta(hours=4)
H_TD = BAR * CFG['horizons']['H_NEW']
EMB_TD = BAR * CFG['horizons']['EMBARGO_BARS']

PRICE = ['pct_from_ath', 'days_since_ath', 'pct_from_low_1y', 'range_width_90d', 'range_width_180d', 'base_duration_days',
         'close_ema300', 'close_ema1200', 'ema300_slope_5d', 'ema1200_slope_30d', 'breakout_180', 'breakout_540',
         'breakout_1080', 'higher_lows_6w']
VOLA = ['atr_pct', 'atr_pct_rank_1y', 'bb_width', 'bb_width_rank_1y']
FLOW = ['qv_20d_180d', 'log_qv_30d', 'obv_slope_30d', 'taker_buy_7d', 'taker_buy_30d', 'taker_buy_chg',
        'log_trades_per_day', 'qv_per_trade_7d_90d']
RS = ['rs_7d', 'rs_30d', 'rs_90d', 'rs_7d_rank', 'rs_30d_rank', 'rs_90d_rank', 'sector_ret_30d_ex_self', 'sector_rank_30d']
DERIV = ['funding_7d', 'funding_30d', 'has_futures', 'days_since_futures', 'coin_age_days']
DOWN_X = ['rsi14', 'rsi_divergence', 'volume_climax', 'funding_pos_share_7d']
# Macro / regime variables are NOT model inputs (spec section 1: macro only as a market-state filter).
REGIME = ['btc_trend', 'netliq_chg_4w', 'netliq_chg_13w']
FEATS = {'up': PRICE + VOLA + FLOW + RS + DERIV,
         'down': PRICE + VOLA + FLOW + RS + DERIV + DOWN_X}


def fold_masks(day, ts, valid_frac):
    """Boolean masks (fit, valid) for a test block starting at ts. `day` = decision day (close of day)."""
    day = pd.DatetimeIndex(day)
    end_of_label = day + pd.Timedelta(days=1) + H_TD                 # last bar of the label window (upper bound)
    train = (end_of_label < ts) & (day < ts - EMB_TD)
    d_tr = day[train]
    if len(d_tr) == 0:
        return train, train
    v_start = d_tr.min() + (d_tr.max() - d_tr.min()) * (1 - valid_frac)
    valid = train & (day >= v_start)
    fit = train & (day + pd.Timedelta(days=1) + H_TD < v_start) & (day < v_start - EMB_TD)
    return fit, valid


def _impute(Xtr, Xte_list):
    med = Xtr.median()
    flags = [c for c in Xtr.columns if Xtr[c].isna().any()]
    def tf(X):
        Z = X.fillna(med).fillna(0.0)                                 # median of TRAIN rows only (no zero-fill of meaning)
        for c in flags:
            Z[c + '_isna'] = X[c].isna().astype(float)
        return Z
    return tf(Xtr), [tf(X) for X in Xte_list]


def rule_score(df, rule, thr):
    """Rule baseline: a coin-day passes when every condition holds; score = number of conditions met (ties broken
    by the last condition's value) so common_eval can order candidates."""
    ok = np.ones(len(df), bool)
    n = np.zeros(len(df))
    for feat, op, q in rule:
        t = thr[feat]
        c = (df[feat] >= t) if op == '>=' else (df[feat] <= t)
        c = c.fillna(False).values
        ok &= c
        n += c
    last = df[rule[-1][0]].rank(pct=True).fillna(0).values
    return np.where(ok, 10 + n + last, n + 0.1 * last)


def fit_rule_thresholds(train, rule):
    """Thresholds re-fit on TRAIN rows only: quantile q of each feature among the train POSITIVES."""
    return {f: float(train.loc[train['_y'] == 1, f].quantile(q)) for f, op, q in rule}


def run_walkforward(panel, target, models=('rule', 'logistic', 'xgb'), rule=None, label_col=None, verbose=True):
    """Returns OOF scores per model for all test rows (also unlabeled ones, which still produce signals) and
    per-fold info (thresholds from the validation slice)."""
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ycol = label_col or target
    feats = FEATS[target]
    p = panel.copy()
    p['_y'] = p[ycol]
    day = pd.DatetimeIndex(p['day'])
    out = p[['day', 'symbol', 'close', 'sector', ycol] + [c for c in ['btc_trend', 'log_qv_30d'] if c in p]].copy()
    out = out.rename(columns={ycol: 'y'})
    for m in models:
        out[m] = np.nan
    info = []
    prm = dict(CFG['new']['model_params']['xgb'])
    es = prm.pop('early_stopping_rounds')
    vf = CFG['new']['model_params']['VALID_FRAC']
    top = CFG['new']['SIGNAL_TOP_FRAC']
    for ts, te in blocks(CFG):
        test = (day >= ts) & (day < te)
        fit, valid = fold_masks(day, ts, vf)
        fit &= p['_y'].notna().values
        valid &= p['_y'].notna().values
        fi = {'test_start': ts, 'test_end': te, 'n_fit': int(fit.sum()), 'n_valid': int(valid.sum()),
              'n_test': int(test.sum()), 'pos_fit': int(p.loc[fit, '_y'].sum()), 'pos_valid': int(p.loc[valid, '_y'].sum()),
              'pos_test': int(np.nansum(p.loc[test, '_y'])), 'fit_last_day': day[fit].max(), 'valid_first_day': day[valid].min()}
        Xf, Xv, Xt = p.loc[fit, feats], p.loc[valid, feats], p.loc[test, feats]
        yf, yv = p.loc[fit, '_y'].values.astype(int), p.loc[valid, '_y'].values.astype(int)
        if 'xgb' in models:
            spw = (len(yf) - yf.sum()) / max(1, yf.sum())
            m = XGBClassifier(**prm, scale_pos_weight=spw, n_jobs=12, verbosity=0, eval_metric='aucpr',
                              early_stopping_rounds=es)
            m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
            sv, st = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
            out.loc[test, 'xgb'] = st
            fi['xgb_best_iter'] = int(m.best_iteration)
            fi['xgb_thr'] = float(np.quantile(sv, 1 - top))
            fi['_xgb_model'] = m
        if 'logistic' in models:
            (Zf, [Zv, Zt]) = _impute(Xf, [Xv, Xt])
            sc = StandardScaler().fit(Zf)
            lr = LogisticRegression(class_weight='balanced', **CFG['new']['model_params']['logistic'])
            lr.fit(sc.transform(Zf), yf)
            sv, st = lr.predict_proba(sc.transform(Zv))[:, 1], lr.predict_proba(sc.transform(Zt))[:, 1]
            out.loc[test, 'logistic'] = st
            fi['logistic_thr'] = float(np.quantile(sv, 1 - top))
        if 'rule' in models and rule is not None:
            trn = p.loc[fit | valid]
            thr = fit_rule_thresholds(trn, rule)
            out.loc[test, 'rule'] = rule_score(p.loc[test], rule, thr)
            fi['rule_thr'] = 10.0                                      # passes all conditions
            fi['rule_thresholds'] = thr
        if verbose:
            print(f"  [{target}] fold {ts.date()}: fit {fi['n_fit']:,} ({fi['pos_fit']} pos, <= {fi['fit_last_day'].date()}) "
                  f"valid {fi['n_valid']:,} ({fi['pos_valid']} pos) test {fi['n_test']:,} ({fi['pos_test']} pos)", flush=True)
        fi['coins_with_event_test'] = int(p.loc[test & (p['_y'] == 1).values, 'symbol'].nunique())
        if fi['coins_with_event_test'] < CFG['new']['MIN_EVENTS_PER_FOLD']:
            print(f"  WARNING: fold {ts.date()} has events in only {fi['coins_with_event_test']} coins (< MIN_EVENTS_PER_FOLD)")
        info.append(fi)
    return out[day >= blocks(CFG)[0][0]], info
