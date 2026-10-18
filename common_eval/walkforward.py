"""Walk-forward folds with purge + embargo, and a generic retrain helper. Never loads data at/after the holdout."""
import os
import numpy as np
import pandas as pd
from .config import ROOT, dev_cut

NON_FEAT = {'Date', 'symbol', 'Entry_Label', 'Label_EndDate', 'Sample_Weight'}
SLOW_MACRO = ['CPI_YoY', 'CPI_MoM', 'M2_YoY', 'M2_MoM', 'FEDFUNDS', 'HY_SPREAD', 'YieldCurve_10Y_2Y']   # audit H2


def blocks(cfg):
    """[(test_start, test_end)] half-year blocks from WF_START up to the holdout start."""
    t = cfg['time']
    s, cut = pd.Timestamp(t['WF_START']), dev_cut(cfg)
    out = []
    while s < cut:
        e = min(s + pd.DateOffset(months=t['WF_BLOCK_MONTHS']), cut)
        out.append((s, e))
        s = e
    return out


def load_panel(path, cfg):
    p = pd.read_parquet(path)
    p = p[p['Date'] < dev_cut(cfg)].reset_index(drop=True)          # holdout rows never enter memory
    return p


def retrain_oof(panel, feats, cfg, fillna0=True, n_jobs=8, verbose=True, params=None):
    """Expanding-window walk-forward. Train rows: Date < test_start - EMBARGO and Label_EndDate < test_start."""
    from xgboost import XGBClassifier
    emb = pd.Timedelta(hours=4 * cfg['horizons']['EMBARGO_BARS'])
    X = panel[feats].astype('float32')
    if fillna0:
        X = X.fillna(0.0)
    y = panel['Entry_Label'].astype(int).values
    w = panel['Sample_Weight'].astype(float).values
    d = panel['Date'].values
    le = panel['Label_EndDate'].values
    oof = np.full(len(panel), np.nan)
    prm = dict(params or cfg['model']['xgb'])
    for ts, te in blocks(cfg):
        tr = np.where((d < np.datetime64(ts - emb)) & (le < np.datetime64(ts)))[0]
        ts_ = np.where((d >= np.datetime64(ts)) & (d < np.datetime64(te)))[0]
        if len(tr) == 0 or len(ts_) == 0 or y[tr].sum() == 0:
            continue
        spw = (len(tr) - y[tr].sum()) / max(1, y[tr].sum())
        m = XGBClassifier(**prm, scale_pos_weight=spw, n_jobs=n_jobs, verbosity=0, eval_metric='aucpr')
        m.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
        oof[ts_] = m.predict_proba(X.iloc[ts_])[:, 1]
        if verbose:
            print(f'   fold {ts.date()}..{te.date()}: train {len(tr):,} (<= {pd.Timestamp(d[tr].max()).date()}) test {len(ts_):,}', flush=True)
    out = panel[['Date', 'symbol', 'Entry_Label', 'Label_EndDate']].copy()
    out['prob'] = oof
    return out.dropna(subset=['prob'])
