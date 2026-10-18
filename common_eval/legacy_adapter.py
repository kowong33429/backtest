"""Turns LEGACY model output (OOF probabilities) into (timestamp, symbol, score, side) signals for common_eval.

N-1: LEGACY row s carries features up to bar s-1 and LEGACY executes at Open[s+1]; the adapter therefore stamps the
signal with Date[s] + 1 bar (the execution bar), exactly reproducing LEGACY's own delay.
"""
import os
import numpy as np
import pandas as pd
from .config import ROOT, dev_cut

BAR = pd.Timedelta(hours=4)


def read_oof(path, prob_col='oof_prob_xgb'):
    o = pd.read_parquet(path)
    o['Date'] = pd.to_datetime(o['Date'])
    o = o.rename(columns={prob_col: 'prob'}) if prob_col in o.columns else o
    return o.dropna(subset=['prob'])[[c for c in ['Date', 'symbol', 'Entry_Label', 'Label_EndDate', 'prob'] if c in o.columns]]


def in_window(o, cfg):
    return o[(o['Date'] >= pd.Timestamp(cfg['time']['WF_START']) - BAR) & (o['Date'] < dev_cut(cfg) - BAR)]


def to_signals(o, thr, side, cfg):
    o = in_window(o, cfg)
    s = o[o['prob'] >= thr]
    return pd.DataFrame({'ts': s['Date'] + BAR, 'symbol': s['symbol'].values, 'score': s['prob'].values, 'side': side})


def pool_mask(mk, o, cfg):
    """(T, N) bool: bars/coins for which the model produced a score (execution-bar stamped). RANDOM draws from this."""
    o = in_window(o, cfg)
    t = np.fromiter((mk.idx(x) for x in o['Date'] + BAR), int, len(o))
    j = np.fromiter((mk.sym_idx.get(s, -1) for s in o['symbol']), int, len(o))
    ok = (t >= 0) & (j >= 0)
    m = np.zeros((mk.T, mk.N), bool)
    m[t[ok], j[ok]] = True
    return m


def label_cut_rows(o, cfg):
    """Rows whose label window is fully inside the development period (no holdout peeking in model metrics)."""
    o = in_window(o, cfg)
    if 'Label_EndDate' in o.columns:
        return o[o['Label_EndDate'] < dev_cut(cfg)]
    return o[o['Date'] < dev_cut(cfg) - pd.Timedelta(days=60)]
