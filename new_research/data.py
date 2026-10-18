"""Loads the cleaned NEW 4H data (data_new/, strictly before the holdout cut)."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config, holdout_start   # noqa: E402

CFG = load_config()
CUT = holdout_start(CFG)


def load_coin(sym):
    p = os.path.join(ROOT, 'data_new', sym.lower(), f'{sym}_4h.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, parse_dates=['Date'])
    if d['Date'].max() >= CUT:                                    # the holdout must never be in data_new
        raise RuntimeError(f'{sym}: data_new contains rows at/after the holdout cut')
    d = d[d['Volume'] > 0]                                        # B1.3 (same rule as common_eval)
    # bad ticks: cap wicks at 3x / 0.33x of the bar body (49 bars in 49 coins, mostly first bars of a spot listing);
    # close-to-close moves are kept (real crashes such as OMUSDT 2025-04).
    body_hi, body_lo = d[['Open', 'Close']].max(axis=1), d[['Open', 'Close']].min(axis=1)
    d['High'] = d['High'].clip(upper=3 * body_hi)
    d['Low'] = d['Low'].clip(lower=body_lo / 3)
    return d.drop_duplicates('Date').sort_values('Date').set_index('Date')


def metadata():
    return pd.read_csv(os.path.join(ROOT, 'data', 'coin_metadata.csv'),
                       parse_dates=['spot_first', 'futures_first', 'listed_first', 'delist_date'])
