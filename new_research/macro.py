"""Steps B1.4-B1.7: macro / FRED / Fear&Greed snapshots (fetched only up to the holdout cut) and the regime table.

Each series is stored with its observation date AND the date it becomes usable (`avail`), i.e. the observation
date plus a publication lag of at least one day (AGENTS.md rule 2, spec section 1). Regime features (max 3 macro
variables, spec section 1/6): BTC trend from BTCUSDT 4H, net liquidity = WALCL - RRP, HY_SPREAD.
"""
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config, holdout_start   # noqa: E402

SNAP = os.path.join(ROOT, 'data', 'snapshots')
FRED = {  # name: (FRED id, publication lag in days)
    'US2Y': ('DGS2', 1), 'FED_BAL': ('WALCL', 2), 'RRP': ('RRPONTSYD', 1), 'HY_SPREAD': ('BAMLH0A0HYM2', 2),
}


def _save(df, name):
    os.makedirs(SNAP, exist_ok=True)
    p = os.path.join(SNAP, f'{name}_{dt.date.today():%Y%m%d}.csv')
    df.to_csv(p, index=False)
    return p


def fetch_all(start='2018-01-01'):
    cut = holdout_start(load_config())
    end = cut - pd.Timedelta(days=1)
    from pandas_datareader import data as web
    for name, (fid, lag) in FRED.items():
        s = web.DataReader(fid, 'fred', start, end)[fid].dropna()
        s = s[s.index < cut]
        df = pd.DataFrame({'obs_date': s.index, 'value': s.values})
        df['avail'] = df['obs_date'] + pd.Timedelta(days=lag)
        print(name, _save(df, name), len(df), df['obs_date'].min().date())
    import yfinance as yf
    d = yf.download('^IRX', start=start, end=str(end.date()), progress=False)['Close'].squeeze().dropna()
    idx = pd.to_datetime(d.index)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    df = pd.DataFrame({'obs_date': idx.normalize(), 'value': d.values})
    df = df[df['obs_date'] < cut]
    df['avail'] = df['obs_date'] + pd.Timedelta(days=1)               # US close known after the session
    print('US3M (^IRX, renamed)', _save(df, 'US3M'), len(df))
    js = requests.get('https://api.alternative.me/fng/?limit=0&format=json', timeout=30).json()['data']
    fg = pd.DataFrame({'obs_date': pd.to_datetime([int(x['timestamp']) for x in js], unit='s').normalize(),
                       'value': [int(x['value']) for x in js]})
    fg = fg[fg['obs_date'] < cut].sort_values('obs_date')
    fg['avail'] = fg['obs_date'] + pd.Timedelta(days=1)              # D+1 (spec B1.6)
    print('FearGreed', _save(fg, 'FearGreed'), len(fg))


def latest(name):
    fs = sorted(f for f in os.listdir(SNAP) if f.startswith(name + '_'))
    return pd.read_csv(os.path.join(SNAP, fs[-1]), parse_dates=['obs_date', 'avail'])


def asof(series_df, times):
    """Value usable at each timestamp (backward as-of on the availability date, ffill only)."""
    s = series_df.sort_values('avail')
    m = pd.merge_asof(pd.DataFrame({'t': pd.DatetimeIndex(times)}).sort_values('t'), s[['avail', 'value']],
                      left_on='t', right_on='avail', direction='backward')
    return m['value'].values


def regime_daily(days, btc_close_4h):
    """Daily regime table indexed by decision day D (info up to the close of D, used for the Open of D+1).
    btc_trend: BTCUSDT close / EMA1200 - 1 (4H bars) at the last bar of D."""
    ema = btc_close_4h.ewm(span=1200, adjust=False, min_periods=600).mean()
    tr = (btc_close_4h / ema) - 1                                    # > 0 = BTC above its EMA1200
    tr_d = tr.groupby(tr.index.normalize()).last().reindex(days)
    t_avail = days + pd.Timedelta(hours=23, minutes=59)                 # end of day D
    fed, rrp = latest('FED_BAL'), latest('RRP')
    fed_v, rrp_v = asof(fed, t_avail) * 1e6, asof(rrp, t_avail) * 1e9  # WALCL $mn, RRP $bn -> $
    nl = pd.Series(fed_v - rrp_v, index=days)
    hy = pd.Series(asof(latest('HY_SPREAD'), t_avail), index=days)
    out = pd.DataFrame({'btc_trend': tr_d.values,
                        'netliq_chg_4w': (nl / nl.shift(28) - 1).values,
                        'netliq_chg_13w': (nl / nl.shift(91) - 1).values,
                        'hy_spread': hy.values}, index=days)
    return out


if __name__ == '__main__':
    fetch_all()
