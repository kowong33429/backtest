"""Market data: cleaned 4H matrices, point-in-time liquidity, slippage. Never reads past the holdout cut."""
import os
import glob
import numpy as np
import pandas as pd
from .config import ROOT, dev_cut, guard_holdout


class Market:
    def __init__(self, times, symbols, O, H, L, C, qv30, cfg):
        self.times, self.symbols = times, list(symbols)
        self.O, self.H, self.L, self.C, self.qv30 = O, H, L, C, qv30
        self.cfg = cfg
        self.sym_idx = {s: i for i, s in enumerate(self.symbols)}
        self.t_idx = {t: i for i, t in enumerate(times)}
        self.T, self.N = O.shape
        mq = cfg['universe']['MIN_QUOTE_VOL_30D']
        self.elig = np.isfinite(O) & (qv30 >= mq)
        sc = cfg['costs']
        with np.errstate(invalid='ignore', divide='ignore'):
            mult = np.clip(np.sqrt(sc['SLIP_REF_QV'] / np.where(qv30 > 0, qv30, np.nan)), 1.0, sc['SLIP_MAX_MULT'])
        self.slip = sc['SLIPPAGE_BASE'] * np.where(np.isfinite(mult), mult, sc['SLIP_MAX_MULT'])

    def idx(self, ts):
        return self.t_idx.get(pd.Timestamp(ts), -1)


def _read(path, drop_zero):
    d = pd.read_csv(path)
    d['Date'] = pd.to_datetime(d['Date'])
    d = d.sort_values('Date').drop_duplicates('Date')
    if 'Quote_Volume' not in d.columns:
        d['Quote_Volume'] = d['Volume'] * d['Close']
    if drop_zero:
        d = d[d['Volume'] > 0]
    return d[['Date', 'Open', 'High', 'Low', 'Close', 'Quote_Volume']].set_index('Date')


def load_market(cfg, root=None, symbols=None, start='2021-10-01', suffix='_4h_full.csv'):
    """Cleaned matrices on the 4H grid [start, holdout cut). `symbols` = None -> every CSV under root."""
    root = root or cfg['universe']['data_root']
    cut = dev_cut(cfg)
    guard_holdout(cfg, cut - pd.Timedelta(hours=4))
    paths = {os.path.basename(p)[:-len(suffix)].upper(): p
             for p in glob.glob(os.path.join(ROOT, root, '*', '*' + suffix))}
    syms = sorted(paths) if symbols is None else sorted(s for s in symbols if s in paths)
    times = pd.date_range(pd.Timestamp(start), cut - pd.Timedelta(hours=4), freq='4h')
    drop0 = cfg['universe']['drop_zero_volume_bars']
    cols = {k: [] for k in 'OHLC'}
    daily = {}
    keep = []
    for s in syms:
        d = _read(paths[s], drop0)
        d = d[d.index < cut]
        if d.empty:
            continue
        # daily quote volume needs 30d history before `start`
        daily[s] = d['Quote_Volume'].resample('D').sum(min_count=1)
        w = d[d.index >= times[0]].reindex(times)
        if w['Open'].notna().sum() == 0:
            continue
        for k, c in zip('OHLC', ['Open', 'High', 'Low', 'Close']):
            cols[k].append(w[c].values)
        keep.append(s)
    O, H, L, C = (np.array(cols[k]).T for k in 'OHLC')
    V = pd.DataFrame(daily)[keep]
    med = V.rolling(30, min_periods=20).median().shift(1)         # previous days only
    day = times.normalize()
    q = med.reindex(med.index.union(day.unique())).ffill().reindex(day).values
    return Market(times, keep, O, H, L, C, q, cfg)
