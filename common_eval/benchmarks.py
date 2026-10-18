"""BTC_HOLD, EQUAL_WEIGHT (monthly rebalance) and RANDOM (matched to a reference signal set)."""
import numpy as np
import pandas as pd
from .backtest import Signals
from .metrics import full_metrics


def _t0(mk, cfg):
    return int(np.searchsorted(mk.times.values, np.datetime64(pd.Timestamp(cfg['time']['WF_START']))))


def btc_hold(mk, cfg):
    j = mk.sym_idx['BTCUSDT']
    t0 = _t0(mk, cfg)
    f = cfg['costs']['FEE_PER_SIDE']
    units = (1 - f) / (mk.O[t0, j] * (1 + mk.slip[t0, j]))
    c = pd.Series(mk.C[:, j], index=mk.times).ffill().values
    eq = pd.Series(units * c, index=mk.times)[t0:]
    eq.iloc[-1] *= (1 - mk.slip[-1, j]) * (1 - f)                      # liquidate at the end
    return eq


def equal_weight(mk, cfg):
    """All point-in-time eligible coins, equal weight, rebalanced on the first bar of each month. Costs on turnover."""
    f = cfg['costs']['FEE_PER_SIDE']
    t0 = _t0(mk, cfg)
    C = pd.DataFrame(mk.C, index=mk.times).ffill().values
    months = pd.Series(mk.times).dt.to_period('M').values
    eq = np.ones(mk.T)
    units = np.zeros(mk.N)
    E = 1.0
    n_hold = []
    for t in range(t0, mk.T):
        if t == t0 or months[t] != months[t - 1]:
            val = units * np.nan_to_num(C[t - 1] if t > t0 else 0)       # value before the rebalance (prior close)
            E = val.sum() if t > t0 else 1.0
            sel = np.where(mk.elig[t])[0]
            tgt = np.zeros(mk.N)
            if len(sel):
                tgt[sel] = E / len(sel)
            turn_cost = (np.abs(tgt - val) * (f + np.nan_to_num(mk.slip[t], nan=0.0))).sum()
            tgt = tgt * (1 - turn_cost / max(E, 1e-12))
            units = np.zeros(mk.N)
            units[sel] = tgt[sel] / mk.O[t, sel]
            n_hold.append(len(sel))
        eq[t] = (units * np.nan_to_num(C[t])).sum()
    s = pd.Series(eq, index=mk.times)[t0:]
    s.attrs['avg_names'] = float(np.mean(n_hold))
    return s


def random_signals(mk, rng, rate, mode='rate', ref=None, mask=None):
    """mode 'rate': every eligible (bar, coin) is a signal with prob `rate`, random score (matches frequency).
    mode 'timing': at every bar draw as many random eligible coins as `ref` has signals there (matches timing too)."""
    if mode == 'rate':
        m = mk.elig & mask & (rng.random(mk.elig.shape) < rate)
        t, j = np.nonzero(m)
        return Signals(mk, t, j, rng.random(len(t)), np.ones(len(t), int))
    ts, js = [], []
    cnt = np.diff(ref.start)
    for t in np.nonzero(cnt)[0]:
        cand = np.nonzero(mk.elig[t] & mask[t])[0]
        if len(cand) == 0:
            continue
        k = min(int(cnt[t]), len(cand))
        pick = rng.choice(cand, k, replace=False)
        ts.extend([t] * k)
        js.extend(pick)
    ts, js = np.array(ts), np.array(js)
    return Signals(mk, ts, js, rng.random(len(ts)), np.ones(len(ts), int))
