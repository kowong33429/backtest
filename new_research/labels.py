"""Section 4: UP / DOWN event labels (they look forward; they are Y, never features).

UP   at bar t: within H bars after t, High reaches >= UP_TARGET x Close[t] before Low falls below
     (1 - MAE_LIMIT) x Close[t]. Same-bar ambiguity counts as the adverse move first (conservative).
DOWN at bar t: Low reaches <= DOWN_TARGET x Close[t] before High rises above (1 + MAE_LIMIT) x Close[t].
A label is NaN when its window is cut off by the end of the development data (never use holdout prices); a coin
that was delisted before the cut gets a definite outcome (the window ended with the coin).
"""
import numpy as np
import pandas as pd


def _first(mask):
    k = int(mask.argmax())
    return k if mask[k] else -1


def label_bars(d, idx, H, up_t, dn_t, mae, delisted):
    """d: 4H frame; idx: integer positions to label. Returns (up, down, up_hit_bar, dn_hit_bar, up_peak)."""
    c, h, l = d['Close'].values, d['High'].values, d['Low'].values
    n = len(c)
    up = np.full(len(idx), np.nan); dn = np.full(len(idx), np.nan)
    upk = np.full(len(idx), -1); dnk = np.full(len(idx), -1)
    peak = np.full(len(idx), np.nan)
    for q, t in enumerate(idx):
        end = t + H
        full = end <= n - 1
        if not full and not delisted:
            continue                                   # window would need prices we do not have -> NaN
        hi, lo = h[t + 1:min(end, n - 1) + 1], l[t + 1:min(end, n - 1) + 1]
        if len(hi) == 0:
            if delisted:
                up[q] = dn[q] = 0
            continue
        p = c[t]
        peak[q] = hi.max() / p
        a_up, a_bad = _first(hi >= up_t * p), _first(lo < (1 - mae) * p)
        up[q] = 1.0 if a_up >= 0 and (a_bad < 0 or a_up < a_bad) else 0.0
        upk[q] = t + 1 + a_up if up[q] == 1 else -1
        b_dn, b_bad = _first(lo <= dn_t * p), _first(hi > (1 + mae) * p)
        dn[q] = 1.0 if b_dn >= 0 and (b_bad < 0 or b_dn < b_bad) else 0.0
        dnk[q] = t + 1 + b_dn if dn[q] == 1 else -1
    return up, dn, upk, dnk, peak


def events(daily, col, cooldown_days, kind):
    """Event-level labels: consecutive positive days of one coin (gaps <= cooldown) form one run = one event.
    Start = swing low (UP: lowest close in the run) / swing high (DOWN: highest close)."""
    rows = []
    for sym, g in daily[daily[col] == 1].groupby('symbol'):
        g = g.sort_index()
        run = (g.index.to_series().diff() > pd.Timedelta(days=cooldown_days)).cumsum()
        for _, r in g.groupby(run.values):
            k = r['close'].idxmin() if kind == 'up' else r['close'].idxmax()
            rows.append({'symbol': sym, 'start': k, 'start_close': r.loc[k, 'close'], 'run_first': r.index.min(),
                         'run_last': r.index.max(), 'run_days': len(r), 'sector': r['sector'].iloc[0],
                         'peak_multiple': r.loc[k, f'{kind}_peak'] if f'{kind}_peak' in r else np.nan})
    ev = pd.DataFrame(rows)
    if len(ev):
        ev['year'] = pd.to_datetime(ev['start']).dt.year
    return ev
