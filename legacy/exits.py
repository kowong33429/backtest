"""
exits.py — Shared LONG-trade exit engine (used by entry_backtest.py and
portfolio_backtest.py so both report the same trades).

Why: a fixed +100% TP caps the big runners (a coin that goes 10x is sold at 2x).
The 'short' mode replaces the TP with the SHORT model as the exit signal:

  exit when  short_prob >= short_thr                  (the short model says "top")
        AND  Close <= trade_peak * (1 - min_dd)       (price has ALREADY rolled over)

The pullback confirmation matters: on its own the short model fires repeatedly
INSIDE strong rallies (it cannot tell "still ripping" from "topped"), which would
sell a 10x runner far too early. Requiring the price to be min_dd below the
trade's running peak filters those out; the short signal then decides whether a
pullback is the real top or just noise (a plain trailing stop at min_dd is much
worse — normal pullbacks inside the trend shake it out).

TIMING (AGENTS.md N-1 rule)
  * Entry: signal row s (features use bars <= s-1) -> buy at Open[s+1].
  * Resting stops (hard SL / trailing): level is computed from bars < j and filled
    at min(Open[j], level) when Low[j] touches it (a gap down fills at the Open).
  * Signal exits (short model, exit model): decided with information up to the
    CLOSE of bar j and filled at Open[j+1]. The probability used is the one on row
    j+1, whose features only see bars <= j.

Modes
  barrier  : TP +tp / SL -sl / time stop (the original rule, unchanged).
  trailing : hard SL, plus a trail `trail` below the running peak High, armed once
             the trade is up `activate`.
  short    : hard SL + short-model exit above (+ optional trailing as a backstop).
  model    : hard SL + exit when a dedicated exit-model prob >= exit_thr
             (+ optional min_dd confirmation), see exit_model.py.
"""
import numpy as np
import pandas as pd


def resolve_long(o, hi, lo, cl, dt, s, cfg, horizon_bars, fee, slip, exit_p=None):
    """
    Simulate ONE long trade for a signal at bar s.

    exit_p : optional float array aligned to the coin's bars — the exit-signal
             probability for each row (short-model prob for mode='short', exit-model
             prob for mode='model'); NaN = no prediction available on that bar.

    Returns dict(entry_i, exit_i, entry_px, exit_px, outcome, net, peak) or None.
    """
    n = len(o)
    entry_i = s + 1
    if entry_i >= n:
        return None
    px = o[entry_i]
    if not (px > 0 and np.isfinite(px)):
        return None
    end = min(entry_i + horizon_bars, n - 1)
    mode = cfg['mode']
    exit_i, exit_px, outcome = end, cl[end], 'TIME'

    if mode == 'barrier':
        up, dn = px * (1 + cfg['tp']), px * (1 - cfg['sl'])
        for j in range(entry_i, end + 1):
            if lo[j] <= dn:                      # same-bar ambiguity -> SL (conservative)
                exit_i, exit_px, outcome = j, dn, 'SL'
                break
            if hi[j] >= up:
                exit_i, exit_px, outcome = j, up, 'TP'
                break
        peak = None
    else:
        hard = px * (1 - cfg['sl'])
        trail = cfg.get('trail') or 0.0
        armed_at = px * (1 + (cfg.get('activate') or 0.0))
        thr = cfg.get('exit_thr', np.inf)
        min_dd = cfg.get('min_dd') or 0.0
        peak = px
        armed = trail > 0 and armed_at <= px
        for j in range(entry_i, end + 1):
            # 1) resting stops, levels known before bar j
            stop = hard
            if armed:
                stop = max(stop, peak * (1 - trail))
            if lo[j] <= stop:
                exit_i, exit_px = j, min(o[j], stop)
                outcome = 'TRAIL' if stop > hard else 'SL'
                break
            peak = max(peak, hi[j])
            if trail > 0 and peak >= armed_at:
                armed = True
            # 2) signal exit at the close of j -> fill at Open[j+1]
            if mode in ('short', 'model') and exit_p is not None and j + 1 <= end:
                p = exit_p[j + 1]
                if np.isfinite(p) and p >= thr and cl[j] <= peak * (1 - min_dd):
                    exit_i, exit_px, outcome = j + 1, o[j + 1], 'SHORT' if mode == 'short' else 'EXIT'
                    break

    net = (exit_px / px - 1.0) - 2 * (fee + slip)
    return {'entry_i': entry_i, 'exit_i': exit_i, 'entry_px': float(px),
            'exit_px': float(exit_px), 'outcome': outcome, 'net': float(net),
            'peak': None if peak is None else float(peak)}


def load_exit_probs(path, prob_col='oof_prob_xgb'):
    """{symbol: pd.Series(prob, index=Date)} from an OOF-style parquet."""
    df = pd.read_parquet(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=[prob_col])
    return {s: g.set_index('Date')[prob_col] for s, g in df.groupby('symbol')}


def align_probs(dates, series):
    """Map a {Date: prob} series onto a coin's bar dates (NaN where missing)."""
    if series is None:
        return np.full(len(dates), np.nan)
    return pd.Series(pd.DatetimeIndex(dates)).map(series).astype(float).values
