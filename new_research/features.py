"""Section 6 features. Per coin on 4H bars using data up to and including bar t, then sampled once per UTC day at
the day's last bar (decision at the close of day D, execution at the Open of D+1).

Every rolling window ends at bar t; nothing is centred or shifted backwards. Cross-sectional features (relative
strength ranks, sector returns) are added on the daily panel in panel.py, using only same-day values.
"""
import numpy as np
import pandas as pd

BPD = 6  # 4H bars per day


def _days(n):
    return int(n * BPD)


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def _days_since(flag):
    """Bars since `flag` was last True (NaN until it has been True once)."""
    idx = np.arange(len(flag), dtype=float)
    last = pd.Series(np.where(flag.values, idx, np.nan), index=flag.index).ffill()
    return (idx - last) / BPD


def coin_features(d, listed_first=None, futures_first=None):
    o, h, l, c, v = d['Open'], d['High'], d['Low'], d['Close'], d['Volume']
    qv = d['Quote_Volume']
    f = pd.DataFrame(index=d.index)
    # --- price structure ---
    ath = h.cummax()
    f['pct_from_ath'] = c / ath - 1
    f['days_since_ath'] = _days_since(h >= ath)
    f['pct_from_low_1y'] = c / l.rolling(_days(365), min_periods=_days(90)).min() - 1
    for nd in (90, 180):
        f[f'range_width_{nd}d'] = h.rolling(_days(nd), min_periods=_days(nd) // 2).max() / \
            l.rolling(_days(nd), min_periods=_days(nd) // 2).min() - 1
    f['base_duration_days'] = _days_since(f['range_width_90d'] > 1.0)       # days of a <100% 90d range
    e300, e1200 = c.ewm(span=300, adjust=False, min_periods=150).mean(), c.ewm(span=1200, adjust=False, min_periods=600).mean()
    f['close_ema300'] = c / e300 - 1
    f['close_ema1200'] = c / e1200 - 1
    f['ema300_slope_5d'] = e300 / e300.shift(30) - 1
    f['ema1200_slope_30d'] = e1200 / e1200.shift(180) - 1
    for n in (180, 540, 1080):
        f[f'breakout_{n}'] = c / h.shift(1).rolling(n, min_periods=n // 2).max() - 1
    wk_low = l.rolling(42).min()
    hl = pd.concat([(wk_low.shift(42 * k) > wk_low.shift(42 * (k + 1))).astype(float) for k in range(6)], axis=1)
    f['higher_lows_6w'] = hl.sum(axis=1).where(wk_low.shift(42 * 6).notna())
    # --- volatility ---
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atrp = tr.rolling(14).mean() / c
    f['atr_pct'] = atrp
    f['atr_pct_rank_1y'] = atrp.rolling(_days(365), min_periods=_days(90)).rank(pct=True)
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    bbw = 4 * s20 / m20
    f['bb_width'] = bbw
    f['bb_width_rank_1y'] = bbw.rolling(_days(365), min_periods=_days(90)).rank(pct=True)
    # --- volume / order flow ---
    qv20, qv180 = qv.rolling(_days(20)).sum(), qv.rolling(_days(180), min_periods=_days(60)).sum()
    f['qv_20d_180d'] = (qv20 / 20) / (qv180 / (qv.rolling(_days(180), min_periods=_days(60)).count() / BPD))
    f['log_qv_30d'] = np.log10(qv.rolling(_days(30), min_periods=_days(20)).sum() / 30 + 1)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f['obv_slope_30d'] = (obv - obv.shift(_days(30))) / v.rolling(_days(30)).sum()
    tbr = d['Taker_Buy_Volume'] / v
    f['taker_buy_7d'] = tbr.rolling(_days(7)).mean()
    f['taker_buy_30d'] = tbr.rolling(_days(30)).mean()
    f['taker_buy_chg'] = f['taker_buy_7d'] - f['taker_buy_30d']
    tr7 = d['Trades'].rolling(_days(7)).sum() / 7
    f['log_trades_per_day'] = np.log10(tr7 + 1)
    qpt = qv.rolling(_days(7)).sum() / d['Trades'].rolling(_days(7)).sum()
    f['qv_per_trade_7d_90d'] = qpt / (qv.rolling(_days(90), min_periods=_days(30)).sum() / d['Trades'].rolling(_days(90), min_periods=_days(30)).sum())
    # --- returns (cross-sectional parts in panel.py) ---
    for nd in (7, 30, 90):
        f[f'ret_{nd}d'] = c / c.shift(_days(nd)) - 1
    # --- derivatives ---
    fr = d['Funding_Rate']                                                   # NaN before futures (no fillna)
    f['funding_7d'] = fr.rolling(_days(7), min_periods=_days(3)).mean()
    f['funding_30d'] = fr.rolling(_days(30), min_periods=_days(10)).mean()
    f['has_futures'] = d['has_futures'].astype(float)
    if futures_first is not None and pd.notna(futures_first):
        f['days_since_futures'] = ((d.index - futures_first).days).where(d.index >= futures_first)
    else:
        f['days_since_futures'] = np.nan
    first = listed_first if (listed_first is not None and pd.notna(listed_first)) else d.index[0]
    f['coin_age_days'] = (d.index - first).days
    # --- DOWN extras ---
    rsi = _rsi(c)
    f['rsi14'] = rsi
    f['rsi_divergence'] = (c / c.rolling(_days(30)).max()) - (rsi / rsi.rolling(_days(30)).max())
    dq = qv.rolling(BPD).sum()
    f['volume_climax'] = dq.rolling(_days(3)).max() / dq.rolling(_days(90), min_periods=_days(30)).median()
    f['funding_pos_share_7d'] = (fr > 0.0001).astype(float).where(fr.notna()).rolling(_days(7), min_periods=_days(3)).mean()
    return f.replace([np.inf, -np.inf], np.nan)


def daily_sample(f):
    """One row per UTC day: the last 4H bar of that day (decision at its close)."""
    g = f.groupby(f.index.normalize())
    out = g.tail(1).copy()
    out['bar_time'] = out.index
    out.index = out.index.normalize()
    return out
