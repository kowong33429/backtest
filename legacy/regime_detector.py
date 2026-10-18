"""
regime_detector.py — Market-regime filters for the entry strategy (leakage-safe)

Builds an equal-weight BASKET INDEX from all traded coins, then labels each bar's
market regime two ways, BOTH strictly causal (AGENTS.md Rule #1):

  1. SuperTrend (transparent baseline) — ATR-band trend flip on the index.
     Shifted +1 bar so an entry at bar T only uses the regime known at T-1.

  2. Gaussian HMM (walk-forward, FILTERED state) — fits a hidden-regime model on
     the index's daily log-returns + realized vol, refit periodically on an
     EXPANDING past-only window. The regime at day t is the FILTERED posterior
     P(state_t | data up to t) — obtained as predict_proba(X[:t+1])[-1], which
     for the LAST observation equals the filtered (not smoothed) estimate, so it
     uses NO future data. "Favorable" = regimes whose trained emission mean
     return is positive (so the bull/bear meaning is learned from the past only).

Output: data/model/regime.parquet with the 4H grid + supertrend_bull + hmm_favorable.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------
# 1. Equal-weight basket index (synthetic OHLC on the common 4H grid)
# ----------------------------------------------------------------------------
def build_basket_index(base_dir, basket_json=None):
    csvs = {os.path.basename(c).replace('_4h_full.csv', '').upper(): c
            for c in glob.glob(os.path.join(base_dir, 'data', '*', '*_4h_full.csv'))}
    if basket_json and os.path.exists(basket_json):
        with open(basket_json) as f:
            want = {s.upper() for s in json.load(f)}
        csvs = {s: p for s, p in csvs.items() if s in want}

    rets = {}
    for sym, path in csvs.items():
        d = pd.read_csv(path, usecols=['Date', 'Open', 'High', 'Low', 'Close'])
        d['Date'] = pd.to_datetime(d['Date'])
        d = d.sort_values('Date').drop_duplicates('Date').set_index('Date')
        prev = d['Close'].shift(1)
        rets[sym] = pd.DataFrame({
            'o': d['Open'] / prev - 1, 'h': d['High'] / prev - 1,
            'l': d['Low'] / prev - 1, 'c': d['Close'] / prev - 1,
        })
    # Cross-sectional mean return per bar (equal weight, coins available that bar).
    o = pd.concat([r['o'] for r in rets.values()], axis=1).mean(axis=1)
    h = pd.concat([r['h'] for r in rets.values()], axis=1).mean(axis=1)
    l = pd.concat([r['l'] for r in rets.values()], axis=1).mean(axis=1)
    c = pd.concat([r['c'] for r in rets.values()], axis=1).mean(axis=1)
    idx = pd.DataFrame({'o': o, 'h': h, 'l': l, 'c': c}).sort_index().dropna()

    close = 100 * (1 + idx['c']).cumprod()
    prevc = close.shift(1).fillna(100)
    index = pd.DataFrame({
        'Open': prevc * (1 + idx['o']),
        'High': prevc * (1 + idx[['o', 'h', 'c']].max(axis=1)),
        'Low':  prevc * (1 + idx[['o', 'l', 'c']].min(axis=1)),
        'Close': close,
    })
    index.index.name = 'Date'
    print(f"[Regime] basket index: {len(index)} 4H bars from {len(rets)} coins "
          f"({index.index.min().date()} ~ {index.index.max().date()})")
    return index


# ----------------------------------------------------------------------------
# 2. SuperTrend regime (causal via shift)
# ----------------------------------------------------------------------------
def supertrend_regime(index_daily, length=10, multiplier=3.0):
    import pandas_ta as ta
    st = ta.supertrend(index_daily['High'], index_daily['Low'],
                       index_daily['Close'], length=length, multiplier=multiplier)
    # The direction column: 1 = uptrend, -1 = downtrend.
    dir_col = [c for c in st.columns if c.startswith('SUPERTd')][0]
    bull = (st[dir_col] > 0).astype(int)
    return bull.rename('supertrend_bull')


# ----------------------------------------------------------------------------
# 3. HMM regime (walk-forward, filtered, favorable = positive trained mean)
# ----------------------------------------------------------------------------
def hmm_regime(index_daily, n_states=3, refit_every=30, min_train=365,
               vol_window=20, seed=42):
    from hmmlearn.hmm import GaussianHMM
    close = index_daily['Close']
    logret = np.log(close / close.shift(1))
    vol = logret.rolling(vol_window).std()
    feat = pd.DataFrame({'ret': logret, 'vol': vol}).dropna()
    X = feat.values
    dates = feat.index
    n = len(X)

    favorable = np.full(n, np.nan)   # 1 = favorable regime at that day (filtered)
    model = None
    last_fit = -10**9
    fav_states = set()

    for t in range(n):
        if t < min_train:
            continue
        # Refit on expanding PAST-only window (data strictly up to t).
        if model is None or (t - last_fit) >= refit_every:
            try:
                m = GaussianHMM(n_components=n_states, covariance_type='full',
                                n_iter=100, random_state=seed)
                m.fit(X[:t])                      # past only
                model = m
                last_fit = t
                # Favorable = states whose emission MEAN return (dim 0) > 0.
                fav_states = {s for s in range(n_states) if m.means_[s, 0] > 0}
            except Exception:
                pass
        if model is None:
            continue
        # Filtered posterior of state_t: last row of predict_proba on data up to t.
        try:
            post = model.predict_proba(X[:t + 1])[-1]
            state_t = int(np.argmax(post))
            favorable[t] = 1.0 if state_t in fav_states else 0.0
        except Exception:
            favorable[t] = np.nan

    out = pd.Series(favorable, index=dates, name='hmm_favorable')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basket', default='data/scan_results/scanned_priority_coins.json')
    ap.add_argument('--n-states', type=int, default=3)
    ap.add_argument('--st-length', type=int, default=10)
    ap.add_argument('--st-mult', type=float, default=3.0)
    ap.add_argument('--out', default='data/model/regime.parquet')
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    basket_json = os.path.join(base, args.basket)
    index = build_basket_index(base, basket_json)

    # Daily index for regime computation (regimes are daily, mapped to 4H later).
    daily = index.resample('1D').agg({'Open': 'first', 'High': 'max',
                                      'Low': 'min', 'Close': 'last'}).dropna()

    st = supertrend_regime(daily, args.st_length, args.st_mult)
    hmm = hmm_regime(daily, n_states=args.n_states)

    reg_daily = pd.concat([st, hmm], axis=1)
    # CAUSAL mapping: shift the daily regime by 1 day so a bar on day D uses the
    # regime known at the close of day D-1.
    reg_daily = reg_daily.shift(1)

    # Map daily regime onto the 4H grid (backward asof — only past info).
    grid = index.reset_index()[['Date']].copy()
    rd = reg_daily.reset_index().rename(columns={'index': 'Date'})
    rd['Date'] = pd.to_datetime(rd['Date'])
    reg_4h = pd.merge_asof(grid.sort_values('Date'), rd.sort_values('Date'),
                           on='Date', direction='backward')
    reg_4h['mkt_close'] = index['Close'].values

    out_path = os.path.join(base, args.out)
    reg_4h.to_parquet(out_path, index=False)

    # Diagnostics: how often is each filter "on"?
    stp = reg_4h['supertrend_bull'].mean()
    hmp = reg_4h['hmm_favorable'].mean()
    print(f"[Regime] SuperTrend bullish: {stp*100:.0f}% of bars | "
          f"HMM favorable: {hmp*100:.0f}% of bars")
    print(f"[Regime] saved -> {out_path}")
    # Show regime by year to sanity-check 2025.
    reg_4h['year'] = pd.to_datetime(reg_4h['Date']).dt.year
    print("\n  % of bars ALLOWED to trade, by year:")
    print((reg_4h.groupby('year')[['supertrend_bull', 'hmm_favorable']].mean() * 100).round(0).to_string())


if __name__ == '__main__':
    main()
