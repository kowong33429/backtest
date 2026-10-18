"""
rotation_backtest.py — Cross-sectional ROTATION: hold the top-K coins ranked by the
buy model's out-of-sample probability, instead of trading every bar that crosses a
threshold with a fixed TP.

WHY
---
The bar-by-bar entry backtest asks "is this bar an entry?" and then needs a fixed
take-profit to get out. Rotation asks a relative question every rebalance day:
"which K coins look best RIGHT NOW?" A coin is held for as long as it stays near the
top of the ranking (hysteresis) and sold when it drops out — winners run without a
TP, laggards are recycled. A market-breadth filter moves the book to BTC / cash when
most alts are below their 50-day average (the 2022 / 2025 alt bear markets).

RULES (AGENTS.md N-1)
---------------------
  Data     : 4H CSVs aggregated to DAILY bars (Open = first 4H Open, Close = last 4H Close).
  Score    : buy-model OOF prob (last 4H row of each day; that row's features see bars
             <= the previous 4H bar), smoothed over --smooth days.
  Eligible : >= 60 days of history, score present, regime filter ON for the day.
  Regime   : breadth = share of eligible coins with Close > their 50d SMA; trade alts
             only if breadth > --breadth (default 0.5).
  Select   : every --rebal days, keep held coins still ranked <= K * --hyst, fill the
             remaining slots with the best-ranked new coins. Equal slots of 1/K;
             empty slots are cash, or BTC (if --fallback btc and BTC > its 100d SMA).
  Timing   : everything above uses information up to the CLOSE of day d; the target
             weights are traded at the OPEN of day d+1. Daily P&L = weights x
             (Open[d+1] -> Open[d+2]) return, minus cost x drift-adjusted turnover.
  Costs    : --cost per side on traded notional (default 0.15% = fee 0.10% + slip 0.05%).

LEADER SLEEVE (--leaders N, off by default)
--------------------------------------------
The breadth filter switches the core off in alt bear markets, which is exactly when
a lone leader can still run 10x (ZEC Sep-Nov 2025 rallied while < 15% of alts were
above their 50d SMA, and its buy-model prob had faded). The leader sleeve buys
relative-strength leaders on their OWN trend, in any market regime:
  Eligible : >= --lead-min-age days of history (filters fresh-listing pumps),
             --lead-lookback-day return > 0 AND > BTC's over the same window,
             Close > its 50d SMA.
  Select   : top-N by that return, rebalanced with the core's --rebal / --hyst.
  Exit     : dropped from the ranking / gate fails, or the daily Close falls
             --lead-trail below its highest Close since entry (then blocked for
             --lead-cooldown days).
  Sizing   : 1/K per leader, but ONLY from cash the core leaves unused (never
             shrinks the core, never double-holds a core coin), before the BTC fallback.

PRESET b6 (--preset b6) — result of the iteration log (each change kept only if it improved
full-period Calmar without hurting either the 2020-23 or 2024-26 half by > 10%):
  1. Leaders enter only within 25% of their 60d high and are held while Close >= 75% of
     it — widened to 65% once the position is up >= 100% (lets 10x runners breathe).
  2. A second, smaller leader sleeve for coins aged 90-365 days (2 slots of 5%).
  3. BTC fallback only when BTC > SMA50 AND > SMA200.
  4. Portfolio vol target 80% (30d realized vol of the book, info <= close d-1).
  5. Core also needs BTC > SMA100 (cuts bear-market rallies, e.g. 2022).

PRESET b7 (--preset b7) = b6 with 3 leaders (not 5) and a 70% vol target. Picked under a
stricter rule: also had to raise the median Calmar over 10 random 70% coin subsets.
PRESET b7-ens = b7 averaged over leader near-high bands {20%, 25%, 30%}: b7's 25% band
is a sharp optimum (20% / 30% are much worse), so averaging the three books gives up some
backtest return for not depending on that one value.

Equity is marked to market daily (it is a fully-invested weight book), so the max
drawdown is directly comparable to portfolio_backtest.py's MTM drawdown.

CAVEAT: the coin universe is the trend_scanner basket (coins that HAD big rallies),
so absolute returns are survivorship-inflated for every strategy run on it; compare
strategies on the same universe, not against the outside world.

Usage:
    python core/rotation_backtest.py
    python core/rotation_backtest.py --k 10 --rebal 7 --hyst 2 --breadth 0.5 --fallback btc
    python core/rotation_backtest.py --start 2024-01-01            # out-of-sample window only
    python core/rotation_backtest.py --leaders 5                   # + leader sleeve (catches lone 10x runners)
    python core/rotation_backtest.py --preset b6                   # best iteration (see PRESET b6)
    python core/rotation_backtest.py --preset b7                   # final pick (see PRESET b7)
    python core/rotation_backtest.py --preset b7-ens               # b7, robust to the near-high band
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd


def load_daily(base, symbols, interval='4h', root='data'):
    """{O,C}: daily DataFrames (index=day, cols=symbol) built from the 4H CSVs under base/root."""
    suffix = f'_{interval}_full.csv'
    csvs = {os.path.basename(c).replace(suffix, '').upper(): c
            for c in glob.glob(os.path.join(base, root, '*', f'*{suffix}'))}
    O, C = {}, {}
    for s in symbols:
        if s not in csvs:
            continue
        d = pd.read_csv(csvs[s], usecols=['Date', 'Open', 'Close'], parse_dates=['Date'])
        g = d.sort_values('Date').set_index('Date').resample('D')
        O[s], C[s] = g['Open'].first(), g['Close'].last()
    return pd.DataFrame(O).sort_index(), pd.DataFrame(C).sort_index()


def select_topk(rank, k, rebal, hyst):
    """rank: 1 = best, NaN = not eligible (info <= close of day d).
    Returns weights decided at close d (NOT yet shifted)."""
    R = rank.values
    n, m = R.shape
    W = np.zeros((n, m))
    held = np.zeros(m, bool)
    for i in range(n):
        rk = R[i]
        elig = np.isfinite(rk)
        if i % rebal == 0:
            keep = held & elig & (np.where(elig, rk, np.inf) <= k * hyst)
            cand = np.where(elig & ~keep)[0]
            new = cand[np.argsort(rk[cand])][:max(0, k - keep.sum())]
            held = keep.copy()
            held[new] = True
        else:
            held &= elig            # a coin that stops trading / regime turns off is dropped
        W[i, held] = 1.0 / k
    return pd.DataFrame(W, index=rank.index, columns=rank.columns)


def trail_stop(W, C, trail, cooldown):
    """W: weights decided at close d (unshifted). Zero a position once the daily Close is
    `trail` below its highest Close since entry; block re-entry for `cooldown` days."""
    Wv = W.values.copy()
    Cv = C[W.columns].values
    for j in range(Wv.shape[1]):
        peak, blocked, inpos = np.nan, -1, False
        for i in range(Wv.shape[0]):
            if i <= blocked:
                Wv[i, j], inpos = 0.0, False
                continue
            if Wv[i, j] > 0:
                c = Cv[i, j]
                if not inpos:
                    peak, inpos = c, True
                peak = np.nanmax([peak, c])
                if c <= peak * (1 - trail):
                    Wv[i, j], inpos, blocked = 0.0, False, i + cooldown
            else:
                inpos = False
    return pd.DataFrame(Wv, index=W.index, columns=W.columns)


def leader_sleeve(C, btc_c, k, lookback, min_age, max_age, rebal, hyst, near_high, win_at, win_x, cooldown,
                  mask=None):
    """Profit-aware leader sleeve (0/1 holdings decided at close d, UNSHIFTED).
    Eligible : age in [min_age, max_age), `lookback`d return > 0 and > BTC's, Close > SMA50.
    Entry    : at a rebalance, top-k by return AND Close >= (1-near_high) x 60d-high Close.
    Exit     : daily if no longer eligible, or Close < (1-x) x 60d-high with x = near_high, widened
               to win_x once Close >= (1+win_at) x entry Close (then blocked `cooldown` days);
               at a rebalance if rank > k*hyst."""
    age = C.notna().cumsum()
    mom = C / C.shift(lookback) - 1
    ok = ((age >= min_age) & mom.notna() & (mom > 0) & mom.gt(btc_c / btc_c.shift(lookback) - 1, axis=0)
          & (C > C.rolling(50).mean()))
    if max_age:
        ok &= age < max_age
    if mask is not None:
        ok &= mask
    rank = mom.where(ok).rank(axis=1, ascending=False).values
    Cv, Hv = C.values, C.rolling(60, min_periods=20).max().values
    n, m = rank.shape
    W = np.zeros((n, m))
    held = np.zeros(m, bool)
    entry = np.full(m, np.nan)
    blocked = np.full(m, -1)
    for i in range(n):
        rk = rank[i]
        el = np.isfinite(rk)
        x = np.where(Cv[i] >= (1 + win_at) * entry, win_x, near_high)
        broke = held & (Cv[i] < (1 - x) * Hv[i])
        blocked[broke] = i + cooldown
        out = held & (~el | broke)
        held &= ~out
        entry[out] = np.nan
        if i % rebal == 0:
            keep = held & (np.where(el, rk, np.inf) <= k * hyst)
            entry[held & ~keep] = np.nan
            held = keep
            cand = np.where(el & ~held & (Cv[i] >= (1 - near_high) * Hv[i]) & (blocked < i))[0]
            new = cand[np.argsort(rk[cand])][:max(0, k - held.sum())]
            held[new] = True
            entry[new] = Cv[i, new]
        W[i, held] = 1.0
    return pd.DataFrame(W, index=C.index, columns=C.columns)


def breakout_sleeve(C, V, k, lookback, vol_x, min_usd, min_age, near_high, win_at, win_x, cooldown, rank_by='vx'):
    """Base-breakout sleeve (0/1 holdings decided at close d, UNSHIFTED), checked DAILY.
    Entry : age >= min_age, Close > highest Close of the previous `lookback` days, 7d/90d quote-volume
            ratio >= vol_x and 30d median daily quote volume >= min_usd (a $ liquidity floor instead of a
            volume RANK — a quiet coin breaking out of a long base is usually not yet a top-volume coin).
            Several candidates on one day -> highest volume ratio first (rank_by='vx') or OLDEST coin
            first (rank_by='age'); at most k positions.
    Hold  : until Close < (1-x) x its 60d-high Close (x = near_high, widened to win_x once Close >=
            (1+win_at) x entry) or Close < SMA50; then blocked `cooldown` days."""
    age = C.notna().cumsum()
    brk = C > C.rolling(lookback, min_periods=lookback).max().shift(1)
    vx = V.rolling(7, min_periods=5).mean() / V.rolling(90, min_periods=60).mean()
    liq = V.rolling(30, min_periods=20).median() >= min_usd
    entry_ok = (brk & (vx >= vol_x) & liq & (age >= min_age)).values
    vxv = (age if rank_by == 'age' else vx).values
    Cv, Hv, Sv = C.values, C.rolling(60, min_periods=20).max().values, C.rolling(50).mean().values
    n, m = Cv.shape
    W = np.zeros((n, m))
    held = np.zeros(m, bool)
    entry = np.full(m, np.nan)
    blocked = np.full(m, -1)
    for i in range(n):
        x = np.where(Cv[i] >= (1 + win_at) * entry, win_x, near_high)
        out = held & ((Cv[i] < (1 - x) * Hv[i]) | ~(Cv[i] > Sv[i]) | ~np.isfinite(Cv[i]))
        blocked[out] = i + cooldown
        held &= ~out
        entry[out] = np.nan
        slots = k - held.sum()
        if slots > 0:
            cand = np.where(entry_ok[i] & ~held & (blocked < i))[0]
            cand = cand[np.argsort(-np.nan_to_num(vxv[i, cand]))][:slots]
            held[cand] = True
            entry[cand] = Cv[i, cand]
        W[i, held] = 1.0
    return pd.DataFrame(W, index=C.index, columns=C.columns)


def load_quote_volume(base, symbols, root):
    """Daily quote volume (USDT) from the 4H CSVs that have a Quote_Volume column (data_all)."""
    V = {}
    for s in symbols:
        p = os.path.join(base, root, s.lower(), f'{s}_4h_full.csv')
        if os.path.exists(p):
            d = pd.read_csv(p, nrows=1)
            if 'Quote_Volume' in d.columns:
                d = pd.read_csv(p, usecols=['Date', 'Quote_Volume'], parse_dates=['Date'])
                V[s] = d.set_index('Date')['Quote_Volume'].resample('D').sum(min_count=1)
    return pd.DataFrame(V).sort_index()


def fill_idle(W, Wl, size):
    """Add a sleeve (0/1 holdings) at `size` per position, only from cash the book leaves unused."""
    Wl = (Wl > 0).astype(float) * size
    Wl = Wl.where(W == 0, 0.0)                                       # never double-hold a coin
    free = (1 - W.sum(axis=1)).clip(lower=0)
    scale = (free / Wl.sum(axis=1).replace(0, np.nan)).clip(upper=1).fillna(0)
    return W + Wl.mul(scale, axis=0)


def run_book(W, O, C, cost):
    """W: weights HELD on day d (open d -> open d+1). Returns daily returns and turnover."""
    O = O[W.columns]
    r = O.shift(-1) / O - 1.0
    # data ends while held (delisting): exit at that day's last Close (conservative proxy)
    r = r.where(r.notna(), (C[W.columns] / O - 1.0).where(O.notna())).fillna(0.0)
    prev = W.shift(1).fillna(0.0)
    grown = prev * (1 + r.shift(1).fillna(0.0))
    total = grown.sum(axis=1) + (1 - prev.sum(axis=1))
    drift = grown.div(total.replace(0, np.nan), axis=0).fillna(0.0)
    turnover = (W - drift).abs().sum(axis=1)
    return (W * r).sum(axis=1) - cost * turnover, turnover


def summarize(eq, pr, turnover, label):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    ye = eq.resample('YE').last()
    yr = ye.pct_change()
    yr.iloc[0] = ye.iloc[0] - 1
    print(f"\n  [{label}] {eq.index[0].date()} -> {eq.index[-1].date()} ({yrs:.1f}y)")
    print(f"    Final multiple : x{eq.iloc[-1]:,.2f}   (${eq.iloc[-1] * 10000:,.0f} from $10,000)")
    print(f"    CAGR           : {cagr * 100:+.1f}%")
    print(f"    Max DD (MTM)   : {dd.min() * 100:.1f}%  on {dd.idxmin().date()}")
    print(f"    Calmar / Sharpe: {cagr / -dd.min():.2f} / {pr.mean() / pr.std() * np.sqrt(365):.2f}")
    print(f"    Turnover / yr  : {turnover.sum() / yrs:.1f}x")
    print("    By year        : " + ", ".join(f"{d.year} {v * 100:+.0f}%" for d, v in yr.items()))


def get_parser():
    ap = argparse.ArgumentParser(description="Top-K rotation on the buy model's OOF prob")
    ap.add_argument('--oof', default='data/model/v2/m1_100pct_60d/entry_oof.parquet')
    ap.add_argument('--prob-col', default='oof_prob_xgb')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--rebal', type=int, default=7, help='rebalance every N days')
    ap.add_argument('--hyst', type=float, default=2.0, help='keep a holding while rank <= k*hyst')
    ap.add_argument('--smooth', type=int, default=7, help='rolling mean of the daily prob (days)')
    ap.add_argument('--min-prob', type=float, default=0.0)
    ap.add_argument('--breadth', type=float, default=0.5,
                    help='trade alts only if this share of coins is above its 50d SMA (0 = off)')
    ap.add_argument('--fallback', choices=['cash', 'btc'], default='btc')
    ap.add_argument('--min-age', type=int, default=60, help='days of history before a coin is eligible')
    ap.add_argument('--leaders', type=int, default=0, help='leader-sleeve size (0 = off; 5 = tested)')
    ap.add_argument('--lead-lookback', type=int, default=60, help='days of return used to rank leaders')
    ap.add_argument('--lead-min-age', type=int, default=365, help='days of history a leader needs')
    ap.add_argument('--lead-trail', type=float, default=0.35, help='trailing stop on daily closes')
    ap.add_argument('--lead-cooldown', type=int, default=30, help='days blocked after a trail stop')
    ap.add_argument('--lead-near-high', type=float, default=0.0,
                    help='> 0: profit-aware leaders (entry/hold within this of the 60d high) instead of --lead-trail')
    ap.add_argument('--lead-win-at', type=float, default=1.0, help='widen the hold band once up this much')
    ap.add_argument('--lead-win-x', type=float, default=0.35, help='hold band after --lead-win-at')
    ap.add_argument('--young', type=int, default=0, help='slots of the young-coin leader sleeve (0 = off)')
    ap.add_argument('--young-min-age', type=int, default=90)
    ap.add_argument('--young-size', type=float, default=0.05, help='weight per young-sleeve position')
    ap.add_argument('--core-btc-sma', type=int, default=0, help='core also needs BTC > SMA(n) (0 = off)')
    ap.add_argument('--fallback-sma', default='100', help="BTC fallback needs BTC above these SMAs, e.g. '50,200'")
    ap.add_argument('--vol-target', type=float, default=0.0, help='annualised vol target for the book (0 = off)')
    ap.add_argument('--vol-lookback', type=int, default=30)
    ap.add_argument('--preset', choices=['b6', 'b7', 'b7-ens', 'btc-trend', 'greedy', 'greedy-alts'], default=None,
                    help='apply a tested configuration')
    ap.add_argument('--no-core', action='store_true', help='skip the buy-prob core (sleeves / BTC fallback only)')
    ap.add_argument('--ensemble-near-high', default=None,
                    help="average the books built with these leader near-high bands, e.g. '0.2,0.25,0.3'")
    ap.add_argument('--cost', type=float, default=0.0015, help='per side (fee + slippage)')
    ap.add_argument('--start', default=None, help='default: first OOF day')
    ap.add_argument('--end', default=None)
    ap.add_argument('--root', default='data', help="price-data folder ('data' basket or 'data_all' universe)")
    ap.add_argument('--breakout', type=int, default=0, help='slots of the base-breakout sleeve (0 = off; needs root=data_all)')
    ap.add_argument('--bo-lookback', type=int, default=90, help='breakout above the highest Close of N days')
    ap.add_argument('--bo-vol-x', type=float, default=1.3, help='7d/90d quote-volume ratio needed')
    ap.add_argument('--bo-min-usd', type=float, default=3e6, help='30d median daily quote volume floor (USDT)')
    ap.add_argument('--bo-size', type=float, default=0.10, help='weight per breakout position')
    ap.add_argument('--bo-min-age', type=int, default=365, help='days of history a breakout coin needs')
    ap.add_argument('--bo-rank', choices=['vx', 'age'], default='vx', help='priority when several coins break out')
    ap.add_argument('--universe-top', type=int, default=0,
                    help='only coins in the point-in-time top-N by 30d volume (needs <root>/volume_rank.parquet)')
    ap.add_argument('--out-dir', default='data/model/rotation')
    return ap


def apply_preset(args):
    if args.preset in ('b6', 'b7', 'b7-ens'):
        args.leaders, args.lead_near_high, args.lead_win_at, args.lead_win_x = 5, 0.25, 1.0, 0.35
        args.young, args.young_min_age, args.young_size = 2, 90, 0.05
        args.core_btc_sma, args.fallback_sma, args.vol_target = 100, '50,200', 0.8
    if args.preset in ('b7', 'b7-ens'):
        args.leaders, args.vol_target = 3, 0.7
    if args.preset == 'b7-ens':
        args.ensemble_near_high = '0.2,0.25,0.3'
    if args.preset == 'btc-trend':
        # hold BTC while it is above its SMA50 AND SMA200, else cash — the benchmark to beat on data_all
        args.no_core, args.leaders, args.young, args.breakout = True, 0, 0, 0
        args.fallback, args.fallback_sma, args.vol_target = 'btc', '50,200', 0.0
    if args.preset in ('greedy', 'greedy-alts'):
        # catch as many big alt runs as possible: every liquid 90d breakout, 1% each (up to 100), exits as b7
        args.no_core, args.leaders, args.young = True, 0, 0
        args.lead_near_high, args.lead_win_at, args.lead_win_x, args.lead_cooldown = 0.25, 1.0, 0.35, 30
        args.breakout, args.bo_size, args.bo_lookback, args.bo_vol_x = 100, 0.01, 90, 1.3
        args.bo_min_age, args.bo_rank, args.bo_min_usd = 90, 'vx', 3e6
        args.fallback, args.fallback_sma, args.vol_target = 'btc', '50,200', 0.7
        if args.preset == 'greedy-alts':
            args.fallback = 'cash'

    return args


def load_inputs(args, base):
    """OOF probs + daily O/C for the universe (+ BTC)."""
    P = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    oof = pd.read_parquet(P(args.oof), columns=['Date', 'symbol', args.prob_col])
    oof['Date'] = pd.to_datetime(oof['Date'])
    oof = oof.dropna(subset=[args.prob_col])
    oof['day'] = oof['Date'].dt.normalize()
    syms = sorted(oof['symbol'].unique())
    O, C = load_daily(base, syms + ['BTCUSDT'], root=getattr(args, 'root', 'data'))
    out = {'oof': oof, 'syms': syms, 'O': O, 'C': C}
    if getattr(args, 'breakout', 0):
        alts = [c for c in C.columns if c != 'BTCUSDT']
        if getattr(args, 'universe_top', 0):      # dynamic universe: every coin with data can break out
            alts = sorted({os.path.basename(d).upper() for d in glob.glob(os.path.join(base, args.root, '*'))
                           if os.path.isdir(d)} - {'BTCUSDT', 'MODEL', 'PAPER'})
            O2, C2 = load_daily(base, [a for a in alts if a not in C.columns], root=args.root)
            out['O'], out['C'] = O.join(O2, how='outer'), C.join(C2, how='outer')
        out['V'] = load_quote_volume(base, alts, args.root)
    return out


def build_weights(args, data, symbols=None):
    """Target weights HELD each day (already shifted to N-1, vol-targeted). `symbols` restricts the
    alt universe (used for robustness tests on random coin subsets)."""
    oof, O, C = data['oof'], data['O'], data['C']
    if symbols is not None:
        keep = [s for s in symbols if s in C.columns] + ['BTCUSDT']
        O, C, oof = O[keep], C[keep], oof[oof['symbol'].isin(keep)]
    btc_c = C['BTCUSDT']
    O_alt, C_alt = O.drop(columns='BTCUSDT'), C.drop(columns='BTCUSDT')

    # Daily score = last 4H prob of the day (its features see bars before that 4H row).
    prob = (oof.sort_values('Date').groupby(['day', 'symbol'])[args.prob_col].last()
            .unstack().reindex(index=C_alt.index, columns=C_alt.columns))
    score = prob.rolling(args.smooth, min_periods=1).mean() if args.smooth > 1 else prob

    elig = C_alt.notna().cumsum() >= args.min_age
    univ = None
    if getattr(args, 'universe_top', 0):
        # point-in-time liquid universe: rank by 30d median quote volume known at the start of the day
        vr = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          args.root, 'volume_rank.parquet'))
        univ = (vr.reindex(index=C_alt.index, columns=C_alt.columns) <= args.universe_top).fillna(False)
        elig = elig & univ
    ok = elig & score.notna() & (score >= args.min_prob)
    if args.breadth > 0:
        breadth = (C_alt > C_alt.rolling(50).mean()).where(elig).mean(axis=1)
        ok &= np.repeat((breadth > args.breadth).values[:, None], ok.shape[1], axis=1)
    if args.core_btc_sma:
        ok &= np.repeat((btc_c > btc_c.rolling(args.core_btc_sma).mean()).values[:, None], ok.shape[1], axis=1)
    W = select_topk(score.where(ok).rank(axis=1, ascending=False), args.k, args.rebal, args.hyst)
    if getattr(args, 'no_core', False):
        W = W * 0.0
    if args.leaders > 0 and args.lead_near_high > 0:
        Wl = leader_sleeve(C_alt, btc_c, args.leaders, args.lead_lookback, args.lead_min_age, None, args.rebal,
                           args.hyst, args.lead_near_high, args.lead_win_at, args.lead_win_x, args.lead_cooldown,
                           mask=univ)
        W = fill_idle(W, Wl, 1.0 / args.k)
        if args.young > 0:
            Wy = leader_sleeve(C_alt, btc_c, args.young, args.lead_lookback, args.young_min_age, args.lead_min_age,
                               args.rebal, args.hyst, args.lead_near_high, args.lead_win_at, args.lead_near_high,
                               args.lead_cooldown, mask=univ)
            W = fill_idle(W, Wy, args.young_size)
    elif args.leaders > 0:
        age = C_alt.notna().cumsum()
        mom = C_alt / C_alt.shift(args.lead_lookback) - 1
        btc_mom = btc_c / btc_c.shift(args.lead_lookback) - 1
        okl = ((age >= args.lead_min_age) & mom.notna() & (mom > 0) & mom.gt(btc_mom, axis=0)
               & (C_alt > C_alt.rolling(50).mean()))
        if univ is not None:
            okl &= univ
        Wl = select_topk(mom.where(okl).rank(axis=1, ascending=False), args.leaders, args.rebal, args.hyst)
        Wl = (trail_stop(Wl, C_alt, args.lead_trail, args.lead_cooldown) > 0).astype(float) / args.k
        Wl = Wl.where(W == 0, 0.0)                                   # never double-hold a core coin
        free = (1 - W.sum(axis=1)).clip(lower=0)
        scale = (free / Wl.sum(axis=1).replace(0, np.nan)).clip(upper=1).fillna(0)
        W = W + Wl.mul(scale, axis=0)                                # leaders only use idle cash
    if getattr(args, 'breakout', 0) and 'V' in data:
        Vb = data['V'].reindex(index=C_alt.index, columns=C_alt.columns)
        Wb = breakout_sleeve(C_alt, Vb, args.breakout, args.bo_lookback, args.bo_vol_x, args.bo_min_usd,
                             args.bo_min_age, args.lead_near_high or 0.25, args.lead_win_at, args.lead_win_x,
                             args.lead_cooldown, rank_by=args.bo_rank)
        W = W.reindex(columns=C_alt.columns, fill_value=0.0)
        W = fill_idle(W, Wb, args.bo_size)
    W['BTCUSDT'] = 0.0
    if args.fallback == 'btc':
        btc_up = pd.Series(True, index=btc_c.index)
        for n_ in (int(x) for x in str(args.fallback_sma).split(',')):
            btc_up &= btc_c > btc_c.rolling(n_).mean()
        W['BTCUSDT'] = ((1 - W.drop(columns='BTCUSDT').sum(axis=1)) * btc_up).clip(lower=0)
    W = W.shift(1).fillna(0.0)          # decided at close d -> held from open d+1  (N-1)
    if args.vol_target > 0:
        # Book return of day t (open t -> open t+1) is only known at open t+1, so the size used
        # from open d uses returns up to day d-2, i.e. prices up to open d-1 (strict N-1).
        r_raw, _ = run_book(W, O.loc[W.index], C.loc[W.index], 0.0)
        vol = r_raw.rolling(args.vol_lookback, min_periods=10).std().shift(2) * np.sqrt(365)
        W = W.mul((args.vol_target / vol).clip(upper=1.0).fillna(1.0), axis=0)

    return W


def main():
    args = apply_preset(get_parser().parse_args())
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    P = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    out_dir = P(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    data = load_inputs(args, base)
    oof, syms, O, C = data['oof'], data['syms'], data['O'], data['C']
    if args.ensemble_near_high:
        import copy
        Ws = []
        for x in (float(v) for v in args.ensemble_near_high.split(',')):
            a = copy.copy(args)
            a.lead_near_high = x
            Ws.append(build_weights(a, data))
        cols = sorted(set().union(*[w.columns for w in Ws]))
        W = sum(w.reindex(columns=cols, fill_value=0.0) for w in Ws) / len(Ws)
        print(f"  ENSEMBLE: average of {len(Ws)} books, leader near-high bands {args.ensemble_near_high}")
    else:
        W = build_weights(args, data)

    start = pd.Timestamp(args.start) if args.start else oof['day'].min()
    W = W.loc[start:args.end]
    pr, turnover = run_book(W, O.loc[W.index], C.loc[W.index], args.cost)
    eq = (1 + pr).cumprod()

    print("=" * 72)
    print(f"  ROTATION | top-{args.k} by {args.prob_col} (smooth {args.smooth}d) | rebal {args.rebal}d | "
          f"hyst {args.hyst} | breadth>{args.breadth} | fallback {args.fallback} | cost {args.cost*100:.2f}%/side")
    if args.leaders and args.lead_near_high > 0:
        print(f"  + LEADERS: top-{args.leaders} by {args.lead_lookback}d return, age >= {args.lead_min_age}d, entry/hold within "
              f"{args.lead_near_high:.0%} of 60d high ({args.lead_win_x:.0%} once +{args.lead_win_at:.0%})"
              + (f" | YOUNG: {args.young} x {args.young_size:.0%}, age {args.young_min_age}-{args.lead_min_age}d" if args.young else ''))
        print(f"  + core needs BTC > SMA{args.core_btc_sma} | BTC fallback above SMA {args.fallback_sma}"
              + (f" | vol target {args.vol_target:.0%}" if args.vol_target else ''))
    elif args.leaders:
        print(f"  + LEADERS: top-{args.leaders} by {args.lead_lookback}d return (> BTC, > SMA50, age >= "
              f"{args.lead_min_age}d), trail {args.lead_trail:.0%}, cooldown {args.lead_cooldown}d, idle cash only")
    print(f"  Universe: {len(syms)} coins from {os.path.basename(args.oof)}")
    print("=" * 72)
    summarize(eq, pr, turnover, 'FULL')
    alt = W.drop(columns='BTCUSDT').sum(axis=1)
    print(f"    Avg exposure   : alts {alt.mean():.0%} | BTC {W['BTCUSDT'].mean():.0%} | cash {(1 - alt - W['BTCUSDT']).mean():.0%}")

    # Holdings log: one row per (coin, holding spell) with entry/exit open prices.
    rows = []
    for s in W.columns:
        h = W[s] > 0
        if not h.any():
            continue
        spell = (h != h.shift(fill_value=False)).cumsum()[h]
        for _, days in spell.groupby(spell):
            d0, d1 = days.index[0], days.index[-1] + pd.Timedelta(days=1)
            px0, px1 = O.at[d0, s], O[s].get(d1, C[s].get(days.index[-1]))
            rows.append({'symbol': s, 'entry_date': d0, 'exit_date': d1, 'days': len(days),
                         'entry_open': px0, 'exit_open': px1, 'ret': px1 / px0 - 1})
    hold = pd.DataFrame(rows).sort_values('entry_date')
    hold.to_csv(os.path.join(out_dir, 'rotation_holdings.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame({'equity': eq, 'daily_ret': pr, 'turnover': turnover,
                  'alt_exposure': alt, 'btc_weight': W['BTCUSDT']}).to_csv(
        os.path.join(out_dir, 'rotation_equity.csv'), index_label='Date')
    alts = hold[hold['symbol'] != 'BTCUSDT']
    if len(alts):
        print(f"\n  Alt holding spells: {len(alts)} | median {alts['days'].median():.0f}d | "
              f"win {(alts['ret'] > 0).mean():.0%} | > +100%: {(alts['ret'] > 1).sum()} | best "
              f"{alts.loc[alts['ret'].idxmax(), 'symbol']} {alts['ret'].max() * 100:+.0f}%")
    else:
        print("\n  No alt holdings (BTC / cash only).")
    print(f"  Files: {out_dir}\\rotation_equity.csv, rotation_holdings.csv")
    print("=" * 72)


if __name__ == '__main__':
    main()
