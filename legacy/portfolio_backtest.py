"""
portfolio_backtest.py — Capital-constrained portfolio backtest of the entry model

Unlike entry_backtest.py (unlimited capital, fixed notional), this shares ONE
account across all coins. Signals are processed chronologically; a trade opens
only if there is free cash. When the book is full, new signals are SKIPPED (a
missed trade) — exactly what happens with a real, finite account.

RULES (same signal/exit as entry_backtest.py, AGENTS.md Rule #1)
  Signal   : oof_prob_xgb >= threshold (OUT-OF-SAMPLE, no leakage).
  Entry    : at the Open of the bar AFTER the signal.
  Exit     : TP +100% / SL -40% / time stop 30d (first touch; SL on ambiguity).
  Per coin : one open position at a time.
  Sizing   : alloc_frac of current equity (cash + open cost basis), capped by
             free cash; skip if the affordable size < min_trade. Compounding.
  Costs    : fee + slippage on both sides.

EXIT MODES (all simulated by exits.resolve_long, same engine as entry_backtest.py):
barrier (above), trailing, and short / model: no fixed TP, hard SL (--sl, alias
--init-sl), exit when the exit-signal prob (short model or exit model) >= --exit-thr
AND price is >= --min-dd below the trade peak.

DRAWDOWN: besides the realized curve (equity only moves when a trade closes), a
daily MARK-TO-MARKET curve is reported — open positions valued at each day's last
4H Close. With long holds (months) the realized max-DD badly understates the real
pain; the MTM figure is the one to size the account on.

    python core/portfolio_backtest.py --oof data/model/v2/m1_100pct_60d/entry_oof.parquet \
        --threshold 0.8 --exit-mode short --exit-probs data/model/short_v2/entry_oof.parquet \
        --exit-thr 0.5 --min-dd 0.20 --horizon-days 365 --out-dir data/model/v2/m1_short_exit
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exits import resolve_long, load_exit_probs, align_probs  # noqa: E402


def coin_arrays(csv):
    df = pd.read_csv(csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return (df['Open'].values, df['High'].values, df['Low'].values,
            df['Close'].values, df['Date'].values,
            {d: i for i, d in enumerate(df['Date'].values)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default='data/model/entry_oof.parquet')
    ap.add_argument('--prob-col', default='oof_prob_xgb')
    ap.add_argument('--threshold', type=float, default=0.70)
    ap.add_argument('--exit-mode', choices=['barrier', 'trailing', 'short', 'model'], default='barrier')
    ap.add_argument('--exit-probs', default=None, help='short/model: OOF parquet of the exit signal')
    ap.add_argument('--exit-prob-col', default='oof_prob_xgb')
    ap.add_argument('--exit-thr', type=float, default=0.5)
    ap.add_argument('--min-dd', type=float, default=0.20)
    ap.add_argument('--tp', type=float, default=1.00, help='barrier mode: take profit')
    ap.add_argument('--sl', '--init-sl', dest='sl', type=float, default=0.40,
                    help='stop loss (barrier) / hard stop floor (other modes)')
    ap.add_argument('--trail', type=float, default=None,
                    help='trailing stop below the peak (default: 0.25 in trailing mode, off otherwise)')
    ap.add_argument('--activate', type=float, default=0.0, help='arm the trail only after +this profit')
    ap.add_argument('--horizon-days', type=int, default=30, help='time cap (barrier=30; trailing can be larger)')
    ap.add_argument('--initial-equity', type=float, default=10000.0)
    ap.add_argument('--alloc-frac', type=float, default=0.10,
                    help='fraction of equity per trade (0.10 -> ~10 concurrent)')
    ap.add_argument('--min-trade', type=float, default=100.0)
    ap.add_argument('--fee', type=float, default=0.001)
    ap.add_argument('--slippage', type=float, default=0.0005)
    ap.add_argument('--out-dir', default='data/model')
    ap.add_argument('--tag', default='', help='suffix for output files to keep runs separate')
    ap.add_argument('--root', default='data', help="price-data folder ('data' or 'data_all')")
    ap.add_argument('--regime-file', default=None, help='parquet from regime_detector.py')
    ap.add_argument('--regime-col', default=None, help='column to gate on (e.g. supertrend_bull / hmm_favorable)')
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof = pd.read_parquet(os.path.join(base, args.oof))
    oof['Date'] = pd.to_datetime(oof['Date'])
    oof = oof.dropna(subset=[args.prob_col])
    sig = oof[oof[args.prob_col] >= args.threshold][['Date', 'symbol']].copy()

    csvs = {os.path.basename(c).replace('_4h_full.csv', '').upper(): c
            for c in glob.glob(os.path.join(base, args.root, '*', '*_4h_full.csv'))}
    med = (pd.read_csv(next(iter(csvs.values())), usecols=['Date'])['Date']
           .pipe(pd.to_datetime).diff().median())
    bpd = max(1, round(pd.Timedelta(days=1) / med))
    H = args.horizon_days * bpd
    if args.trail is None:
        args.trail = 0.25 if args.exit_mode == 'trailing' else 0.0
    cfg = {'mode': args.exit_mode, 'tp': args.tp, 'sl': args.sl, 'trail': args.trail,
           'activate': args.activate, 'exit_thr': args.exit_thr, 'min_dd': args.min_dd}
    exit_probs = None
    if args.exit_mode in ('short', 'model'):
        if not args.exit_probs:
            ap.error('--exit-probs is required for --exit-mode short/model')
        exit_probs = load_exit_probs(os.path.join(base, args.exit_probs), args.exit_prob_col)
        start = min(v.index.min() for v in exit_probs.values())
        sig = sig[sig['Date'] >= start]   # every trade must be able to see the exit signal
    ep_cache = {}

    # Precompute every candidate trade (price path is capital-independent).
    arrs = {}
    cands = []
    for sym, grp in sig.groupby('symbol'):
        if sym not in csvs:
            continue
        if sym not in arrs:
            arrs[sym] = coin_arrays(csvs[sym])
        o, hi, lo, cl, dt, d2p = arrs[sym]
        for d in grp['Date']:
            s = d2p.get(np.datetime64(d))
            if s is None:
                continue
            if exit_probs is not None and sym not in ep_cache:
                ep_cache[sym] = align_probs(dt, exit_probs.get(sym))
            r = resolve_long(o, hi, lo, cl, dt, s, cfg, H, args.fee, args.slippage,
                             exit_p=ep_cache.get(sym))
            if r:
                cands.append({'symbol': sym, 'signal_date': pd.Timestamp(d),
                              'entry_date': pd.Timestamp(dt[r['entry_i']]),
                              'exit_date': pd.Timestamp(dt[r['exit_i']]),
                              'net_ret': r['net'], 'outcome': r['outcome'], 'entry_px': r['entry_px']})
    # Optional market-regime gate: allow a signal only if the regime (known as of
    # the signal bar) permits trading. Backward-asof -> only past info.
    regime_fn = None
    if args.regime_file and args.regime_col:
        reg = pd.read_parquet(os.path.join(base, args.regime_file))
        reg['Date'] = pd.to_datetime(reg['Date'])
        reg = reg[['Date', args.regime_col]].dropna().sort_values('Date').reset_index(drop=True)
        rdates = reg['Date'].values
        rvals = reg[args.regime_col].values

        def regime_fn(when):
            pos = np.searchsorted(rdates, np.datetime64(when), side='right') - 1
            return pos >= 0 and rvals[pos] >= 0.5

    cands = sorted(cands, key=lambda x: x['entry_date'])
    print("=" * 72)
    print(f"  PORTFOLIO BACKTEST | ${args.initial_equity:,.0f} start | "
          f"{args.alloc_frac*100:.0f}% per trade | thr {args.threshold}")
    if args.exit_mode == 'barrier':
        print(f"  EXIT barrier: TP +{args.tp*100:.0f}% / SL -{args.sl*100:.0f}% / {args.horizon_days}d | "
              f"candidate signals: {len(cands)}")
    elif args.exit_mode == 'trailing':
        print(f"  EXIT trailing: {args.trail*100:.0f}% below peak, hard stop -{args.sl*100:.0f}%, "
              f"max {args.horizon_days}d | candidate signals: {len(cands)}")
    else:
        print(f"  EXIT {args.exit_mode}: prob >= {args.exit_thr} & >= {args.min_dd*100:.0f}% off peak, "
              f"hard stop -{args.sl*100:.0f}%"
              + (f", trail {args.trail*100:.0f}%" if args.trail > 0 else '')
              + f", max {args.horizon_days}d | candidate signals: {len(cands)}")
    print("=" * 72)

    # Event-driven portfolio pass.
    cash = args.initial_equity
    open_pos = []        # list of dicts with exit_date, size, net_ret, symbol
    busy_until = {}      # symbol -> exit_date of its last opened trade (same re-entry rule as entry_backtest)
    executed, skipped_cash, skipped_held, skipped_regime = [], 0, 0, 0
    equity_curve = []    # (date, realized_equity) recorded at each close

    def realized_equity():
        # cash + cost basis of still-open positions = initial + realized P&L.
        return cash + sum(p['size'] for p in open_pos)

    def close_due(now):
        nonlocal cash
        due = [p for p in open_pos if p['exit_date'] <= now]
        for p in sorted(due, key=lambda x: x['exit_date']):
            cash += p['size'] * (1 + p['net_ret'])       # return capital + P&L
            open_pos.remove(p)
            equity_curve.append((p['exit_date'], realized_equity()))

    for c in cands:
        close_due(c['entry_date'])
        if regime_fn is not None and not regime_fn(c['signal_date']):
            skipped_regime += 1
            continue
        if c['signal_date'] <= busy_until.get(c['symbol'], pd.Timestamp.min):
            skipped_held += 1
            continue
        size = args.alloc_frac * realized_equity()
        if size > cash:
            size = cash                      # use remaining cash for the last slot
        if size < args.min_trade:
            skipped_cash += 1                # book full / not enough cash
            continue
        cash -= size
        open_pos.append({'exit_date': c['exit_date'], 'size': size,
                         'net_ret': c['net_ret'], 'symbol': c['symbol']})
        busy_until[c['symbol']] = c['exit_date']
        c2 = dict(c); c2['size'] = size; c2['pnl_usd'] = size * c['net_ret']
        executed.append(c2)

    # Close everything left (in exit-date order so the curve is monotonic in time).
    for p in sorted(open_pos[:], key=lambda x: x['exit_date']):
        cash += p['size'] * (1 + p['net_ret'])
        open_pos.remove(p)
        equity_curve.append((p['exit_date'], realized_equity()))
    final_equity = cash

    td = pd.DataFrame(executed)
    if td.empty:
        print("No trades executed.")
        return
    td.to_csv(os.path.join(base, args.out_dir, f'portfolio_trades{tag}.csv'), index=False, encoding='utf-8-sig')
    ec = pd.DataFrame(equity_curve, columns=['Date', 'equity']).sort_values('Date')
    ec = ec.drop_duplicates('Date', keep='last').reset_index(drop=True)
    ec.to_csv(os.path.join(base, args.out_dir, f'portfolio_equity{tag}.csv'), index=False)

    # Daily mark-to-market equity: open positions valued at each day's last Close.
    days = pd.date_range(td['entry_date'].min().normalize(), td['exit_date'].max().normalize(), freq='D')
    mtm = np.zeros(len(days))
    daily_close = {}
    for sym in td['symbol'].unique():
        o_, h_, l_, c_, d_, _ = arrs[sym]
        daily_close[sym] = pd.Series(c_, index=pd.DatetimeIndex(d_)).resample('D').last().ffill()
    for _, t in td.iterrows():
        dc = daily_close[t['symbol']].reindex(days).ffill().values
        a = days.searchsorted(t['entry_date'].normalize())
        b = days.searchsorted(t['exit_date'].normalize())
        mtm[a:b] += t['size'] * (dc[a:b] / t['entry_px'] - 1.0)   # unrealized while open
        mtm[b:] += t['size'] * t['net_ret']                         # realized after exit
    mtm_eq = pd.Series(args.initial_equity + mtm, index=days)
    mtm_dd = (mtm_eq / mtm_eq.cummax() - 1.0)
    mtm_eq.rename('equity_mtm').to_frame().to_csv(
        os.path.join(base, args.out_dir, f'portfolio_equity_mtm{tag}.csv'), index_label='Date')

    # Metrics
    ret = final_equity / args.initial_equity - 1.0
    peak = ec['equity'].cummax()
    max_dd = ((ec['equity'] - peak) / peak).min()
    span_years = (td['exit_date'].max() - td['entry_date'].min()).days / 365.25
    cagr = (final_equity / args.initial_equity) ** (1/span_years) - 1 if span_years > 0 else float('nan')
    n = len(td)
    wr = (td['net_ret'] > 0).mean()
    oc = td['outcome'].value_counts().to_dict()

    print(f"\n  EXECUTED trades:   {n}")
    print(f"  Skipped (book full/no cash): {skipped_cash}   |  skipped (coin held): {skipped_held}"
          f"   |  skipped (regime off): {skipped_regime}")
    print(f"  Outcomes: {', '.join(f'{k} {v}' for k, v in sorted(oc.items()))}")
    print(f"  Win rate:          {wr*100:.1f}%")
    print(f"  Max concurrent:    {args.alloc_frac and round(1/args.alloc_frac)} (by sizing)")
    print(f"\n  Start equity:      ${args.initial_equity:,.0f}")
    print(f"  FINAL equity:      ${final_equity:,.0f}")
    print(f"  Total return:      {ret*100:+.1f}%   over ~{span_years:.1f} years")
    print(f"  CAGR:              {cagr*100:+.1f}% / year")
    print(f"  Max drawdown:      {max_dd*100:.1f}%  (realized, closed trades only)")
    print(f"  Max DD (MTM):      {mtm_dd.min()*100:.1f}%  on {mtm_dd.idxmin().date()} "
          f"(peak {mtm_eq[:mtm_dd.idxmin()].idxmax().date()})  <- open positions marked daily")
    yr_ret = mtm_eq.resample('YE').last().pct_change()
    yr_ret.iloc[0] = mtm_eq.resample('YE').last().iloc[0] / args.initial_equity - 1
    print("  Calendar-year return (MTM): " + ", ".join(f"{d.year} {v*100:+.0f}%" for d, v in yr_ret.items()))
    print(f"\n  Trades: {os.path.join(args.out_dir,f'portfolio_trades{tag}.csv')} | "
          f"Equity: {os.path.join(args.out_dir,f'portfolio_equity{tag}.csv')}")

    td['year'] = td['entry_date'].dt.year
    yr = td.groupby('year').agg(trades=('net_ret','size'),
                                win=('net_ret', lambda s:(s>0).mean()*100),
                                pnl=('pnl_usd','sum')).round(1)
    print("\n  By year:")
    print(yr.to_string())
    print("=" * 72)


if __name__ == '__main__':
    main()
