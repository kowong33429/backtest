"""
entry_backtest_short.py — Honest OOS backtest of the SHORT entry model (mirror of
entry_backtest.py), with a THRESHOLD SWEEP to pick the operating point.

Trades the OUT-OF-SAMPLE probabilities from train_entry_model.py (entry_oof.parquet
for the SHORT panel) so there is NO in-sample leakage. Execution and exit follow
AGENTS.md Rule #1 and the SHORT label definition (labeler_short.py).

RULES (everything is the mirror of the long backtest)
-----------------------------------------------------
  Signal   : oof_prob_xgb >= threshold.
  Entry    : decision uses features up to bar n-1; OPEN A SHORT at the Open of
             bar n+1 (next bar after the signal) — the only unshifted price.
  Exit     : whichever comes first:
               * Take Profit : price FALLS to entry/(1+tp)   -> short P&L = +tp
                               (tp=1.00 -> price x0.5 -> +100%)
               * Stop Loss   : price RISES to entry/(1-sl)   -> short P&L = -sl
                               (sl=0.40 -> price x1.667 -> -40%)
               * Time stop   : horizon days -> exit at that bar's Close
             Same-bar TP&SL ambiguity resolves to SL (conservative).
  P&L      : SHORT, reciprocal/leveraged convention (user decision B):
               gross = entry_px / exit_px - 1
             so a price-halving = +100% and a price-doubling = -50%.
  Position : one open short per coin at a time; different coins concurrent.
  Costs    : fee + slippage on BOTH sides (AGENTS.md Rule #9).

Sizing: fixed notional per trade, so each trade's $ P&L = notional * net return.
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd


def compute_trend_ok(df, lookback, pullback):
    """
    Trend FILTER (variant 1): True on bars where a short is ALLOWED — i.e. price
    has ALREADY pulled back at least `pullback` from its trailing `lookback`-bar
    high, so we are NOT shorting into a fresh new high / parabola still ripping.

    Uses only data up to the signal bar s (High/Close rolling max shifted so bar s
    sees highs through s), consistent with the n-1 decision timing.
    """
    rollmax = df['High'].rolling(lookback, min_periods=lookback).max()
    # price is >= pullback below the recent peak -> rolled over -> short allowed
    allowed = (df['Close'] <= (1.0 - pullback) * rollmax)
    return allowed.fillna(False).values


def simulate_coin_short(df, sig_dates, tp, sl, horizon_bars, fee, slip, notional,
                        trend_ok=None):
    """Walk one coin's bars; SHORT at next Open on a signal, cover on TP/SL/time.

    trend_ok : optional bool array aligned to df positions; when given, a signal
               at bar s is SKIPPED unless trend_ok[s] is True (variant-1 filter).
    """
    df = df.sort_values('Date').reset_index(drop=True)
    o, hi, lo, cl = (df['Open'].values, df['High'].values,
                     df['Low'].values, df['Close'].values)
    dt = df['Date'].values
    date_to_pos = {d: i for i, d in enumerate(dt)}
    sig_pos = sorted(date_to_pos[d] for d in sig_dates if d in date_to_pos)

    trades = []
    cooldown_until = -1
    for s in sig_pos:
        if s <= cooldown_until:
            continue
        if trend_ok is not None and not trend_ok[s]:
            continue  # variant-1 trend filter: not pulled back from the high yet
        entry_i = s + 1  # execute at the NEXT bar's Open
        if entry_i >= len(df):
            continue
        entry_px = o[entry_i]
        if entry_px <= 0 or not np.isfinite(entry_px):
            continue
        # MIRROR: TP is BELOW entry (short wins), SL is ABOVE entry (short loses).
        dn, up = entry_px / (1 + tp), entry_px / (1 - sl)
        end = min(entry_i + horizon_bars, len(df) - 1)

        exit_i, exit_px, outcome = end, cl[end], 'TIME'
        for j in range(entry_i, end + 1):
            hit_up, hit_dn = hi[j] >= up, lo[j] <= dn
            if hit_up and hit_dn:
                exit_i, exit_px, outcome = j, up, 'SL'        # conservative
                break
            if hit_up:
                exit_i, exit_px, outcome = j, up, 'SL'
                break
            if hit_dn:
                exit_i, exit_px, outcome = j, dn, 'TP'
                break

        # Short P&L (reciprocal) after costs on both sides.
        gross = entry_px / exit_px - 1.0
        cost = 2 * (fee + slip)
        net = gross - cost
        trades.append({
            'entry_date': pd.Timestamp(dt[entry_i]),
            'entry_price': float(entry_px),
            'exit_date': pd.Timestamp(dt[exit_i]),
            'exit_price': float(exit_px),
            'bars_held': int(exit_i - entry_i),
            'outcome': outcome,
            'gross_ret': float(gross),
            'net_ret': float(net),
            'pnl_usd': float(net * notional),
        })
        cooldown_until = exit_i  # no re-entry until this short closes
    return trades


def summarize(td, notional, bpd):
    """Return a dict of the headline metrics for a trade DataFrame."""
    n = len(td)
    wins = td[td['net_ret'] > 0]
    losses = td[td['net_ret'] <= 0]
    gross_win = wins['net_ret'].sum()
    gross_loss = -losses['net_ret'].sum()
    oc = td['outcome'].value_counts().to_dict()
    return {
        'trades': n,
        'tp': int(oc.get('TP', 0)), 'sl': int(oc.get('SL', 0)),
        'time': int(oc.get('TIME', 0)),
        'win_rate': len(wins) / n * 100 if n else float('nan'),
        'avg_win': wins['net_ret'].mean() * 100 if len(wins) else float('nan'),
        'avg_loss': losses['net_ret'].mean() * 100 if len(losses) else float('nan'),
        'expectancy': td['net_ret'].mean() * 100 if n else float('nan'),
        'profit_factor': (gross_win / gross_loss) if gross_loss > 0 else float('inf'),
        'avg_days': td['bars_held'].mean() / bpd if n else float('nan'),
        'total_pnl': td['pnl_usd'].sum(),
        'ret_on_deployed': td['pnl_usd'].sum() / (n * notional) * 100 if n else float('nan'),
    }


def main():
    ap = argparse.ArgumentParser(description="Backtest the SHORT entry model (OOS) + threshold sweep")
    ap.add_argument('--oof', default='data/model/short/entry_oof.parquet')
    ap.add_argument('--prob-col', default='oof_prob_xgb')
    ap.add_argument('--thresholds', default='0.50,0.60,0.70,0.80,0.90',
                    help='comma-separated thresholds to sweep')
    ap.add_argument('--tp', type=float, default=1.00)
    ap.add_argument('--sl', type=float, default=0.40)
    ap.add_argument('--horizon-days', type=int, default=60)
    ap.add_argument('--fee', type=float, default=0.001, help='fee per side (0.001=0.1%%)')
    ap.add_argument('--slippage', type=float, default=0.0005)
    ap.add_argument('--notional', type=float, default=100.0, help='$ per trade')
    ap.add_argument('--interval', default='4h')
    ap.add_argument('--save-threshold', type=float, default=None,
                    help='also write the full trade log for this threshold')
    ap.add_argument('--trend-filter', action='store_true',
                    help='variant 1: only short when price has already pulled back '
                         'from its recent high (skip shorts into fresh new highs)')
    ap.add_argument('--trend-lookback', type=int, default=60,
                    help='bars for the trailing-high window of the trend filter')
    ap.add_argument('--trend-pullback', type=float, default=0.10,
                    help='required pullback from the trailing high to allow a short')
    ap.add_argument('--out-dir', default='data/model/short')
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof_path = os.path.join(base, args.oof) if not os.path.isabs(args.oof) else args.oof
    out_dir = os.path.join(base, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    oof = pd.read_parquet(oof_path)
    oof['Date'] = pd.to_datetime(oof['Date'])
    oof = oof.dropna(subset=[args.prob_col])
    thresholds = [float(t) for t in args.thresholds.split(',')]

    print("=" * 78)
    print(f"  SHORT ENTRY BACKTEST | prob={args.prob_col} | "
          f"TP price x{1/(1+args.tp):.3f} (+{args.tp*100:.0f}%) / "
          f"SL price x{1/(1-args.sl):.3f} (-{args.sl*100:.0f}%) / {args.horizon_days}d")
    print(f"  P&L: reciprocal short (entry/exit - 1) | OOS bars scored: {len(oof)} "
          f"across {oof['symbol'].nunique()} coins")
    print(f"  Costs: fee {args.fee*100:.2f}%/side + slip {args.slippage*100:.3f}%/side "
          f"(round-trip {2*(args.fee+args.slippage)*100:.2f}%)")
    print("=" * 78)

    suffix = f'_{args.interval}_full.csv'
    csvs = {os.path.basename(c).replace(suffix, '').upper(): c
            for c in glob.glob(os.path.join(base, 'data', '*', f'*{suffix}'))}
    med = (pd.read_csv(next(iter(csvs.values())), usecols=['Date'])['Date']
           .pipe(pd.to_datetime).diff().median())
    bpd = max(1, round(pd.Timedelta(days=1) / med))
    horizon_bars = args.horizon_days * bpd

    if args.trend_filter:
        print(f"  TREND FILTER ON: short only if price <= {(1-args.trend_pullback)*100:.0f}% "
              f"of its trailing {args.trend_lookback}-bar high "
              f"(pullback >= {args.trend_pullback*100:.0f}%)")

    # Pre-load each coin's OHLCV once (reused across all thresholds).
    coin_df, coin_trend = {}, {}
    for sym in oof['symbol'].unique():
        csv = csvs.get(sym)
        if csv is None:
            continue
        d = pd.read_csv(csv)
        d['Date'] = pd.to_datetime(d['Date'])
        d = d.sort_values('Date').reset_index(drop=True)
        coin_df[sym] = d
        coin_trend[sym] = (compute_trend_ok(d, args.trend_lookback, args.trend_pullback)
                           if args.trend_filter else None)

    sweep_rows = []
    for thr in thresholds:
        sig = oof[oof[args.prob_col] >= thr]
        all_trades = []
        for sym, grp in sig.groupby('symbol'):
            if sym not in coin_df:
                continue
            tr = simulate_coin_short(coin_df[sym], set(grp['Date']), args.tp, args.sl,
                                     horizon_bars, args.fee, args.slippage, args.notional,
                                     trend_ok=coin_trend.get(sym))
            for t in tr:
                t['symbol'] = sym
            all_trades.extend(tr)
        if not all_trades:
            print(f"  thr {thr:.2f}: no trades")
            continue
        td = pd.DataFrame(all_trades).sort_values('entry_date').reset_index(drop=True)
        m = summarize(td, args.notional, bpd)
        m['threshold'] = thr
        sweep_rows.append(m)
        if args.save_threshold is not None and abs(thr - args.save_threshold) < 1e-9:
            out_csv = os.path.join(out_dir, 'entry_backtest_trades.csv')
            td.to_csv(out_csv, index=False, encoding='utf-8-sig')
            print(f"  thr {thr:.2f}: trade log -> {out_csv}")

    if not sweep_rows:
        print("No trades at any threshold.")
        return

    sw = pd.DataFrame(sweep_rows)[['threshold', 'trades', 'tp', 'sl', 'time',
                                   'win_rate', 'avg_win', 'avg_loss', 'expectancy',
                                   'profit_factor', 'avg_days', 'total_pnl',
                                   'ret_on_deployed']]
    sweep_csv = os.path.join(out_dir, 'threshold_sweep.csv')
    sw.to_csv(sweep_csv, index=False, encoding='utf-8-sig')

    print("\n  THRESHOLD SWEEP (out-of-sample, $%.0f/trade):" % args.notional)
    print(f"  {'thr':>5} {'trades':>7} {'TP/SL/TIME':>14} {'win%':>6} {'avgW%':>7} "
          f"{'avgL%':>7} {'exp%':>7} {'PF':>5} {'days':>5} {'totalP&L':>11} {'ret%':>7}")
    for _, r in sw.iterrows():
        oc = f"{int(r['tp'])}/{int(r['sl'])}/{int(r['time'])}"
        print(f"  {r['threshold']:>5.2f} {int(r['trades']):>7} {oc:>14} "
              f"{r['win_rate']:>6.1f} {r['avg_win']:>+7.1f} {r['avg_loss']:>+7.1f} "
              f"{r['expectancy']:>+7.2f} {r['profit_factor']:>5.2f} {r['avg_days']:>5.0f} "
              f"${r['total_pnl']:>+10,.0f} {r['ret_on_deployed']:>+7.1f}")

    best_pnl = sw.loc[sw['total_pnl'].idxmax()]
    best_exp = sw.loc[sw['expectancy'].idxmax()]
    print(f"\n  Max total P&L : thr {best_pnl['threshold']:.2f} -> ${best_pnl['total_pnl']:+,.0f} "
          f"({int(best_pnl['trades'])} trades, exp {best_pnl['expectancy']:+.2f}%/tr, PF {best_pnl['profit_factor']:.2f})")
    print(f"  Max expectancy: thr {best_exp['threshold']:.2f} -> {best_exp['expectancy']:+.2f}%/tr "
          f"({int(best_exp['trades'])} trades, ${best_exp['total_pnl']:+,.0f}, PF {best_exp['profit_factor']:.2f})")
    print(f"\n  Sweep CSV: {sweep_csv}")
    print("=" * 78)


if __name__ == '__main__':
    main()
