"""
entry_backtest.py — Honest backtest of the entry model's signal

Trades the OUT-OF-SAMPLE probabilities from train_entry_model.py (entry_oof.parquet
— each bar scored by a model that never trained on it) so there is NO in-sample
leakage. Execution and exit follow AGENTS.md Rule #1 and the label's own target.

RULES
-----
  Signal   : oof_prob_xgb >= threshold (default 0.70).
  Entry    : decision uses features up to bar n-1; EXECUTE at the Open of bar n+1
             (the next bar after the signal) — the only unshifted price we trade on.
  Exit     : whichever comes first (matches the label definition for the -40% model):
               * Take Profit  : +100%
               * Stop Loss    : -40%
               * Time stop     : 30 days -> exit at that bar's Close
             Same-bar TP&SL ambiguity resolves to SL (conservative).
  Position : one open trade per coin at a time (new signals ignored while in a
             trade); different coins can hold positions concurrently.
  Costs    : fee + slippage charged on BOTH entry and exit (AGENTS.md Rule #9).

Sizing: fixed notional per trade, so each trade's $ P&L = notional * net return.
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd


def simulate_coin(df, sig_dates, tp, sl, horizon_bars, fee, slip, notional):
    """Walk one coin's bars; open at next Open on a signal, exit on TP/SL/time."""
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
        entry_i = s + 1  # execute at the NEXT bar's Open
        if entry_i >= len(df):
            continue
        entry_px = o[entry_i]
        if entry_px <= 0 or not np.isfinite(entry_px):
            continue
        up, dn = entry_px * (1 + tp), entry_px * (1 - sl)
        end = min(entry_i + horizon_bars, len(df) - 1)

        exit_i, exit_px, outcome = end, cl[end], 'TIME'
        for j in range(entry_i, end + 1):
            hit_dn, hit_up = lo[j] <= dn, hi[j] >= up
            if hit_dn and hit_up:
                exit_i, exit_px, outcome = j, dn, 'SL'       # conservative
                break
            if hit_dn:
                exit_i, exit_px, outcome = j, dn, 'SL'
                break
            if hit_up:
                exit_i, exit_px, outcome = j, up, 'TP'
                break

        # Net return after costs on both sides.
        gross = exit_px / entry_px - 1.0
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
        cooldown_until = exit_i  # no re-entry until this trade closes
    return trades


def main():
    ap = argparse.ArgumentParser(description="Backtest the entry model's OOS signal")
    ap.add_argument('--oof', default='data/model/entry_oof.parquet')
    ap.add_argument('--prob-col', default='oof_prob_xgb')
    ap.add_argument('--threshold', type=float, default=0.70)
    ap.add_argument('--tp', type=float, default=1.00)
    ap.add_argument('--sl', type=float, default=0.40)
    ap.add_argument('--horizon-days', type=int, default=30)
    ap.add_argument('--fee', type=float, default=0.001, help='fee per side (0.001=0.1%%)')
    ap.add_argument('--slippage', type=float, default=0.0005)
    ap.add_argument('--notional', type=float, default=100.0, help='$ per trade')
    ap.add_argument('--out-dir', default='data/model')
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof_path = os.path.join(base, args.oof)
    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    oof = pd.read_parquet(oof_path)
    oof['Date'] = pd.to_datetime(oof['Date'])
    oof = oof.dropna(subset=[args.prob_col])
    sig = oof[oof[args.prob_col] >= args.threshold]
    print("=" * 72)
    print(f"  ENTRY BACKTEST | prob={args.prob_col} >= {args.threshold} | "
          f"TP +{args.tp*100:.0f}% / SL -{args.sl*100:.0f}% / {args.horizon_days}d")
    print(f"  OOS bars scored: {len(oof)} | raw signal bars: {len(sig)} "
          f"across {sig['symbol'].nunique()} coins")
    print(f"  Costs: fee {args.fee*100:.2f}%/side + slip {args.slippage*100:.3f}%/side "
          f"(round-trip {2*(args.fee+args.slippage)*100:.2f}%)")
    print("=" * 72)

    csvs = {os.path.basename(c).replace('_4h_full.csv', '').upper(): c
            for c in glob.glob(os.path.join(base, 'data', '*', '*_4h_full.csv'))}

    med = (pd.read_csv(next(iter(csvs.values())), usecols=['Date'])['Date']
           .pipe(pd.to_datetime).diff().median())
    bpd = max(1, round(pd.Timedelta(days=1) / med))
    horizon_bars = args.horizon_days * bpd

    all_trades = []
    for sym, grp in sig.groupby('symbol'):
        csv = csvs.get(sym)
        if csv is None:
            continue
        df = pd.read_csv(csv)
        df['Date'] = pd.to_datetime(df['Date'])
        tr = simulate_coin(df, set(grp['Date']), args.tp, args.sl, horizon_bars,
                           args.fee, args.slippage, args.notional)
        for t in tr:
            t['symbol'] = sym
        all_trades.extend(tr)

    if not all_trades:
        print("No trades generated.")
        return

    td = pd.DataFrame(all_trades).sort_values('entry_date').reset_index(drop=True)
    out_csv = os.path.join(out_dir, 'entry_backtest_trades.csv')
    td.to_csv(out_csv, index=False, encoding='utf-8-sig')

    # ---- Summary ----
    n = len(td)
    wins = td[td['net_ret'] > 0]
    losses = td[td['net_ret'] <= 0]
    win_rate = len(wins) / n
    gross_win = wins['net_ret'].sum()
    gross_loss = -losses['net_ret'].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    expectancy = td['net_ret'].mean()
    total_pnl = td['pnl_usd'].sum()
    oc = td['outcome'].value_counts().to_dict()

    print(f"\n  TRADES: {n}  (TP {oc.get('TP',0)} / SL {oc.get('SL',0)} / TIME {oc.get('TIME',0)})")
    print(f"  Win rate (ชนะ %):      {win_rate*100:.1f}%")
    print(f"  Avg win:               +{wins['net_ret'].mean()*100:.1f}%" if len(wins) else "  Avg win: -")
    print(f"  Avg loss:              {losses['net_ret'].mean()*100:.1f}%" if len(losses) else "  Avg loss: -")
    print(f"  Expectancy / trade:    {expectancy*100:+.2f}%  (net, after costs)")
    print(f"  Profit factor:         {pf:.2f}   (>1 = ได้กำไรรวม)")
    print(f"  Avg bars held:         {td['bars_held'].mean():.0f}  (~{td['bars_held'].mean()/bpd:.0f} days)")
    print(f"  Total P&L (@${args.notional:.0f}/trade): ${total_pnl:,.0f}  "
          f"on {n} trades = {total_pnl/(n*args.notional)*100:+.1f}% of deployed")
    print(f"\n  Trades CSV: {out_csv}")

    # Per-year and per-coin breakdown
    td['year'] = td['entry_date'].dt.year
    yr = td.groupby('year').agg(trades=('net_ret', 'size'),
                                win_rate=('net_ret', lambda s: (s > 0).mean()*100),
                                expectancy=('net_ret', lambda s: s.mean()*100),
                                pnl=('pnl_usd', 'sum')).round(2)
    print("\n  By year:")
    print(yr.to_string())
    print("\n  Top 10 coins by P&L:")
    coin = td.groupby('symbol').agg(trades=('net_ret', 'size'),
                                    win_rate=('net_ret', lambda s: (s > 0).mean()*100),
                                    pnl=('pnl_usd', 'sum')).round(1).sort_values('pnl', ascending=False)
    print(coin.head(10).to_string())
    print(f"\n  Worst 5 coins by P&L:")
    print(coin.tail(5).to_string())
    print("=" * 72)


if __name__ == '__main__':
    main()
