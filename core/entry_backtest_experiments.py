import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import pandas_ta as ta

def calculate_indicators(df):
    """Calculate necessary indicators for dynamic exits."""
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    # N-1 rule: forward-fill only (bfill would leak future warmup values backward).
    df['ATR_14'] = df['ATR_14'].ffill()
    df['SMA_50'] = df['SMA_50'].ffill()
    df['SMA_200'] = df['SMA_200'].ffill()
    return df

def simulate_coin_mode(df, sig_dates, mode, tp, sl, horizon_bars, fee, slip, notional, atr_mult=6.0):
    """Walk one coin's bars; simulate different exit modes."""
    df = df.sort_values('Date').reset_index(drop=True)
    o, hi, lo, cl = (df['Open'].values, df['High'].values,
                     df['Low'].values, df['Close'].values)
    atr = df['ATR_14'].values
    sma50 = df['SMA_50'].values
    sma200 = df['SMA_200'].values
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
        
        end = min(entry_i + horizon_bars, len(df) - 1)
        
        # Fixed logic
        fixed_up = entry_px * (1 + tp)
        fixed_dn = entry_px * (1 - sl)
        
        # State tracking
        highest_seen = hi[entry_i]
        exit_i, exit_px, outcome = end, cl[end], 'TIME'
        gross = 0.0
        
        if mode == 'Fixed':
            for j in range(entry_i, end + 1):
                hit_dn, hit_up = lo[j] <= fixed_dn, hi[j] >= fixed_up
                if hit_dn and hit_up:
                    exit_i, exit_px, outcome = j, fixed_dn, 'SL'
                    break
                if hit_dn:
                    exit_i, exit_px, outcome = j, fixed_dn, 'SL'
                    break
                if hit_up:
                    exit_i, exit_px, outcome = j, fixed_up, 'TP'
                    break
            gross = exit_px / entry_px - 1.0

        elif mode == 'Trailing':
            # No TP, just trailing stop
            # Trailing SL = max_high - ATR * mult
            for j in range(entry_i, end + 1):
                highest_seen = max(highest_seen, hi[j])
                # Stop loss is either initial -40% or the ATR trailing stop (whichever is higher)
                current_atr_sl = highest_seen - (atr[j] * atr_mult)
                current_sl = max(fixed_dn, current_atr_sl)
                
                if lo[j] <= current_sl:
                    exit_i, exit_px, outcome = j, current_sl, 'TRAIL_SL'
                    break
            gross = exit_px / entry_px - 1.0

        elif mode == 'ScaleOut':
            # Sell 50% at Fixed TP, trail the rest 50%.
            half_sold = False
            half_exit_px = 0.0
            for j in range(entry_i, end + 1):
                highest_seen = max(highest_seen, hi[j])
                current_atr_sl = highest_seen - (atr[j] * atr_mult)
                current_sl = max(fixed_dn, current_atr_sl)
                
                # Check for first half TP
                if not half_sold and hi[j] >= fixed_up:
                    half_sold = True
                    half_exit_px = fixed_up
                    
                # Check SL
                if lo[j] <= current_sl:
                    exit_i, exit_px, outcome = j, current_sl, 'TRAIL_SL'
                    break
                    
            if half_sold:
                outcome = 'SCALE_OUT'
                gross = 0.5 * (half_exit_px / entry_px - 1.0) + 0.5 * (exit_px / entry_px - 1.0)
            else:
                gross = exit_px / entry_px - 1.0

        elif mode == 'Regime':
            # N-1 rule: judge the regime on candle n-1 (the signal candle, entry_i - 1),
            # not the execution candle whose close is still in the future at entry.
            dec = entry_i - 1
            is_uptrend = (np.isfinite(sma50[dec]) and np.isfinite(sma200[dec])
                          and sma50[dec] > sma200[dec])
            if is_uptrend:
                # Same as Trailing
                for j in range(entry_i, end + 1):
                    highest_seen = max(highest_seen, hi[j])
                    current_atr_sl = highest_seen - (atr[j] * atr_mult)
                    current_sl = max(fixed_dn, current_atr_sl)
                    if lo[j] <= current_sl:
                        exit_i, exit_px, outcome = j, current_sl, 'TRAIL_SL (Uptrend)'
                        break
                gross = exit_px / entry_px - 1.0
            else:
                # Same as Fixed
                for j in range(entry_i, end + 1):
                    hit_dn, hit_up = lo[j] <= fixed_dn, hi[j] >= fixed_up
                    if hit_dn and hit_up:
                        exit_i, exit_px, outcome = j, fixed_dn, 'SL (Downtrend)'
                        break
                    if hit_dn:
                        exit_i, exit_px, outcome = j, fixed_dn, 'SL (Downtrend)'
                        break
                    if hit_up:
                        exit_i, exit_px, outcome = j, fixed_up, 'TP (Downtrend)'
                        break
                gross = exit_px / entry_px - 1.0


        # Net return after costs on both sides (ScaleOut technically pays 3 times fees, but we simplify to 2 for now to compare)
        cost = 2 * (fee + slip)
        if mode == 'ScaleOut' and outcome == 'SCALE_OUT':
            cost = 3 * (fee + slip) # Entry 100%, Exit 50%, Exit 50%
            
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
        cooldown_until = exit_i
    return trades

def run_experiment():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default='data/model/v2/m1_100pct_60d/entry_oof.parquet')
    ap.add_argument('--threshold', type=float, default=0.70)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof_path = os.path.join(base, args.oof)
    
    print("Loading predictions...")
    oof = pd.read_parquet(oof_path)
    oof['Date'] = pd.to_datetime(oof['Date'])
    sig = oof[oof['oof_prob_xgb'] >= args.threshold]
    
    suffix = '_4h_full.csv'
    csvs = {os.path.basename(c).replace(suffix, '').upper(): c
            for c in glob.glob(os.path.join(base, 'data', '*', f'*{suffix}'))}

    modes = ['Fixed', 'Trailing', 'ScaleOut', 'Regime']
    results = {m: [] for m in modes}
    
    print(f"Running experiments for {len(sig)} signals across {sig['symbol'].nunique()} coins...")
    
    # Process coin by coin
    for sym, grp in sig.groupby('symbol'):
        csv = csvs.get(sym)
        if csv is None: continue
        
        df = pd.read_csv(csv)
        df['Date'] = pd.to_datetime(df['Date'])
        df = calculate_indicators(df)
        
        dates_set = set(grp['Date'])
        # 30 days * 6 bars = 180 bars
        
        for mode in modes:
            tr = simulate_coin_mode(df, dates_set, mode=mode, tp=1.0, sl=0.40, horizon_bars=180, 
                                    fee=0.001, slip=0.0005, notional=100.0, atr_mult=6.0)
            for t in tr: t['symbol'] = sym
            results[mode].extend(tr)
            
    print("\n" + "="*70)
    print("  COMPARISON REPORT: EXIT STRATEGIES")
    print("="*70)
    
    summary_data = []
    
    for mode in modes:
        td = pd.DataFrame(results[mode])
        if len(td) == 0: continue
        
        n = len(td)
        wins = td[td['net_ret'] > 0]
        losses = td[td['net_ret'] <= 0]
        win_rate = len(wins) / n * 100
        expectancy = td['net_ret'].mean() * 100
        total_pnl = td['pnl_usd'].sum()
        
        # ZEC specific
        zec_td = td[td['symbol'] == 'ZECUSDT']
        zec_pnl = zec_td['pnl_usd'].sum() if len(zec_td) > 0 else 0
        
        # Max win
        max_win = td['net_ret'].max() * 100
        
        summary_data.append({
            'Mode': mode,
            'Trades': n,
            'Win Rate %': round(win_rate, 1),
            'Expectancy %': round(expectancy, 2),
            'Total PnL $': round(total_pnl, 0),
            'Max Win %': round(max_win, 1),
            'ZEC PnL $': round(zec_pnl, 1)
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*70)

    # COIN-BY-COIN COMPARISON (Regime vs Fixed)
    fixed_td = pd.DataFrame(results['Fixed'])
    regime_td = pd.DataFrame(results['Regime'])
    
    fixed_coin = fixed_td.groupby('symbol').agg(
        Fixed_Trades=('net_ret', 'size'),
        Fixed_PnL=('pnl_usd', 'sum'),
        Fixed_MaxWin=('net_ret', 'max')
    )
    regime_coin = regime_td.groupby('symbol').agg(
        Regime_Trades=('net_ret', 'size'),
        Regime_PnL=('pnl_usd', 'sum'),
        Regime_MaxWin=('net_ret', 'max')
    )
    
    comp = fixed_coin.join(regime_coin, how='outer').fillna(0)
    comp['PnL_Diff'] = comp['Regime_PnL'] - comp['Fixed_PnL']
    
    # Scale from raw ratio to %
    comp['Fixed_MaxWin'] = comp['Fixed_MaxWin'] * 100
    comp['Regime_MaxWin'] = comp['Regime_MaxWin'] * 100
    
    comp = comp.sort_values('PnL_Diff', ascending=False)
    
    out_csv = os.path.join(base, 'data', 'model', 'v2', 'm1_100pct_60d', 'regime_vs_fixed_all_coins.csv')
    comp.to_csv(out_csv)
    
    print(f"\nDetailed coin-by-coin comparison saved to: {out_csv}")
    
    improved_coins = (comp['PnL_Diff'] > 0).sum()
    worse_coins = (comp['PnL_Diff'] < 0).sum()
    print(f"Coins improved by Regime: {improved_coins}")
    print(f"Coins worse with Regime:  {worse_coins}")
    
    print("\nTop 5 Coins IMPROVED by Regime Switching:")
    print(comp[['Fixed_PnL', 'Regime_PnL', 'PnL_Diff', 'Regime_MaxWin']].head(5).round(1))
    
    print("\nTop 5 Coins WORSE with Regime Switching:")
    print(comp[['Fixed_PnL', 'Regime_PnL', 'PnL_Diff', 'Regime_MaxWin']].tail(5).round(1))

if __name__ == '__main__':
    run_experiment()
