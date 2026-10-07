import os
import glob
import argparse
import numpy as np
import pandas as pd

def compute_physics_features(df):
    # Standard pseudo-ATR for ecosystem volatility
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean().fillna(df['Close'] * 0.05)
    
    # Ecosystem Seasons (Bull vs Bear Market)
    df['summer'] = df['Close'].rolling(50).mean() > df['Close'].rolling(200).mean()
    
    return df

def simulate_coin_physics(df, sig_dates, tp, sl, horizon_bars, fee, slip, notional):
    """Walk one coin's bars; open at next Open on a signal, exit based on Anti-Fragile Biological model."""
    df = df.sort_values('Date').reset_index(drop=True)
    df = compute_physics_features(df)
    
    o, hi, lo, cl = (df['Open'].values, df['High'].values,
                     df['Low'].values, df['Close'].values)
    dt = df['Date'].values
    atr = df['ATR'].values
    summer = df['summer'].values
    
    date_to_pos = {d: i for i, d in enumerate(dt)}
    sig_pos = sorted(date_to_pos[d] for d in sig_dates if d in date_to_pos)

    trades = []
    cooldown_until = -1
    for s in sig_pos:
        if s <= cooldown_until:
            continue
        entry_i = s + 1
        if entry_i >= len(df):
            continue
        entry_px = o[entry_i]
        if entry_px <= 0 or not np.isfinite(entry_px):
            continue
            
        up, dn = entry_px * (1 + tp), entry_px * (1 - sl)
        end = min(entry_i + horizon_bars, len(df) - 1)
        
        exit_i, exit_px, outcome = end, cl[end], 'TIME'
        highest_seen = hi[entry_i]
        is_summer = summer[entry_i]
        
        for j in range(entry_i, end + 1):
            if not is_summer:
                # Winter (Downtrend): Harsh conditions, fixed survival bounds
                hit_dn, hit_up = lo[j] <= dn, hi[j] >= up
                if hit_dn and hit_up:
                    exit_i, exit_px, outcome = j, dn, 'SL'
                    break
                if hit_dn:
                    exit_i, exit_px, outcome = j, dn, 'SL'
                    break
                if hit_up:
                    exit_i, exit_px, outcome = j, up, 'TP'
                    break
            else:
                # Summer (Uptrend): Organisms can thrive and become Anti-Fragile
                highest_seen = max(highest_seen, hi[j])
                
                # Anti-Fragile Biology: The larger the organism (profit), the more resistant it is to being killed.
                profit_ratio = highest_seen / entry_px
                
                # Base resilience is 6.0 ATR
                resilience = 6.0
                
                if profit_ratio > 1.2:
                    # Grew by 20%, becomes extremely hard to kill
                    resilience = 12.0
                if profit_ratio > 1.5:
                    # Grew by 50%, apex predator
                    resilience = 25.0
                if profit_ratio > 2.0:
                    # Grew by 100%, Godzilla
                    resilience = 50.0
                    
                # Evaluate fresh every bar! If ATR expands (volatility spikes), the stop dynamically widens
                # just like an organism bracing for a sudden storm.
                current_atr_sl = highest_seen - (atr[j] * resilience)
                current_sl = max(dn, current_atr_sl)
                
                if lo[j] <= current_sl:
                    exit_i, exit_px, outcome = j, current_sl, 'ANTI_FRAGILE_DEATH'
                    break

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
        cooldown_until = exit_i
    return trades

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default='data/model/v2/m1_100pct_60d/entry_oof.parquet')
    ap.add_argument('--prob-col', default='oof_prob_xgb')
    ap.add_argument('--threshold', type=float, default=0.70)
    ap.add_argument('--tp', type=float, default=1.00)
    ap.add_argument('--sl', type=float, default=0.40)
    ap.add_argument('--horizon-days', type=int, default=30)
    ap.add_argument('--fee', type=float, default=0.001)
    ap.add_argument('--slippage', type=float, default=0.0005)
    ap.add_argument('--notional', type=float, default=100.0)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof_path = os.path.join(base, args.oof)
    
    oof = pd.read_parquet(oof_path)
    oof['Date'] = pd.to_datetime(oof['Date'])
    oof = oof.dropna(subset=[args.prob_col])
    sig = oof[oof[args.prob_col] >= args.threshold]
    
    print("=" * 72)
    print("  SCIENTIST ENTRY BACKTEST (Physics Model)")
    print("=" * 72)

    suffix = '_4h_full.csv'
    csvs = {os.path.basename(c).replace(suffix, '').upper(): c
            for c in glob.glob(os.path.join(base, 'data', '*', f'*{suffix}'))}

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
        tr = simulate_coin_physics(df, set(grp['Date']), args.tp, args.sl, horizon_bars,
                                   args.fee, args.slippage, args.notional)
        for t in tr:
            t['symbol'] = sym
        all_trades.extend(tr)

    td = pd.DataFrame(all_trades)
    n = len(td)
    wins = td[td['net_ret'] > 0]
    total_pnl = td['pnl_usd'].sum()
    print(f"Total PNL: ${total_pnl:,.0f} on {n} trades")
    print(td['outcome'].value_counts())

if __name__ == '__main__':
    main()
