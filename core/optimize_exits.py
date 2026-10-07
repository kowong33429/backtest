import os
import glob
import numpy as np
import pandas as pd
import pandas_ta as ta
import optuna

def calculate_indicators(df, sma_fast=50, sma_slow=200):
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['SMA_Fast'] = ta.sma(df['Close'], length=sma_fast)
    df['SMA_Slow'] = ta.sma(df['Close'], length=sma_slow)
    # N-1 rule: forward-fill only (bfill would leak future warmup values backward).
    df['ATR'] = df['ATR'].ffill()
    df['SMA_Fast'] = df['SMA_Fast'].ffill()
    df['SMA_Slow'] = df['SMA_Slow'].ffill()
    return df

def simulate_coin_fast(df, sig_dates, mode, tp, sl, horizon_bars, fee, slip, atr_mult):
    df = df.sort_values('Date').reset_index(drop=True)
    o, hi, lo, cl = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    atr = df['ATR'].values
    sma_f = df['SMA_Fast'].values
    sma_s = df['SMA_Slow'].values
    dt = df['Date'].values
    date_to_pos = {d: i for i, d in enumerate(dt)}
    sig_pos = sorted(date_to_pos[d] for d in sig_dates if d in date_to_pos)

    total_net = 0.0
    cooldown_until = -1
    for s in sig_pos:
        if s <= cooldown_until: continue
        entry_i = s + 1
        if entry_i >= len(df): continue
        entry_px = o[entry_i]
        if entry_px <= 0 or not np.isfinite(entry_px): continue
        
        end = min(entry_i + horizon_bars, len(df) - 1)
        fixed_up, fixed_dn = entry_px * (1 + tp), entry_px * (1 - sl)
        highest_seen = hi[entry_i]
        exit_i, exit_px = end, cl[end]
        
        # N-1 rule: regime judged on the signal candle (entry_i - 1), not the
        # execution candle, and skipped while the SMAs are still warming up (NaN).
        dec = entry_i - 1
        is_uptrend = (np.isfinite(sma_f[dec]) and np.isfinite(sma_s[dec])
                      and sma_f[dec] > sma_s[dec])

        if mode == 'regime' and is_uptrend:
            for j in range(entry_i, end + 1):
                highest_seen = max(highest_seen, hi[j])
                current_sl = max(fixed_dn, highest_seen - (atr[j] * atr_mult))
                if lo[j] <= current_sl:
                    exit_i, exit_px = j, current_sl
                    break
        else:
            for j in range(entry_i, end + 1):
                hit_dn, hit_up = lo[j] <= fixed_dn, hi[j] >= fixed_up
                if hit_dn:
                    exit_i, exit_px = j, fixed_dn
                    break
                if hit_up:
                    exit_i, exit_px = j, fixed_up
                    break
                    
        gross = exit_px / entry_px - 1.0
        net = gross - 2 * (fee + slip)
        total_net += net
        cooldown_until = exit_i
        
    return total_net

def evaluate_params(params, sig, csvs):
    """Run the full backtest for one parameter set over the given signal slice."""
    mode = params['mode']
    tp, sl, atr_mult = params['tp'], params['sl'], params['atr_mult']
    sma_fast, sma_slow = params['sma_fast'], params['sma_slow']
    horizon_days = params['horizon_days']

    total_pnl = 0.0
    for sym, grp in sig.groupby('symbol'):
        csv = csvs.get(sym)
        if csv is None: continue

        df = pd.read_csv(csv)
        df['Date'] = pd.to_datetime(df['Date'])
        df = calculate_indicators(df, sma_fast=sma_fast, sma_slow=sma_slow)

        dates_set = set(grp['Date'])
        horizon_bars = horizon_days * 6 # assuming 4h

        # To simulate always trail, we just trick the regime mode by passing a condition that's always true
        sim_mode = 'regime' if mode != 'fixed' else 'fixed'
        if mode == 'always_trail':
            df['SMA_Fast'] = 2
            df['SMA_Slow'] = 1

        net_ret = simulate_coin_fast(df, dates_set, sim_mode, tp, sl, horizon_bars, 0.001, 0.0005, atr_mult)
        total_pnl += (net_ret * 100.0) # $100 notional

    return total_pnl

def objective(trial, sig, csvs):
    # AI (Optuna) will guess these parameters to find the best combination!
    params = {
        'mode': trial.suggest_categorical('mode', ['fixed', 'regime', 'always_trail']),
        'tp': trial.suggest_float('tp', 0.5, 5.0, step=0.1),
        'sl': trial.suggest_float('sl', 0.1, 0.6, step=0.05),
        'atr_mult': trial.suggest_float('atr_mult', 2.0, 15.0, step=0.5),
        'sma_fast': trial.suggest_categorical('sma_fast', [20, 50]),
        'sma_slow': trial.suggest_categorical('sma_slow', [100, 200]),
        'horizon_days': trial.suggest_int('horizon_days', 10, 90, step=10),
    }
    return evaluate_params(params, sig, csvs)


def time_split(sig, frac=0.7):
    """Chronological hold-out: earliest `frac` of signal dates train, the rest validate."""
    dates_sorted = np.sort(sig['Date'].unique())
    cutoff = pd.Timestamp(dates_sorted[int(len(dates_sorted) * frac)])
    return sig[sig['Date'] <= cutoff], sig[sig['Date'] > cutoff], cutoff

def main():
    print("="*60)
    print(" AI OPTIMIZER STARTED (Optuna Exit Strategy Tuning)")
    print("="*60)
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof = pd.read_parquet(os.path.join(base, 'data/model/v2/m1_100pct_60d/entry_oof.parquet'))
    oof['Date'] = pd.to_datetime(oof['Date'])
    sig = oof[oof['oof_prob_xgb'] >= 0.70]

    # Chronological train/validation split: tune on the past, report on the unseen future.
    sig_train, sig_valid, cutoff = time_split(sig, frac=0.7)
    print(f"Train: {len(sig_train)} signals (<= {cutoff.date()}) | "
          f"Validation: {len(sig_valid)} signals (> {cutoff.date()})")

    suffix = '_4h_full.csv'
    csvs = {os.path.basename(c).replace(suffix, '').upper(): c
            for c in glob.glob(os.path.join(base, 'data', '*', f'*{suffix}'))}

    study = optuna.create_study(direction='maximize')

    # Run the AI for 50 trials (experiments) -- ON THE TRAINING SLICE ONLY.
    print("AI is now trying different parameters to maximize PnL (train slice)...")
    study.optimize(lambda trial: objective(trial, sig_train, csvs), n_trials=50, n_jobs=1)

    # Re-run the best parameters on the untouched validation slice.
    val_pnl = evaluate_params(study.best_params, sig_valid, csvs)

    print("\n" + "="*60)
    print(" AI OPTIMIZATION FINISHED")
    print("="*60)
    print(f"IN-SAMPLE (train) best PnL:  ${study.best_value:,.2f}")
    print(f"OUT-OF-SAMPLE (validation):  ${val_pnl:,.2f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == '__main__':
    main()
