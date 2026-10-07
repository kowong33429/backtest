import os
import glob
import numpy as np
import pandas as pd
import pandas_ta as ta
import optuna
import warnings
warnings.filterwarnings('ignore')

def load_btc_regime(base_path):
    # Load BTC to act as the overarching Crypto Market Regime filter
    # If BTC is dying, we shouldn't be buying altcoins!
    btc_path = os.path.join(base_path, 'data', 'binance', 'BTCUSDT_4h_full.csv')
    if not os.path.exists(btc_path): return None
    btc = pd.read_csv(btc_path)
    btc['Date'] = pd.to_datetime(btc['Date'])
    btc['BTC_SMA50'] = ta.sma(btc['Close'], length=50)
    btc['BTC_SMA200'] = ta.sma(btc['Close'], length=200)
    btc['BTC_Regime'] = np.where(btc['BTC_SMA50'] > btc['BTC_SMA200'], 1, -1)
    btc['BTC_Regime'] = btc['BTC_Regime'].ffill().fillna(1)
    return btc[['Date', 'BTC_Regime']]

def calculate_indicators(df):
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    # N-1 rule: forward-fill only (bfill would leak future warmup values backward).
    df['ATR'] = df['ATR'].ffill()
    df['CMF'] = df['CMF'].ffill()
    return df

def simulate_supreme(df, sig_dict, tp, sl, horizon_bars, fee, slip, atr_mult, cmf_thresh, btc_filter_on, dynamic_sizing):
    df = df.sort_values('Date').reset_index(drop=True)
    o, hi, lo, cl = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    atr = df['ATR'].values
    cmf = df['CMF'].values
    btc_regime = df['BTC_Regime'].values if 'BTC_Regime' in df.columns else np.ones(len(df))
    dt = df['Date'].values
    
    date_to_pos = {d: i for i, d in enumerate(dt)}
    # Only keep signals that exist in the price dataframe
    valid_sigs = {date_to_pos[d]: prob for d, prob in sig_dict.items() if d in date_to_pos}
    sig_pos = sorted(valid_sigs.keys())

    total_net = 0.0
    
    cooldown_until = -1
    for s in sig_pos:
        if s <= cooldown_until: continue
        
        # SUPREME FILTER 1: BTC Regime (Don't buy alts in a bear market)
        # N-1 rule: BTC regime read from the signal candle (s), not the execution candle.
        if btc_filter_on and btc_regime[s] == -1:
            continue

        entry_i = s + 1
        if entry_i >= len(df): continue
        entry_px = o[entry_i]
        if entry_px <= 0 or not np.isfinite(entry_px): continue
        if not np.isfinite(atr[entry_i]): continue  # skip while ATR is still warming up
        
        end = min(entry_i + horizon_bars, len(df) - 1)
        fixed_dn = entry_px * (1 - sl)
        highest_seen = hi[entry_i]
        exit_i, exit_px = end, cl[end]
        
        for j in range(entry_i, end + 1):
            highest_seen = max(highest_seen, hi[j])
            
            # EARLY EXIT via Money Flow
            if cmf[j] < cmf_thresh:
                exit_i, exit_px = j, cl[j]
                break
                
            current_sl = max(fixed_dn, highest_seen - (atr[j] * atr_mult))
            if lo[j] <= current_sl:
                exit_i, exit_px = j, current_sl
                break
                    
        gross = exit_px / entry_px - 1.0
        net = gross - 2 * (fee + slip)
        
        # SUPREME SIZING: strictly 100 as requested
        notional = 100.0
            
        total_net += (net * notional)
        cooldown_until = exit_i
        
    return total_net

def evaluate_params(params, oof, csvs, btc_df):
    """Run the full backtest for one parameter set over the given OOF slice."""
    # Entry probability threshold is itself a tuned parameter, so filter here.
    sig = oof[oof['oof_prob_xgb'] >= params['min_prob']]
    sl = params['sl']
    atr_mult = params['atr_mult']
    horizon_days = params['horizon_days']
    cmf_thresh = params['cmf_thresh']
    btc_filter_on = params['btc_filter_on']

    total_pnl = 0.0
    for sym, grp in sig.groupby('symbol'):
        csv = csvs.get(sym)
        if csv is None: continue

        df = pd.read_csv(csv)
        df['Date'] = pd.to_datetime(df['Date'])

        if btc_df is not None:
            df = pd.merge(df, btc_df, on='Date', how='left')
            df['BTC_Regime'] = df['BTC_Regime'].ffill().fillna(1)

        df = calculate_indicators(df)

        # Convert to dictionary of {Date: prob}
        sig_dict = dict(zip(grp['Date'], grp['oof_prob_xgb']))
        horizon_bars = horizon_days * 6

        total_pnl += simulate_supreme(df, sig_dict, 0, sl, horizon_bars, 0.001, 0.0005,
                                      atr_mult, cmf_thresh, btc_filter_on, False)

    return total_pnl

def objective(trial, oof, csvs, btc_df):
    params = {
        # Optimize Entry Probability Threshold (Filter out weak signals)
        'min_prob': trial.suggest_float('min_prob', 0.70, 0.82, step=0.02),
        'sl': trial.suggest_float('sl', 0.4, 0.8, step=0.1),
        'atr_mult': trial.suggest_float('atr_mult', 12.0, 20.0, step=1.0),
        'horizon_days': trial.suggest_int('horizon_days', 60, 120, step=10),
        'cmf_thresh': trial.suggest_float('cmf_thresh', -0.5, -0.1, step=0.1),
        # "Do whatever it takes" flags
        'btc_filter_on': trial.suggest_categorical('btc_filter_on', [True, False]),
    }
    return evaluate_params(params, oof, csvs, btc_df)


def time_split(oof, frac=0.7):
    """Chronological hold-out: earliest `frac` of dates train, the rest validate."""
    dates_sorted = np.sort(oof['Date'].unique())
    cutoff = pd.Timestamp(dates_sorted[int(len(dates_sorted) * frac)])
    return oof[oof['Date'] <= cutoff], oof[oof['Date'] > cutoff], cutoff

def main():
    print("="*60)
    print(" AI OPTIMIZER: PROJECT SUPREME (TARGET: $50,000+)")
    print("="*60)
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof = pd.read_parquet(os.path.join(base, 'data/model/v2/m1_100pct_60d/entry_oof.parquet'))
    oof['Date'] = pd.to_datetime(oof['Date'])

    # Chronological train/validation split: tune on the past, report on the unseen future.
    oof_train, oof_valid, cutoff = time_split(oof, frac=0.7)
    print(f"Train rows: {len(oof_train)} (<= {cutoff.date()}) | "
          f"Validation rows: {len(oof_valid)} (> {cutoff.date()})")

    btc_df = load_btc_regime(base)

    suffix = '_4h_full.csv'
    csvs = {os.path.basename(c).replace(suffix, '').upper(): c for c in glob.glob(os.path.join(base, 'data', '*', f'*{suffix}'))}

    study = optuna.create_study(direction='maximize')

    print("AI is authorized to 'DO WHATEVER IT TAKES' (train slice)...")
    study.optimize(lambda trial: objective(trial, oof_train, csvs, btc_df), n_trials=40, n_jobs=1)

    val_pnl = evaluate_params(study.best_params, oof_valid, csvs, btc_df)

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
