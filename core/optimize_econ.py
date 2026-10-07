import os
import glob
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import optuna
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 1. MACRO ECONOMICS DATA LOADER (Respecting AGENTS.md Rule #2)
# -----------------------------------------------------------------
def load_macro_data(start_date, end_date):
    print("Downloading Macroeconomic Data (DXY, SP500)...")
    # Real-time market prices don't strictly need publication lag like CPI, 
    # but we must ensure strictly forward-fill alignment.
    # DXY = US Dollar Index (Inversely correlated with crypto)
    # SP500 = Global Risk Appetite
    # yfinance returns multi-index columns, we need to extract correctly
    dxy_df = yf.download("DX-Y.NYB", start=start_date, end=end_date, progress=False)
    spy_df = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    
    # Extract the close prices
    dxy = dxy_df.xs('Close', level='Price', axis=1)['DX-Y.NYB'] if isinstance(dxy_df.columns, pd.MultiIndex) else dxy_df['Close']
    spy = spy_df.xs('Close', level='Price', axis=1)['^GSPC'] if isinstance(spy_df.columns, pd.MultiIndex) else spy_df['Close']
    
    macro = pd.DataFrame({'DXY': dxy, 'SP500': spy})
    macro.index = pd.to_datetime(macro.index)
    
    # Calculate Macro Regimes (Trends)
    # If DXY is rising, it's bad for crypto. If SP500 is rising, it's good for crypto.
    macro['DXY_SMA50'] = macro['DXY'].rolling(50).mean()
    macro['DXY_SMA200'] = macro['DXY'].rolling(200).mean()
    macro['SP500_SMA50'] = macro['SP500'].rolling(50).mean()
    macro['SP500_SMA200'] = macro['SP500'].rolling(200).mean()
    
    # Define Risk-On Score (-1 to 1)
    # +1: SP500 Uptrend & DXY Downtrend (Perfect for Crypto)
    # -1: SP500 Downtrend & DXY Uptrend (Crypto Winter)
    macro['Econ_Risk_On'] = np.where((macro['SP500_SMA50'] > macro['SP500_SMA200']) & 
                                     (macro['DXY_SMA50'] < macro['DXY_SMA200']), 1, 
                            np.where((macro['SP500_SMA50'] < macro['SP500_SMA200']) & 
                                     (macro['DXY_SMA50'] > macro['DXY_SMA200']), -1, 0))
    
    return macro.reset_index().rename(columns={'Date': 'Macro_Date'})

def calculate_indicators(df, sma_fast=50, sma_slow=200):
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['SMA_Fast'] = ta.sma(df['Close'], length=sma_fast)
    df['SMA_Slow'] = ta.sma(df['Close'], length=sma_slow)
    # N-1 rule: forward-fill only (bfill would leak future warmup values backward).
    df['ATR'] = df['ATR'].ffill()
    df['SMA_Fast'] = df['SMA_Fast'].ffill()
    df['SMA_Slow'] = df['SMA_Slow'].ffill()
    return df

def simulate_econ_fast(df, sig_dates, tp, sl, horizon_bars, fee, slip, atr_mult_riskon, atr_mult_riskoff):
    df = df.sort_values('Date').reset_index(drop=True)
    o, hi, lo, cl = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    atr = df['ATR'].values
    econ_regime = df['Econ_Risk_On'].values
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
        if not np.isfinite(atr[entry_i]): continue  # skip while ATR is still warming up

        end = min(entry_i + horizon_bars, len(df) - 1)
        fixed_up, fixed_dn = entry_px * (1 + tp), entry_px * (1 - sl)
        highest_seen = hi[entry_i]
        exit_i, exit_px = end, cl[end]

        # N-1 rule: the econ regime is read from the signal candle (entry_i - 1).
        current_econ = econ_regime[entry_i - 1]
        
        # QUANT + ECON LOGIC:
        # If Econ is Risk-On (+1), we let profits run infinitely using a wide ATR trail (e.g. 15x).
        # If Econ is Risk-Off (-1), we tighten the stop loss dramatically to protect capital.
        
        active_atr_mult = atr_mult_riskon if current_econ >= 0 else atr_mult_riskoff
        
        for j in range(entry_i, end + 1):
            highest_seen = max(highest_seen, hi[j])
            current_sl = max(fixed_dn, highest_seen - (atr[j] * active_atr_mult))
            
            if lo[j] <= current_sl:
                exit_i, exit_px = j, current_sl
                break
                    
        gross = exit_px / entry_px - 1.0
        
        # Position Sizing based on Econ (Kelly-inspired):
        # Double size ($200) in Risk-On, Halve size ($50) in Risk-Off
        notional = 100.0
        if current_econ == 1:
            notional = 200.0
        elif current_econ == -1:
            notional = 50.0
            
        net = (gross - 2 * (fee + slip)) * (notional / 100.0) # Scale net by size factor
        total_net += net
        cooldown_until = exit_i
        
    return total_net

def evaluate_params(params, sig, csvs, macro_df):
    """Run the full backtest for one parameter set over the given signal slice."""
    tp, sl = params['tp'], params['sl']
    atr_mult_riskon = params['atr_mult_riskon']
    atr_mult_riskoff = params['atr_mult_riskoff']
    horizon_days = params['horizon_days']

    total_pnl = 0.0
    for sym, grp in sig.groupby('symbol'):
        csv = csvs.get(sym)
        if csv is None: continue

        df = pd.read_csv(csv)
        df['Date'] = pd.to_datetime(df['Date'])

        # AGENTS.md Rule 2: Merge Macro safely with ffill
        # Align macro data to crypto bars
        df['Macro_Date'] = df['Date'].dt.normalize()
        df = pd.merge(df, macro_df[['Macro_Date', 'Econ_Risk_On']], on='Macro_Date', how='left')
        df['Econ_Risk_On'] = df['Econ_Risk_On'].ffill().fillna(0) # Forward fill strictly

        df = calculate_indicators(df)

        dates_set = set(grp['Date'])
        horizon_bars = horizon_days * 6

        net_ret = simulate_econ_fast(df, dates_set, tp, sl, horizon_bars, 0.001, 0.0005, atr_mult_riskon, atr_mult_riskoff)
        total_pnl += (net_ret * 100.0)

    return total_pnl

def objective(trial, sig, csvs, macro_df):
    params = {
        'tp': trial.suggest_float('tp', 2.0, 5.0, step=0.5),
        'sl': trial.suggest_float('sl', 0.2, 0.6, step=0.1),
        # Quant AI optimizes Econ-specific rules:
        'atr_mult_riskon': trial.suggest_float('atr_mult_riskon', 10.0, 20.0, step=1.0),
        'atr_mult_riskoff': trial.suggest_float('atr_mult_riskoff', 2.0, 8.0, step=1.0),
        'horizon_days': trial.suggest_int('horizon_days', 30, 90, step=15),
    }
    return evaluate_params(params, sig, csvs, macro_df)


def time_split(sig, frac=0.7):
    """Chronological hold-out: earliest `frac` of signal dates train, the rest validate."""
    dates_sorted = np.sort(sig['Date'].unique())
    cutoff = pd.Timestamp(dates_sorted[int(len(dates_sorted) * frac)])
    return sig[sig['Date'] <= cutoff], sig[sig['Date'] > cutoff], cutoff

def main():
    print("="*60)
    print(" AI OPTIMIZER STARTED (Econ + Quant)")
    print("="*60)
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof = pd.read_parquet(os.path.join(base, 'data/model/v2/m1_100pct_60d/entry_oof.parquet'))
    oof['Date'] = pd.to_datetime(oof['Date'])
    sig = oof[oof['oof_prob_xgb'] >= 0.70]

    # Chronological train/validation split: tune on the past, report on the unseen future.
    sig_train, sig_valid, cutoff = time_split(sig, frac=0.7)
    print(f"Train: {len(sig_train)} signals (<= {cutoff.date()}) | "
          f"Validation: {len(sig_valid)} signals (> {cutoff.date()})")

    # Load Macro Data
    min_date = oof['Date'].min() - pd.Timedelta(days=365) # Get enough history for 200 SMA
    max_date = oof['Date'].max() + pd.Timedelta(days=30)
    macro_df = load_macro_data(min_date, max_date)

    suffix = '_4h_full.csv'
    csvs = {os.path.basename(c).replace(suffix, '').upper(): c
            for c in glob.glob(os.path.join(base, 'data', '*', f'*{suffix}'))}

    study = optuna.create_study(direction='maximize')

    print("AI is fusing Economics and Quant math to maximize PnL (train slice)...")
    study.optimize(lambda trial: objective(trial, sig_train, csvs, macro_df), n_trials=30, n_jobs=1)

    val_pnl = evaluate_params(study.best_params, sig_valid, csvs, macro_df)

    print("\n" + "="*60)
    print(" AI OPTIMIZATION FINISHED")
    print("="*60)
    print(f"IN-SAMPLE (train) best PnL:  ${study.best_value:,.2f}")
    print(f"OUT-OF-SAMPLE (validation):  ${val_pnl:,.2f}")
    print("Best Parameters (Econ + Quant):")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == '__main__':
    main()
