import os
import glob
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import optuna
import warnings
warnings.filterwarnings('ignore')

def load_macro_data(start_date, end_date):
    dxy_df = yf.download("DX-Y.NYB", start=start_date, end=end_date, progress=False)
    spy_df = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    dxy = dxy_df.xs('Close', level='Price', axis=1)['DX-Y.NYB'] if isinstance(dxy_df.columns, pd.MultiIndex) else dxy_df['Close']
    spy = spy_df.xs('Close', level='Price', axis=1)['^GSPC'] if isinstance(spy_df.columns, pd.MultiIndex) else spy_df['Close']
    macro = pd.DataFrame({'DXY': dxy, 'SP500': spy})
    macro.index = pd.to_datetime(macro.index)
    macro['DXY_SMA50'] = macro['DXY'].rolling(50).mean()
    macro['DXY_SMA200'] = macro['DXY'].rolling(200).mean()
    macro['SP500_SMA50'] = macro['SP500'].rolling(50).mean()
    macro['SP500_SMA200'] = macro['SP500'].rolling(200).mean()
    macro['Econ_Risk_On'] = np.where((macro['SP500_SMA50'] > macro['SP500_SMA200']) & 
                                     (macro['DXY_SMA50'] < macro['DXY_SMA200']), 1, 
                            np.where((macro['SP500_SMA50'] < macro['SP500_SMA200']) & 
                                     (macro['DXY_SMA50'] > macro['DXY_SMA200']), -1, 0))
    return macro.reset_index().rename(columns={'Date': 'Macro_Date'})

def calculate_indicators(df):
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    # Money Flow Index (MFI) - uses price and volume
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    # Chaikin Money Flow (CMF) - measures buying/selling pressure
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    
    df['ATR'] = df['ATR'].ffill().bfill()
    df['MFI'] = df['MFI'].ffill().bfill()
    df['CMF'] = df['CMF'].ffill().bfill()
    return df

def simulate_moneyflow_fast(df, sig_dates, tp, sl, horizon_bars, fee, slip, atr_mult_riskon, atr_mult_riskoff, cmf_thresh, mfi_thresh):
    df = df.sort_values('Date').reset_index(drop=True)
    o, hi, lo, cl = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    atr = df['ATR'].values
    mfi = df['MFI'].values
    cmf = df['CMF'].values
    econ_regime = df['Econ_Risk_On'].values
    dt = df['Date'].values
    
    date_to_pos = {d: i for i, d in enumerate(dt)}
    sig_pos = sorted(date_to_pos[d] for d in sig_dates if d in date_to_pos)

    total_net = 0.0
    notional = 100.0 # STRICTLY FIXED AT $100
    
    cooldown_until = -1
    for s in sig_pos:
        if s <= cooldown_until: continue
        entry_i = s + 1
        if entry_i >= len(df): continue
        entry_px = o[entry_i]
        if entry_px <= 0 or not np.isfinite(entry_px): continue
        
        end = min(entry_i + horizon_bars, len(df) - 1)
        fixed_dn = entry_px * (1 - sl)
        highest_seen = hi[entry_i]
        exit_i, exit_px = end, cl[end]
        
        current_econ = econ_regime[entry_i]
        active_atr_mult = atr_mult_riskon if current_econ >= 0 else atr_mult_riskoff
        
        for j in range(entry_i, end + 1):
            highest_seen = max(highest_seen, hi[j])
            
            # EARLY EXIT via Money Flow:
            # If CMF drops below negative threshold (Smart Money exiting) 
            # OR MFI drops below a panic threshold, we exit early BEFORE hitting the trailing stop.
            if cmf[j] < cmf_thresh or mfi[j] < mfi_thresh:
                exit_i, exit_px = j, cl[j]
                break
                
            current_sl = max(fixed_dn, highest_seen - (atr[j] * active_atr_mult))
            if lo[j] <= current_sl:
                exit_i, exit_px = j, current_sl
                break
                    
        gross = exit_px / entry_px - 1.0
        net = gross - 2 * (fee + slip)
        total_net += (net * notional)
        cooldown_until = exit_i
        
    return total_net

def objective(trial, sig, csvs, macro_df):
    tp = 3.5 # not strictly used if we just trail and early exit
    sl = trial.suggest_float('sl', 0.4, 0.7, step=0.05)
    
    # Quant AI optimizes Econ + Money Flow rules:
    atr_mult_riskon = trial.suggest_float('atr_mult_riskon', 12.0, 20.0, step=1.0)
    atr_mult_riskoff = trial.suggest_float('atr_mult_riskoff', 2.0, 8.0, step=1.0)
    horizon_days = trial.suggest_int('horizon_days', 60, 120, step=10)
    
    # Money Flow parameters
    cmf_thresh = trial.suggest_float('cmf_thresh', -0.5, -0.05, step=0.05) # Exit if CMF is very negative
    mfi_thresh = trial.suggest_float('mfi_thresh', 10, 40, step=5) # Exit if MFI plunges
    
    total_pnl = 0.0
    
    for sym, grp in sig.groupby('symbol'):
        csv = csvs.get(sym)
        if csv is None: continue
        
        df = pd.read_csv(csv)
        df['Date'] = pd.to_datetime(df['Date'])
        
        df['Macro_Date'] = df['Date'].dt.normalize()
        df = pd.merge(df, macro_df[['Macro_Date', 'Econ_Risk_On']], on='Macro_Date', how='left')
        df['Econ_Risk_On'] = df['Econ_Risk_On'].ffill().fillna(0)
        
        df = calculate_indicators(df)
        
        dates_set = set(grp['Date'])
        horizon_bars = horizon_days * 6 
            
        total_pnl += simulate_moneyflow_fast(df, dates_set, tp, sl, horizon_bars, 0.001, 0.0005, 
                                             atr_mult_riskon, atr_mult_riskoff, cmf_thresh, mfi_thresh)
        
    return total_pnl

def main():
    print("="*60)
    print(" AI OPTIMIZER: MONEY FLOW + ECON (Fixed $100)")
    print("="*60)
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof = pd.read_parquet(os.path.join(base, 'data/model/v2/m1_100pct_60d/entry_oof.parquet'))
    oof['Date'] = pd.to_datetime(oof['Date'])
    sig = oof[oof['oof_prob_xgb'] >= 0.70]
    
    min_date = oof['Date'].min() - pd.Timedelta(days=365) 
    max_date = oof['Date'].max() + pd.Timedelta(days=30)
    macro_df = load_macro_data(min_date, max_date)
    
    suffix = '_4h_full.csv'
    csvs = {os.path.basename(c).replace(suffix, '').upper(): c for c in glob.glob(os.path.join(base, 'data', '*', f'*{suffix}'))}
            
    study = optuna.create_study(direction='maximize')
    
    print("AI is optimizing CMF (Chaikin Money Flow) and MFI exits with fixed $100 per trade...")
    study.optimize(lambda trial: objective(trial, sig, csvs, macro_df), n_trials=40, n_jobs=1)
    
    print("\n" + "="*60)
    print(" AI OPTIMIZATION FINISHED")
    print("="*60)
    print(f"Best PnL found: ${study.best_value:,.2f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == '__main__':
    main()
