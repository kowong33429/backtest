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
    # FINANCE METRIC: Asset Volatility Profile (Risk Beta proxy)
    # ATR as a percentage of price tells us how "risky/explosive" this asset is
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100 
    df['ATR'] = df['ATR'].ffill().bfill()
    df['ATR_Pct'] = df['ATR_Pct'].ffill().bfill()
    return df

def simulate_finance_rotation(df, sig_dict, sl, horizon_bars, atr_mult, risk_on_only_high_beta, risk_off_only_bluechip, high_beta_threshold):
    df = df.sort_values('Date').reset_index(drop=True)
    o, hi, lo, cl = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    atr = df['ATR'].values
    atr_pct = df['ATR_Pct'].values
    econ_regime = df['Econ_Risk_On'].values
    dt = df['Date'].values
    
    date_to_pos = {d: i for i, d in enumerate(dt)}
    valid_sigs = {date_to_pos[d]: prob for d, prob in sig_dict.items() if d in date_to_pos}
    sig_pos = sorted(valid_sigs.keys())

    total_net = 0.0
    notional = 100.0 # STRICTLY FIXED AT $100
    fee, slip = 0.001, 0.0005
    
    cooldown_until = -1
    for s in sig_pos:
        if s <= cooldown_until: continue
        
        entry_i = s + 1
        if entry_i >= len(df): continue
        
        current_econ = econ_regime[s]
        asset_risk_profile = atr_pct[s]
        
        is_high_beta = asset_risk_profile > high_beta_threshold
        
        # FINANCE SECTOR ROTATION LOGIC:
        if current_econ == -1: # Bear Market / Risk-Off
            if risk_off_only_bluechip and is_high_beta:
                # Traditional Finance dumps high-risk penny stocks during recessions
                continue 
        elif current_econ == 1: # Bull Market / Risk-On
            if risk_on_only_high_beta and not is_high_beta:
                # Traditional Finance rotates out of boring blue-chips into high-growth risky assets
                continue

        entry_px = o[entry_i]
        if entry_px <= 0 or not np.isfinite(entry_px): continue
        
        end = min(entry_i + horizon_bars, len(df) - 1)
        fixed_dn = entry_px * (1 - sl)
        highest_seen = hi[entry_i]
        exit_i, exit_px = end, cl[end]
        
        # Exit trailing stop logic
        for j in range(entry_i, end + 1):
            highest_seen = max(highest_seen, hi[j])
            current_sl = max(fixed_dn, highest_seen - (atr[j] * atr_mult))
            if lo[j] <= current_sl:
                exit_i, exit_px = j, current_sl
                break
                    
        gross = exit_px / entry_px - 1.0
        net = gross - 2 * (fee + slip)
        total_net += (net * notional)
        cooldown_until = exit_i
        
    return total_net

def objective(trial, sig, csvs, macro_df):
    sl = trial.suggest_float('sl', 0.5, 0.8, step=0.1)
    atr_mult = trial.suggest_float('atr_mult', 12.0, 18.0, step=1.0)
    horizon_days = trial.suggest_int('horizon_days', 60, 120, step=15)
    
    # Finance Portfolio Rotation Flags
    risk_on_only_high_beta = trial.suggest_categorical('risk_on_only_high_beta', [True, False])
    risk_off_only_bluechip = trial.suggest_categorical('risk_off_only_bluechip', [True, False])
    high_beta_threshold = trial.suggest_float('high_beta_threshold', 3.0, 8.0, step=1.0) # ATR % threshold
    
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
        sig_dict = dict(zip(grp['Date'], grp['oof_prob_xgb']))
        horizon_bars = horizon_days * 6 
            
        total_pnl += simulate_finance_rotation(df, sig_dict, sl, horizon_bars, atr_mult, 
                                               risk_on_only_high_beta, risk_off_only_bluechip, high_beta_threshold)
        
    return total_pnl

def main():
    print("="*60)
    print(" AI OPTIMIZER: FINANCE SECTOR ROTATION (Fixed $100)")
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
    
    print("AI is applying Traditional Finance 'Asset Rotation & Beta' strategies...")
    study.optimize(lambda trial: objective(trial, sig, csvs, macro_df), n_trials=30, n_jobs=1)
    
    print("\n" + "="*60)
    print(" AI OPTIMIZATION FINISHED")
    print("="*60)
    print(f"Best PnL found: ${study.best_value:,.2f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == '__main__':
    main()
