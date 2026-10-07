import os
import glob
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
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
    df['ATR'] = df['ATR'].ffill().bfill()
    return df

def run_evaluation():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof = pd.read_parquet(os.path.join(base, 'data/model/v2/m1_100pct_60d/entry_oof.parquet'))
    oof['Date'] = pd.to_datetime(oof['Date'])
    sig = oof[oof['oof_prob_xgb'] >= 0.70]
    
    macro_df = load_macro_data(oof['Date'].min() - pd.Timedelta(days=365), oof['Date'].max() + pd.Timedelta(days=30))
    
    suffix = '_4h_full.csv'
    csvs = {os.path.basename(c).replace(suffix, '').upper(): c for c in glob.glob(os.path.join(base, 'data', '*', f'*{suffix}'))}
            
    # Best Params
    tp = 3.5
    sl = 0.6
    atr_mult_riskon = 16.0
    atr_mult_riskoff = 2.0
    horizon_bars = 75 * 6
    notional = 100.0 # FIXED SIZING FOR APPLES-TO-APPLES
    fee = 0.001
    slip = 0.0005
    
    total_pnl = 0.0
    trades_count = 0
    wins = 0
    
    for sym, grp in sig.groupby('symbol'):
        csv = csvs.get(sym)
        if csv is None: continue
        df = pd.read_csv(csv)
        df['Date'] = pd.to_datetime(df['Date'])
        
        df['Macro_Date'] = df['Date'].dt.normalize()
        df = pd.merge(df, macro_df[['Macro_Date', 'Econ_Risk_On']], on='Macro_Date', how='left')
        df['Econ_Risk_On'] = df['Econ_Risk_On'].ffill().fillna(0)
        df = calculate_indicators(df)
        
        df = df.sort_values('Date').reset_index(drop=True)
        o, hi, lo, cl = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
        atr = df['ATR'].values
        econ_regime = df['Econ_Risk_On'].values
        dt = df['Date'].values
        
        date_to_pos = {d: i for i, d in enumerate(dt)}
        sig_pos = sorted(date_to_pos[d] for d in set(grp['Date']) if d in date_to_pos)

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
                current_sl = max(fixed_dn, highest_seen - (atr[j] * active_atr_mult))
                if lo[j] <= current_sl:
                    exit_i, exit_px = j, current_sl
                    break
                        
            gross = exit_px / entry_px - 1.0
            net = (gross - 2 * (fee + slip)) 
            total_pnl += (net * notional)
            trades_count += 1
            if net > 0: wins += 1
            cooldown_until = exit_i

    print("="*60)
    print(" APPLES-TO-APPLES COMPARISON (Econ Exits, Fixed $100 Size)")
    print("="*60)
    print(f"Trades Executed: {trades_count}")
    print(f"Win Rate:        {wins/trades_count*100:.1f}%")
    print(f"Total PnL:       ${total_pnl:,.2f}")
    print("="*60)

if __name__ == '__main__':
    run_evaluation()
