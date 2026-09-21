import pandas as pd
import numpy as np

class RealisticBacktester:
    def __init__(self, df, tp_pct, sl_pct, max_bars=300):
        self.df = df.copy()
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_bars = max_bars

    def run(self):
        trades = []
        
        # Ensure we have predictions
        if 'ML_Prob_Long' not in self.df.columns or 'ML_Prob_Short' not in self.df.columns:
            print("  [ERROR] Missing ML_Prob_Long or ML_Prob_Short columns for backtesting.")
            return pd.DataFrame()
            
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        entry_date = None
        pos_type = None # 'Long' or 'Short'
        
        dates = self.df['Date'].values
        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values
        prob_long = self.df['ML_Prob_Long'].values
        prob_short = self.df['ML_Prob_Short'].values
        
        n = len(self.df)
        
        for i in range(n):
            if not in_position:
                # Check for entry signals
                is_long = prob_long[i] > 0.8
                is_short = prob_short[i] > 0.8
                
                if is_long and is_short:
                    continue # Conflict, do nothing
                    
                if is_long:
                    in_position = True
                    pos_type = 'Long'
                    entry_price = closes[i]
                    entry_idx = i
                    entry_date = dates[i]
                elif is_short:
                    in_position = True
                    pos_type = 'Short'
                    entry_price = closes[i]
                    entry_idx = i
                    entry_date = dates[i]
            else:
                # Check for exits
                bars_held = i - entry_idx
                exit_reason = None
                exit_price = 0.0
                
                if pos_type == 'Long':
                    tp_price = entry_price * (1 + self.tp_pct)
                    sl_price = entry_price * (1 - self.sl_pct)
                    
                    if lows[i] <= sl_price:
                        exit_reason = 'Hit SL'
                        exit_price = sl_price
                    elif highs[i] >= tp_price:
                        exit_reason = 'Hit TP'
                        exit_price = tp_price
                    elif bars_held >= self.max_bars:
                        exit_reason = 'Time-out'
                        exit_price = closes[i]
                        
                elif pos_type == 'Short':
                    tp_price = entry_price * (1 - self.tp_pct)
                    sl_price = entry_price * (1 + self.sl_pct)
                    
                    if highs[i] >= sl_price:
                        exit_reason = 'Hit SL'
                        exit_price = sl_price
                    elif lows[i] <= tp_price:
                        exit_reason = 'Hit TP'
                        exit_price = tp_price
                    elif bars_held >= self.max_bars:
                        exit_reason = 'Time-out'
                        exit_price = closes[i]
                
                # If an exit occurred
                if exit_reason:
                    pnl_pct = (exit_price - entry_price) / entry_price
                    if pos_type == 'Short':
                        pnl_pct = -pnl_pct
                        
                    trades.append({
                        'Entry_Date': entry_date,
                        'Entry_Idx': entry_idx,
                        'Type': pos_type,
                        'Entry_Price': entry_price,
                        'Exit_Date': dates[i],
                        'Exit_Idx': i,
                        'Exit_Price': exit_price,
                        'PnL_Pct': pnl_pct,
                        'Reason': exit_reason,
                        'Bars_Held': bars_held
                    })
                    in_position = False
        
        # Close any open position at the end
        if in_position:
            pnl_pct = (closes[-1] - entry_price) / entry_price
            if pos_type == 'Short':
                pnl_pct = -pnl_pct
            trades.append({
                'Entry_Date': entry_date,
                'Entry_Idx': entry_idx,
                'Type': pos_type,
                'Entry_Price': entry_price,
                'Exit_Date': dates[-1],
                'Exit_Idx': n - 1,
                'Exit_Price': closes[-1],
                'PnL_Pct': pnl_pct,
                'Reason': 'End of Data',
                'Bars_Held': n - 1 - entry_idx
            })
            
        trade_df = pd.DataFrame(trades)
        return trade_df
