"""
backtester.py — Realistic Backtester with ATR-based SL + Trailing Stop

Exit Strategy:
  Layer 2 (Risk): SL = Entry - ATR×2 (Long) หรือ Entry + ATR×2 (Short)
                  เลื่อน SL มา Break-even เมื่อกำไร >= ATR×1
  Layer 3 (Exit): Trailing Stop 15% จาก Peak (ไม่มี Fixed TP → ปล่อยกำไรวิ่ง)
"""
import pandas as pd
import numpy as np


class RealisticBacktester:
    def __init__(self, df, tp_pct=None, sl_pct=None, max_bars=300,
                 trail_pct=0.15, atr_sl_mult=2.0, use_trailing=True,
                 long_threshold=0.8, short_threshold=0.8):
        """
        Parameters:
            df: DataFrame with OHLCV + ML_Prob_Long + ML_Prob_Short + ATR_14
            tp_pct: Fixed Take Profit % (None = disabled when trailing is on)
            sl_pct: Fixed Stop Loss % (fallback if ATR not available)
            max_bars: Maximum bars before force-exit (time-out)
            trail_pct: Trailing Stop percentage from peak (0.15 = 15%)
            atr_sl_mult: ATR multiplier for initial Stop Loss (2.0 = 2×ATR)
            use_trailing: If True, use trailing stop instead of fixed TP
            long_threshold: Min ML_Prob_Long to open a Long. Should come from the
                            optimizer's out-of-fold threshold, NOT a magic number.
            short_threshold: Min ML_Prob_Short to open a Short (from optimizer).
        """
        self.df = df.copy()
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_bars = max_bars
        self.trail_pct = trail_pct
        self.atr_sl_mult = atr_sl_mult
        self.use_trailing = use_trailing
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def run(self):
        trades = []

        # Ensure we have predictions
        if 'ML_Prob_Long' not in self.df.columns or 'ML_Prob_Short' not in self.df.columns:
            print("  [ERROR] Missing ML_Prob_Long or ML_Prob_Short columns for backtesting.")
            return pd.DataFrame()

        # Check for ATR column (needed for ATR-based SL)
        has_atr = 'ATR_14' in self.df.columns

        in_position = False
        entry_price = 0.0
        entry_idx = 0
        entry_date = None
        pos_type = None  # 'Long' or 'Short'
        initial_sl = 0.0
        current_sl = 0.0
        peak_price = 0.0  # Track highest (Long) or lowest (Short) since entry
        breakeven_triggered = False

        dates = self.df['Date'].values
        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values
        prob_long = self.df['ML_Prob_Long'].values
        prob_short = self.df['ML_Prob_Short'].values
        atr_values = self.df['ATR_14'].values if has_atr else None

        n = len(self.df)

        for i in range(n):
            if not in_position:
                # Check for entry signals against each model's own threshold.
                is_long = prob_long[i] >= self.long_threshold
                is_short = prob_short[i] >= self.short_threshold

                if is_long and is_short:
                    # Both models fire: take the STRONGER conviction, measured as
                    # the margin above each model's own threshold (thresholds may
                    # differ, so compare like-for-like, not raw probabilities).
                    long_margin = prob_long[i] - self.long_threshold
                    short_margin = prob_short[i] - self.short_threshold
                    if long_margin >= short_margin:
                        is_short = False
                    else:
                        is_long = False

                if is_long or is_short:
                    in_position = True
                    pos_type = 'Long' if is_long else 'Short'
                    entry_price = closes[i]
                    entry_idx = i
                    entry_date = dates[i]
                    breakeven_triggered = False

                    # Calculate initial SL
                    if has_atr and atr_values[i] > 0:
                        atr_val = atr_values[i]
                        if pos_type == 'Long':
                            initial_sl = entry_price - (atr_val * self.atr_sl_mult)
                            peak_price = entry_price
                        else:
                            initial_sl = entry_price + (atr_val * self.atr_sl_mult)
                            peak_price = entry_price
                    elif self.sl_pct:
                        # Fallback to fixed SL%
                        if pos_type == 'Long':
                            initial_sl = entry_price * (1 - self.sl_pct)
                            peak_price = entry_price
                        else:
                            initial_sl = entry_price * (1 + self.sl_pct)
                            peak_price = entry_price
                    else:
                        # No SL at all (not recommended)
                        initial_sl = 0 if pos_type == 'Long' else entry_price * 10
                        peak_price = entry_price

                    current_sl = initial_sl

            else:
                # ===== EXIT LOGIC =====
                bars_held = i - entry_idx
                exit_reason = None
                exit_price = 0.0

                if pos_type == 'Long':
                    # Update peak price
                    if highs[i] > peak_price:
                        peak_price = highs[i]

                    # --- Layer 2: Break-even trigger ---
                    # Move SL to entry when profit >= 1×ATR
                    if not breakeven_triggered and has_atr:
                        atr_at_entry = atr_values[entry_idx] if atr_values[entry_idx] > 0 else 0
                        if atr_at_entry > 0 and (peak_price - entry_price) >= atr_at_entry:
                            current_sl = max(current_sl, entry_price)
                            breakeven_triggered = True

                    # --- Layer 3: Trailing Stop ---
                    if self.use_trailing and breakeven_triggered:
                        trailing_sl = peak_price * (1 - self.trail_pct)
                        current_sl = max(current_sl, trailing_sl)

                    # Check SL hit
                    if lows[i] <= current_sl:
                        exit_reason = 'Trailing SL' if breakeven_triggered else 'Initial SL'
                        exit_price = current_sl

                    # Check Fixed TP (only if not using trailing)
                    elif not self.use_trailing and self.tp_pct:
                        tp_price = entry_price * (1 + self.tp_pct)
                        if highs[i] >= tp_price:
                            exit_reason = 'Hit TP'
                            exit_price = tp_price

                    # Time-out
                    elif bars_held >= self.max_bars:
                        exit_reason = 'Time-out'
                        exit_price = closes[i]

                elif pos_type == 'Short':
                    # Update peak (lowest price for short)
                    if lows[i] < peak_price:
                        peak_price = lows[i]

                    # --- Layer 2: Break-even trigger ---
                    if not breakeven_triggered and has_atr:
                        atr_at_entry = atr_values[entry_idx] if atr_values[entry_idx] > 0 else 0
                        if atr_at_entry > 0 and (entry_price - peak_price) >= atr_at_entry:
                            current_sl = min(current_sl, entry_price)
                            breakeven_triggered = True

                    # --- Layer 3: Trailing Stop ---
                    if self.use_trailing and breakeven_triggered:
                        trailing_sl = peak_price * (1 + self.trail_pct)
                        current_sl = min(current_sl, trailing_sl)

                    # Check SL hit
                    if highs[i] >= current_sl:
                        exit_reason = 'Trailing SL' if breakeven_triggered else 'Initial SL'
                        exit_price = current_sl

                    # Check Fixed TP (only if not using trailing)
                    elif not self.use_trailing and self.tp_pct:
                        tp_price = entry_price * (1 - self.tp_pct)
                        if lows[i] <= tp_price:
                            exit_reason = 'Hit TP'
                            exit_price = tp_price

                    # Time-out
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
                        'Peak_Price': peak_price,
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
                'Peak_Price': peak_price,
                'PnL_Pct': pnl_pct,
                'Reason': 'End of Data',
                'Bars_Held': n - 1 - entry_idx
            })

        trade_df = pd.DataFrame(trades)
        return trade_df
