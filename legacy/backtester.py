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
                 long_threshold=0.8, short_threshold=0.8,
                 stagnation_bars=42, stagnation_min_profit_pct=0.0,
                 risk_pct=0.02, kelly_fraction=1.0, max_position=3.0,
                 fee_pct=0.001, slippage_pct=0.0005, initial_equity=10000.0):
        """
        Exit hierarchy — AGENTS.md Rule #6 (evaluated in this exact order):
          1. Hard Risk (ATR-based Trailing Stop)
                Long : close if price drops below (Highest since entry - X*ATR)
                Short: close if price rises above (Lowest since entry + X*ATR)
          2. Price not moving (too stable)
                Close if the trade has NOT reached profit within `stagnation_bars`
                (7 days on 4H = 42 bars).

        Parameters:
            df: DataFrame with OHLCV + ML_Prob_Long + ML_Prob_Short + ATR_14
            tp_pct: Fixed Take Profit % (None = disabled when trailing is on)
            sl_pct: Fixed Stop Loss % (fallback if ATR not available)
            max_bars: Absolute safety time-out (bars) if nothing else fires
            trail_pct: Trailing Stop percentage from peak (0.15 = 15%)
            atr_sl_mult: ATR multiplier for initial Stop Loss (2.0 = 2×ATR)
            use_trailing: If True, use trailing stop instead of fixed TP

            long_threshold / short_threshold: Min ML prob to open a position.
                Should come from the optimizer's out-of-fold threshold.

            stagnation_bars: Rule #6 layer 2. After this many bars, if the trade
                is still not profitable, close it (7 days on 4H = 42). None = off.
            stagnation_min_profit_pct: Profit (as a fraction of entry) the trade
                must have reached by `stagnation_bars` to be allowed to stay open
                (0.0 = any profit keeps it; 0.02 = must be at least +2%).

            --- Position sizing (AGENTS.md Rule #7 — Fractional Kelly) ---
            risk_pct: Max fraction of equity risked per trade (0.02 = 2%).
                Base position (as a fraction of equity) = risk_pct / SL_distance%,
                i.e. a tighter stop => a larger position for the same $ risk.
            kelly_fraction: Scales the base position (0.5 = half-Kelly). The base
                is further multiplied by the ML predict_proba (confidence), so a
                85%-confident signal takes 85% of the base size.
            max_position: Cap on position as a fraction of equity (leverage cap).

            --- Realistic costs (AGENTS.md Rule #9) ---
            fee_pct: Exchange fee per side (0.001 = 0.1%). Charged on entry + exit.
            slippage_pct: Simulated slippage per side. Charged on entry + exit.
            initial_equity: Starting equity for the compounded net equity curve.
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
        self.stagnation_bars = stagnation_bars
        self.stagnation_min_profit_pct = stagnation_min_profit_pct
        self.risk_pct = risk_pct
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.initial_equity = initial_equity

    def _size_and_costs(self, entry_price, initial_sl, entry_prob, gross_pnl_pct):
        """Fractional-Kelly position sizing (Rule #7) + realistic costs (Rule #9).

        Returns (size_frac, sl_dist_pct, cost_pct, net_return_on_equity) where:
          size_frac  = position notional as a fraction of equity
                     = clamp( (risk_pct / SL_distance%) * predict_proba * kelly ,
                              0, max_position )
          cost_pct   = round-trip fees + slippage charged on the notional
          net_return = size_frac * gross_pnl_pct - cost_pct   (equity fraction)
        """
        sl_dist = abs(entry_price - initial_sl) / entry_price if entry_price > 0 else 0.0
        # Floor the stop distance so an accidental ~0 stop can't imply huge size.
        sl_dist = max(sl_dist, 0.01)
        base_frac = self.risk_pct / sl_dist
        size_frac = base_frac * float(entry_prob) * self.kelly_fraction
        size_frac = max(0.0, min(size_frac, self.max_position))
        # Fees + slippage are paid on entry AND exit (round trip) on the notional.
        cost_pct = size_frac * 2.0 * (self.fee_pct + self.slippage_pct)
        net_return = size_frac * gross_pnl_pct - cost_pct
        return size_frac, sl_dist, cost_pct, net_return

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
        entry_prob = 0.0  # ML confidence at the signal bar (for Kelly sizing)
        equity = self.initial_equity  # compounded net equity (Rule #7 & #9)

        dates = self.df['Date'].values
        opens = self.df['Open'].values
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
                    # AGENTS.md Rule #1: the signal at bar i is built from data
                    # up to bar i-1 (features are .shift(1)-ed). We do NOT fill at
                    # bar i's Close (that would use end-of-bar info); we execute at
                    # the OPEN of the NEXT bar, the first tradable price after the
                    # signal is known. No next bar => no fill.
                    entry_bar = i + 1
                    if entry_bar >= n:
                        continue

                    in_position = True
                    pos_type = 'Long' if is_long else 'Short'
                    entry_price = opens[entry_bar]
                    entry_idx = entry_bar
                    entry_date = dates[entry_bar]
                    breakeven_triggered = False
                    # ML confidence that fired the signal (measured at the signal
                    # bar i, before the fill) — drives the Kelly size multiplier.
                    entry_prob = prob_long[i] if pos_type == 'Long' else prob_short[i]

                    # Calculate initial SL (ATR at the entry bar)
                    if has_atr and atr_values[entry_bar] > 0:
                        atr_val = atr_values[entry_bar]
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
                # ===== EXIT LOGIC (AGENTS.md Rule #6, in priority order) =====
                bars_held = i - entry_idx
                exit_reason = None
                exit_price = 0.0

                if pos_type == 'Long':
                    # Track the highest price since entry (for the trailing stop).
                    if highs[i] > peak_price:
                        peak_price = highs[i]

                    # Break-even trigger: move SL to entry once profit >= 1×ATR.
                    if not breakeven_triggered and has_atr:
                        atr_at_entry = atr_values[entry_idx] if atr_values[entry_idx] > 0 else 0
                        if atr_at_entry > 0 and (peak_price - entry_price) >= atr_at_entry:
                            current_sl = max(current_sl, entry_price)
                            breakeven_triggered = True

                    # --- Layer 1: Hard Risk (ATR trailing stop) ---
                    if self.use_trailing and breakeven_triggered:
                        trailing_sl = peak_price * (1 - self.trail_pct)
                        current_sl = max(current_sl, trailing_sl)

                    if lows[i] <= current_sl:
                        exit_reason = 'Trailing SL' if breakeven_triggered else 'Initial SL'
                        exit_price = current_sl

                    elif not self.use_trailing and self.tp_pct:
                        tp_price = entry_price * (1 + self.tp_pct)
                        if highs[i] >= tp_price:
                            exit_reason = 'Hit TP'
                            exit_price = tp_price

                    # --- Layer 2: Price not moving — not profitable within N bars ---
                    elif self.stagnation_bars and bars_held >= self.stagnation_bars:
                        profit = (closes[i] - entry_price) / entry_price
                        if profit <= self.stagnation_min_profit_pct:
                            exit_reason = 'Stagnation (no profit in time)'
                            exit_price = closes[i]

                    # Absolute safety time-out
                    if exit_reason is None and bars_held >= self.max_bars:
                        exit_reason = 'Time-out'
                        exit_price = closes[i]

                elif pos_type == 'Short':
                    # Track the lowest price since entry (for the trailing stop).
                    if lows[i] < peak_price:
                        peak_price = lows[i]

                    # Break-even trigger: move SL to entry once profit >= 1×ATR.
                    if not breakeven_triggered and has_atr:
                        atr_at_entry = atr_values[entry_idx] if atr_values[entry_idx] > 0 else 0
                        if atr_at_entry > 0 and (entry_price - peak_price) >= atr_at_entry:
                            current_sl = min(current_sl, entry_price)
                            breakeven_triggered = True

                    # --- Layer 1: Hard Risk (ATR trailing stop) ---
                    if self.use_trailing and breakeven_triggered:
                        trailing_sl = peak_price * (1 + self.trail_pct)
                        current_sl = min(current_sl, trailing_sl)

                    if highs[i] >= current_sl:
                        exit_reason = 'Trailing SL' if breakeven_triggered else 'Initial SL'
                        exit_price = current_sl

                    elif not self.use_trailing and self.tp_pct:
                        tp_price = entry_price * (1 - self.tp_pct)
                        if lows[i] <= tp_price:
                            exit_reason = 'Hit TP'
                            exit_price = tp_price

                    # --- Layer 2: Price not moving — not profitable within N bars ---
                    elif self.stagnation_bars and bars_held >= self.stagnation_bars:
                        profit = (entry_price - closes[i]) / entry_price
                        if profit <= self.stagnation_min_profit_pct:
                            exit_reason = 'Stagnation (no profit in time)'
                            exit_price = closes[i]

                    # Absolute safety time-out
                    if exit_reason is None and bars_held >= self.max_bars:
                        exit_reason = 'Time-out'
                        exit_price = closes[i]

                # If an exit occurred
                if exit_reason:
                    pnl_pct = (exit_price - entry_price) / entry_price
                    if pos_type == 'Short':
                        pnl_pct = -pnl_pct

                    size_frac, sl_dist, cost_pct, net_return = self._size_and_costs(
                        entry_price, initial_sl, entry_prob, pnl_pct)
                    equity *= (1.0 + net_return)

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
                        'Entry_Prob': entry_prob,
                        'SL_Dist_Pct': sl_dist,
                        'Size_Frac': size_frac,
                        'Cost_Pct': cost_pct,
                        'Net_Return': net_return,
                        'Equity': equity,
                        'Reason': exit_reason,
                        'Bars_Held': bars_held
                    })
                    in_position = False

        # Close any open position at the end
        if in_position:
            pnl_pct = (closes[-1] - entry_price) / entry_price
            if pos_type == 'Short':
                pnl_pct = -pnl_pct
            size_frac, sl_dist, cost_pct, net_return = self._size_and_costs(
                entry_price, initial_sl, entry_prob, pnl_pct)
            equity *= (1.0 + net_return)
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
                'Entry_Prob': entry_prob,
                'SL_Dist_Pct': sl_dist,
                'Size_Frac': size_frac,
                'Cost_Pct': cost_pct,
                'Net_Return': net_return,
                'Equity': equity,
                'Reason': 'End of Data',
                'Bars_Held': n - 1 - entry_idx
            })

        trade_df = pd.DataFrame(trades)
        return trade_df
