"""
labels.py — Dynamic Triple-Barrier Labeling (AGENTS.md Rule #3 & #5)

Concept (faithful to AGENTS.md):
  For EVERY candle we open a hypothetical trade at the Open of that candle
  (the only unshifted execution price, per Rule #1) and race three barriers:

    1. Upper barrier   (Take Profit)
    2. Lower barrier   (Stop Loss)
    3. Vertical barrier (time limit, e.g. 700+ bars for extreme trend-following)

  Directional logic (Rule #5):
    - Long : TP = Entry * (1 + tp)   SL = Entry * (1 - sl)
    - Short: TP = Entry * (1 - tp)   SL = Entry * (1 + sl)   (inverted)

  Long and Short are evaluated INDEPENDENTLY and written to two separate
  binary columns, because Step 4 trains two separate binary classifiers
  (one Long model, one Short model — AGENTS.md Step 4 / Rule #5):

    - Label_Long  = 1 if the Long  TP is touched before the Long  SL, else 0
    - Label_Short = 1 if the Short TP is touched before the Short SL, else 0

  A single candle can be a valid Long AND a valid Short at once (each is a
  positive example for its own model); collapsing them into one class would
  discard a training positive, so we keep them as two columns.

  "Dynamic" barriers: when `use_atr=True` and an ATR column is present, the
  barrier distances are scaled by volatility (ATR * multiplier) instead of a
  fixed percentage, so the target adapts to each coin's regime.

  NOTE: Labels are allowed to look into the future — that is not leakage. The
  leakage guard (.shift(1)) applies to FEATURES only and is handled in
  features.py. Here we deliberately scan forward to build the target (Y).
"""
import numpy as np


class LabelGenerator:
    def __init__(self, df, tp_pct=1.0, sl_pct=0.20, max_bars=700,
                 use_atr=False, atr_col='ATR_14', atr_tp_mult=6.0, atr_sl_mult=2.0):
        """
        Parameters:
            df: DataFrame with OHLCV data (must contain 'Open','High','Low','Close')
            tp_pct: Take Profit distance as a fraction of entry (e.g. 1.0 = +100%)
            sl_pct: Stop Loss distance as a fraction of entry (e.g. 0.20 = -20%)
            max_bars: Vertical barrier — max bars to wait for a TP/SL touch.
                      Wide by default (700+) for extreme trend-following.
            use_atr: If True, size barriers dynamically from ATR instead of pct.
            atr_col: Column holding ATR values (used only when use_atr=True).
            atr_tp_mult: ATR multiplier for the Take Profit barrier.
            atr_sl_mult: ATR multiplier for the Stop Loss barrier.
        """
        self.df = df.copy()
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_bars = max_bars
        self.use_atr = use_atr
        self.atr_col = atr_col
        self.atr_tp_mult = atr_tp_mult
        self.atr_sl_mult = atr_sl_mult

    def _first_touch(self, highs, lows, start, end, up_level, dn_level):
        """
        Scan bars [start, end) and return which barrier is touched first.
        Returns 'up' if the upper level is touched before the lower level,
        'dn' if the lower is touched first, or None if neither is touched.
        A same-bar ambiguity (both touched) is resolved conservatively as 'dn'
        (assume the adverse level hit first).
        """
        for j in range(start, end):
            hit_up = highs[j] >= up_level
            hit_dn = lows[j] <= dn_level
            if hit_up and hit_dn:
                return 'dn'   # conservative: assume stop hit first
            if hit_up:
                return 'up'
            if hit_dn:
                return 'dn'
        return None

    def apply_triple_barrier(self):
        """
        Apply the Dynamic Triple-Barrier to every candle for both directions,
        writing two independent binary target columns:
          Label_Long  = 1 -> Long  TP hit before Long  SL (else 0)
          Label_Short = 1 -> Short TP hit before Short SL (else 0)
        The two are independent — a candle may be 1 in both.
        """
        tp = self.tp_pct
        sl = self.sl_pct
        max_b = self.max_bars

        if self.use_atr and self.atr_col in self.df.columns:
            print(f"Applying Dynamic Triple Barrier (ATR-scaled: TP={self.atr_tp_mult}xATR, "
                  f"SL={self.atr_sl_mult}xATR, Max_Bars={max_b})")
        else:
            print(f"Applying Dynamic Triple Barrier (TP={tp*100:.1f}%, SL={sl*100:.1f}%, Max_Bars={max_b})")

        opens = self.df['Open'].values
        highs = self.df['High'].values
        lows = self.df['Low'].values
        n = len(self.df)

        use_atr = self.use_atr and (self.atr_col in self.df.columns)
        atr_values = self.df[self.atr_col].values if use_atr else None

        label_long = np.zeros(n, dtype=int)
        label_short = np.zeros(n, dtype=int)
        valid_longs = 0
        valid_shorts = 0

        for i in range(n):
            # Enter at the Open of bar i (the unshifted execution price).
            entry = opens[i]
            if entry <= 0 or not np.isfinite(entry):
                continue

            # Barrier distances: dynamic (ATR) or fixed percentage.
            if use_atr:
                atr = atr_values[i]
                if not np.isfinite(atr) or atr <= 0:
                    continue
                tp_dist = atr * self.atr_tp_mult
                sl_dist = atr * self.atr_sl_mult
            else:
                tp_dist = entry * tp
                sl_dist = entry * sl

            end = min(i + 1 + max_b, n)
            if end <= i:
                continue

            # --- Long: TP above, SL below ---
            long_up = entry + tp_dist   # take profit
            long_dn = entry - sl_dist   # stop loss
            long_touch = self._first_touch(highs, lows, i, end, long_up, long_dn)

            # --- Short: TP below, SL above (inverted) ---
            short_dn = entry - tp_dist  # take profit
            short_up = entry + sl_dist  # stop loss
            short_touch = self._first_touch(highs, lows, i, end, short_up, short_dn)

            # Independent labels: a candle can be a valid Long AND a valid Short.
            if long_touch == 'up':
                label_long[i] = 1
                valid_longs += 1
            if short_touch == 'dn':
                label_short[i] = 1
                valid_shorts += 1

        self.df['Label_Long'] = label_long
        self.df['Label_Short'] = label_short
        total = max(1, n)
        print(f"Found {valid_longs} valid Longs and {valid_shorts} valid Shorts "
              f"(imbalance: Long={valid_longs/total*100:.2f}%, Short={valid_shorts/total*100:.2f}% of {n} candles).")
        return self.df

    def generate_labels(self):
        self.apply_triple_barrier()
        return self.df
