"""
labels.py — Triple Barrier Labeling + Momentum Confirmation

วิธีการทำงาน:
1. หา Swing Low โดยใช้ Hindsight (center=True) เพราะเราต้องการมาร์คเป้าหมาย (Y) ให้แม่นยำ
   * สิ่งที่ห้ามโกงคือ Features (X) ไม่ใช่ Labels
2. ใช้ Triple Barrier Method ตรวจสอบว่าชน TP หรือ SL ก่อนกัน
3. ✨ Momentum Filter: จุด Swing ต้องมีโมเมนตัมยืนยันภายใน N แท่ง
   ถ้าไม่มีโมเมนตัม (กราฟ sideway หลังจุดเข้า) จะไม่นับเป็น Signal
"""
import pandas as pd
import numpy as np


class LabelGenerator:
    def __init__(self, df, window=20, tp_pct=0.03, sl_pct=0.01, max_bars=300,
                 momentum_bars=42, momentum_min_pct=0.03):
        """
        Parameters:
            df: DataFrame with OHLCV data
            window: Swing detection window (half-window each side)
            tp_pct: Take Profit percentage for Triple Barrier
            sl_pct: Stop Loss percentage for Triple Barrier
            max_bars: Maximum bars to wait for TP/SL hit
            momentum_bars: Number of bars to check for momentum confirmation
                           (42 = ~7 days on 4H timeframe)
            momentum_min_pct: Minimum price movement required within momentum_bars
                              (0.03 = 3% minimum move in the right direction)
        """
        self.df = df.copy()
        self.window = window
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_bars = max_bars
        self.momentum_bars = momentum_bars
        self.momentum_min_pct = momentum_min_pct

    def get_swings(self):
        """
        หา Swing Low / Swing High แบบ Hindsight (center=True):
        - นี่คือจุดที่ 'ดีที่สุดในอดีต' ที่เราต้องการให้โมเดลทายให้ถูก
        """
        w = self.window
        full_window = 2 * w + 1

        self.df['Swing_Low'] = self.df['Low'] == self.df['Low'].rolling(window=full_window, center=True).min()
        self.df['Swing_High'] = self.df['High'] == self.df['High'].rolling(window=full_window, center=True).max()
        return self.df

    def _check_momentum(self, idx, entry_price, direction):
        """
        Check if there is sufficient momentum after the swing point.

        For Long (direction='long'):
            Max High within momentum_bars must reach entry_price * (1 + momentum_min_pct)
        For Short (direction='short'):
            Min Low within momentum_bars must reach entry_price * (1 - momentum_min_pct)

        Returns True if momentum condition is met.
        """
        n = len(self.df)
        mom_end = min(idx + 1 + self.momentum_bars, n)

        if mom_end <= idx + 1:
            return False

        highs = self.df['High'].values
        lows = self.df['Low'].values

        if direction == 'long':
            # Price must move UP by at least momentum_min_pct within momentum_bars
            max_high = np.max(highs[idx + 1:mom_end])
            threshold = entry_price * (1 + self.momentum_min_pct)
            return max_high >= threshold
        else:
            # Price must move DOWN by at least momentum_min_pct within momentum_bars
            min_low = np.min(lows[idx + 1:mom_end])
            threshold = entry_price * (1 - self.momentum_min_pct)
            return min_low <= threshold

    def apply_triple_barrier(self):
        """
        Triple Barrier Method + Momentum Confirmation:
        - Long (Signal=1): จาก Swing Low ขึ้นไปชน TP ข้างบน + มี momentum ขึ้น
        - Short (Signal=2): จาก Swing High ลงไปชน TP ข้างล่าง + มี momentum ลง
        """
        tp = self.tp_pct
        sl = self.sl_pct
        max_b = self.max_bars
        mom_bars = self.momentum_bars
        mom_pct = self.momentum_min_pct

        print(f"Applying Triple Barrier (TP={tp*100:.1f}%, SL={sl*100:.1f}%, Max_Bars={max_b})")
        print(f"  + Momentum Filter (bars={mom_bars}, min_pct={mom_pct*100:.1f}%)")

        self.df['Signal'] = 0

        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values
        n = len(self.df)

        swing_low_indices = self.df[self.df['Swing_Low']].index.tolist()
        swing_high_indices = self.df[self.df['Swing_High']].index.tolist()

        valid_longs = 0
        valid_shorts = 0
        skipped_momentum_long = 0
        skipped_momentum_short = 0

        # 1. Process Longs
        for idx in swing_low_indices:
            entry_price = closes[idx]
            upper = entry_price * (1 + tp)
            lower = entry_price * (1 - sl)
            end = min(idx + 1 + max_b, n)

            # Check Triple Barrier first
            hit_tp = False
            for j in range(idx + 1, end):
                if lows[j] <= lower:
                    break  # Hit SL
                if highs[j] >= upper:
                    hit_tp = True
                    break  # Hit TP

            if not hit_tp:
                continue

            # Check Momentum Confirmation
            if not self._check_momentum(idx, entry_price, 'long'):
                skipped_momentum_long += 1
                continue

            self.df.iat[idx, self.df.columns.get_loc('Signal')] = 1
            valid_longs += 1

        # 2. Process Shorts
        for idx in swing_high_indices:
            # If already marked as Long, skip to avoid overlap conflicts
            if self.df.iat[idx, self.df.columns.get_loc('Signal')] == 1:
                continue

            entry_price = closes[idx]
            lower = entry_price * (1 - tp)  # TP is below for short
            upper = entry_price * (1 + sl)  # SL is above for short
            end = min(idx + 1 + max_b, n)

            # Check Triple Barrier first
            hit_tp = False
            for j in range(idx + 1, end):
                if highs[j] >= upper:
                    break  # Hit SL
                if lows[j] <= lower:
                    hit_tp = True
                    break  # Hit TP

            if not hit_tp:
                continue

            # Check Momentum Confirmation
            if not self._check_momentum(idx, entry_price, 'short'):
                skipped_momentum_short += 1
                continue

            self.df.iat[idx, self.df.columns.get_loc('Signal')] = 2
            valid_shorts += 1

        print(f"Found {valid_longs} valid Longs and {valid_shorts} valid Shorts.")
        print(f"  Skipped (no momentum): Long={skipped_momentum_long}, Short={skipped_momentum_short}")
        return self.df

    def generate_labels(self):
        self.get_swings()
        self.apply_triple_barrier()
        self.df.drop(columns=['Swing_Low', 'Swing_High'], inplace=True)
        return self.df
