"""
labels.py — Triple Barrier Labeling (Hindsight for best labels)

วิธีการทำงาน:
1. หา Swing Low โดยใช้ Hindsight (center=True) เพราะเราต้องการมาร์คเป้าหมาย (Y) ให้แม่นยำ
   * สิ่งที่ห้ามโกงคือ Features (X) ไม่ใช่ Labels
2. ใช้ Triple Barrier Method ตรวจสอบว่าชน TP หรือ SL ก่อนกัน
"""
import pandas as pd
import numpy as np


class LabelGenerator:
    def __init__(self, df, window=20, tp_pct=0.03, sl_pct=0.01, max_bars=40):
        self.df = df.copy()
        self.window = window
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_bars = max_bars

    def get_swing_lows(self):
        """
        หา Swing Low แบบ Hindsight (center=True):
        - มาร์คจุดที่ Low ต่ำที่สุดในหน้าต่าง 2*window+1 แท่ง
        - นี่คือจุดที่ 'ดีที่สุดในอดีต' ที่เราต้องการให้โมเดลทายให้ถูก
        """
        w = self.window
        full_window = 2 * w + 1

        self.df['Swing_Low'] = self.df['Low'] == self.df['Low'].rolling(window=full_window, center=True).min()
        return self.df

    def apply_triple_barrier(self):
        """
        Triple Barrier Method (Vectorized with NumPy for speed):
        """
        tp = self.tp_pct
        sl = self.sl_pct
        max_b = self.max_bars

        print(f"Applying Triple Barrier (TP={tp*100:.1f}%, SL={sl*100:.1f}%, Max_Bars={max_b}) on Swing Lows...")

        self.df['Signal'] = 0

        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values
        n = len(self.df)

        swing_low_indices = self.df[self.df['Swing_Low']].index.tolist()

        valid_count = 0
        for idx in swing_low_indices:
            entry_price = closes[idx]
            upper = entry_price * (1 + tp)
            lower = entry_price * (1 - sl)

            end = min(idx + 1 + max_b, n)

            for j in range(idx + 1, end):
                # Pessimistic: check SL first
                if lows[j] <= lower:
                    break  # Hit SL → invalid
                if highs[j] >= upper:
                    self.df.iat[idx, self.df.columns.get_loc('Signal')] = 1
                    valid_count += 1
                    break  # Hit TP first → valid!

        print(f"Found {valid_count} valid Safe-Buy signals out of {len(swing_low_indices)} raw swing lows.")
        return self.df

    def generate_labels(self):
        self.get_swing_lows()
        self.apply_triple_barrier()
        self.df.drop(columns=['Swing_Low'], inplace=True)
        return self.df
