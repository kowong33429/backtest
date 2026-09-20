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
    def __init__(self, df, window=20, tp_pct=0.03, sl_pct=0.01, max_bars=300):
        self.df = df.copy()
        self.window = window
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_bars = max_bars

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

    def apply_triple_barrier(self):
        """
        Triple Barrier Method:
        - Long (Signal=1): จาก Swing Low ขึ้นไปชน TP ข้างบน
        - Short (Signal=2): จาก Swing High ลงไปชน TP ข้างล่าง
        """
        tp = self.tp_pct
        sl = self.sl_pct
        max_b = self.max_bars

        print(f"Applying Triple Barrier (TP={tp*100:.1f}%, SL={sl*100:.1f}%, Max_Bars={max_b})...")

        self.df['Signal'] = 0

        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values
        n = len(self.df)

        swing_low_indices = self.df[self.df['Swing_Low']].index.tolist()
        swing_high_indices = self.df[self.df['Swing_High']].index.tolist()

        valid_longs = 0
        valid_shorts = 0
        
        # 1. Process Longs
        for idx in swing_low_indices:
            entry_price = closes[idx]
            upper = entry_price * (1 + tp)
            lower = entry_price * (1 - sl)
            end = min(idx + 1 + max_b, n)

            for j in range(idx + 1, end):
                if lows[j] <= lower:
                    break  # Hit SL
                if highs[j] >= upper:
                    self.df.iat[idx, self.df.columns.get_loc('Signal')] = 1
                    valid_longs += 1
                    break  # Hit TP
                    
        # 2. Process Shorts
        for idx in swing_high_indices:
            # If already marked as Long, skip to avoid overlap conflicts
            if self.df.iat[idx, self.df.columns.get_loc('Signal')] == 1:
                continue
                
            entry_price = closes[idx]
            lower = entry_price * (1 - tp) # TP is below for short
            upper = entry_price * (1 + sl) # SL is above for short
            end = min(idx + 1 + max_b, n)

            for j in range(idx + 1, end):
                if highs[j] >= upper:
                    break  # Hit SL
                if lows[j] <= lower:
                    self.df.iat[idx, self.df.columns.get_loc('Signal')] = 2
                    valid_shorts += 1
                    break  # Hit TP

        print(f"Found {valid_longs} valid Longs and {valid_shorts} valid Shorts.")
        return self.df

    def generate_labels(self):
        self.get_swings()
        self.apply_triple_barrier()
        self.df.drop(columns=['Swing_Low', 'Swing_High'], inplace=True)
        return self.df
