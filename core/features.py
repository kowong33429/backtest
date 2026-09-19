import pandas as pd
import numpy as np
import pandas_ta as ta

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        
    def add_technical_indicators(self):
        print("Adding Technical Indicators (pandas_ta)...")
        # Momentum & Oscillators
        self.df['RSI_14'] = ta.rsi(self.df['Close'], length=14)
        
        # MACD
        macd = ta.macd(self.df['Close'], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            # Safely grab the first two columns regardless of exact naming
            self.df['MACD'] = macd.iloc[:, 0]
            self.df['MACD_Hist'] = macd.iloc[:, 1]
        
        # Volatility (Manual Bollinger Bands to avoid pandas_ta naming issues)
        sma_20 = self.df['Close'].rolling(window=20).mean()
        std_20 = self.df['Close'].rolling(window=20).std()
        upper_bb = sma_20 + (2 * std_20)
        lower_bb = sma_20 - (2 * std_20)
        self.df['BB_Width'] = (upper_bb - lower_bb) / sma_20
        
        self.df['ATR_14'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)
        
        # Trend
        self.df['SMA_50'] = ta.sma(self.df['Close'], length=50)
        self.df['SMA_200'] = ta.sma(self.df['Close'], length=200)
        
        # Distance from moving averages
        self.df['Dist_SMA_50'] = (self.df['Close'] - self.df['SMA_50']) / (self.df['SMA_50'] + 1e-9)
        
        return self.df

    def add_lagged_features(self):
        print("Adding Lagged Features (Previous bar values)...")
        # 1. Price Returns (Rate of Change)
        self.df['Return_1'] = self.df['Close'].pct_change(1)
        self.df['Return_2'] = self.df['Close'].pct_change(2)
        
        # 2. Previous Bar Indicators (Lagged states)
        # ให้โมเดลเห็นข้อมูลของ "แท่งก่อนหน้า" เพื่อดู Momentum การเปลี่ยนแปลง
        cols_to_lag = ['RSI_14', 'MACD', 'MACD_Hist', 'BB_Width', 'Return_1']
        for col in cols_to_lag:
            if col in self.df.columns:
                self.df[f'{col}_Lag1'] = self.df[col].shift(1)
                
        return self.df

    def add_mtf_context(self):
        print("Adding Multi-Timeframe Context (Position in Range)...")
        # การใช้ Rolling คือการดึงย้อนหลัง 24 ชั่วโมง และ 7 วัน ตลอดเวลา (ไม่มีการ Reset ตอนเที่ยงคืน)
        # 4H Timeframe -> 6 bars = 24 Hours (Rolling 1 Day)
        rolling_24h_high = self.df['High'].rolling(window=6).max()
        rolling_24h_low = self.df['Low'].rolling(window=6).min()
        self.df['Pos_In_24H'] = (self.df['Close'] - rolling_24h_low) / (rolling_24h_high - rolling_24h_low + 1e-9)
        
        # 4H Timeframe -> 42 bars = 7 Days (Rolling 1 Week)
        rolling_7d_high = self.df['High'].rolling(window=42).max()
        rolling_7d_low = self.df['Low'].rolling(window=42).min()
        self.df['Pos_In_7D'] = (self.df['Close'] - rolling_7d_low) / (rolling_7d_high - rolling_7d_low + 1e-9)
        
        return self.df

    def add_microstructure_features(self):
        print("Adding Microstructure & Volume Profile Proxies...")
        
        # 1. Order Flow Imbalance Proxy (OFI)
        # (Close - Open) / (High - Low) gives the intra-bar price action dominance (-1 to 1)
        # Multiply by Volume to get the net aggressive volume proxy
        price_range = (self.df['High'] - self.df['Low']) + 1e-9
        self.df['OFI_Proxy'] = ((self.df['Close'] - self.df['Open']) / price_range) * self.df['Volume']
        
        # 2. Volume Profile (Rolling 24H VWAP distance)
        # 4H interval -> 6 bars = 24H
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        vol_price = self.df['Volume'] * typical_price
        vwap_24h = vol_price.rolling(window=6).sum() / (self.df['Volume'].rolling(window=6).sum() + 1e-9)
        self.df['Dist_VWAP_24H'] = (self.df['Close'] - vwap_24h) / (vwap_24h + 1e-9)
        
        return self.df

    def generate_all_features(self):
        self.add_technical_indicators()
        self.add_lagged_features()
        self.add_mtf_context()
        self.add_microstructure_features()
        
        # STRICT ANTI-LEAKAGE RULE (No Lookahead Bias)
        # Shift all calculated features by 1 so that at bar T, the model only sees data up to T-1
        # We don't shift OHLCV because labels.py needs exact prices to calculate Swing Low/High and TP/SL
        print("Applying STRICT Anti-Leakage Shift (.shift(1)) to all features...")
        base_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Funding_Rate']
        feature_cols = [c for c in self.df.columns if c not in base_cols]
        
        # Shift features by 1 bar
        self.df[feature_cols] = self.df[feature_cols].shift(1)
        
        self.df = self.df.dropna().reset_index(drop=True)
        print(f"Feature engineering complete. Shape after dropping NaNs: {self.df.shape}")
        
        # Verification Step: Ensure no future leakage (No NaNs in the middle)
        assert not self.df.isnull().any().any(), "Data Leakage Check Failed: Found NaNs in the dataset after dropna!"
        
        return self.df
