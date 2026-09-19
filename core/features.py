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

    def add_mtf_context(self):
        print("Adding Multi-Timeframe Context (Position in Range)...")
        # 1 Day Range (Rolling 6 bars)
        daily_high = self.df['High'].rolling(window=6).max()
        daily_low = self.df['Low'].rolling(window=6).min()
        self.df['Pos_In_Day'] = (self.df['Close'] - daily_low) / (daily_high - daily_low + 1e-9)
        
        # 1 Week Range (Rolling 42 bars)
        weekly_high = self.df['High'].rolling(window=42).max()
        weekly_low = self.df['Low'].rolling(window=42).min()
        self.df['Pos_In_Week'] = (self.df['Close'] - weekly_low) / (weekly_high - weekly_low + 1e-9)
        
        return self.df

    def generate_all_features(self):
        self.add_technical_indicators()
        self.add_mtf_context()
        
        self.df = self.df.dropna().reset_index(drop=True)
        print(f"Feature engineering complete. Shape after dropping NaNs: {self.df.shape}")
        return self.df
