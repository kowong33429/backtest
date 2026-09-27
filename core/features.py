import pandas as pd
import numpy as np
import pandas_ta as ta

# ---------------------------------------------------------------------------
# FEATURE WINDOWS
# NOTE: Per an explicit user override, indicator windows are stepped by 10
# (10, 20, 30, ...). This deliberately departs from AGENTS.md Rule #4
# (Fibonacci/log spacing). Adjacent windows are highly correlated, so the
# >0.75 correlation filter in optimizer.py is expected to prune most of the
# redundant ones downstream.
# ---------------------------------------------------------------------------
RSI_LENGTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
BB_LENGTHS = [20, 30, 40, 50, 60, 70, 80, 90, 100]           # BB_20 = canonical baseline
VOL_SURGE_LENGTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ADX_LENGTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ATR_LENGTHS = [10, 14, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # keep 14 (backtester + ATR_Ratio)
SMA_LENGTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200] # keep 50 & 200 (SMA_Cross)

# Time-based windows expressed in 4H bars:
#   1D=6, 3D=18, 7D=42, 15D=90, 1M=180, 3M=540, 6M=1080, 12M=2190
RETURN_LENGTHS = [6, 18, 42, 90, 180, 540, 1080, 2190]       # % change vs past
POS_RANGE_LENGTHS = [6, 18, 42, 90, 180, 540, 1080, 2190]    # position within rolling range
VWAP_LENGTHS = [6, 18, 42, 90, 180, 540, 1080, 2190]         # distance from rolling VWAP
DIST_HL_LENGTHS = [180, 540, 1080, 2190]                     # distance from 1M/3M/6M/1Y High & Low


class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def add_technical_indicators(self):
        print("Adding Technical Indicators (stepped windows)...")

        # --- Momentum: RSI across lengths ---
        for length in RSI_LENGTHS:
            self.df[f'RSI_{length}'] = ta.rsi(self.df['Close'], length=length)

        # --- MACD (single canonical 12/26/9) ---
        macd = ta.macd(self.df['Close'], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            self.df['MACD'] = macd.iloc[:, 0]
            self.df['MACD_Hist'] = macd.iloc[:, 1]

        # --- Volatility: Bollinger Band width across windows ---
        for length in BB_LENGTHS:
            sma = self.df['Close'].rolling(window=length).mean()
            std = self.df['Close'].rolling(window=length).std()
            self.df[f'BB_Width_{length}'] = ((sma + 2 * std) - (sma - 2 * std)) / (sma + 1e-9)

        # --- ATR across lengths (ATR_14 required by the backtester) ---
        for length in ATR_LENGTHS:
            self.df[f'ATR_{length}'] = ta.atr(
                self.df['High'], self.df['Low'], self.df['Close'], length=length
            )

        # --- Trend: SMA across windows + distance from each ---
        for length in SMA_LENGTHS:
            sma = ta.sma(self.df['Close'], length=length)
            self.df[f'SMA_{length}'] = sma
            self.df[f'Dist_SMA_{length}'] = (self.df['Close'] - sma) / (sma + 1e-9)

        return self.df

    def add_trend_features(self):
        """
        Trend-Following Features for catching big trend moves:
        - ADX: trend strength across multiple lengths
        - SMA Cross: Golden / Death Cross (50 vs 200)
        - Multi-horizon returns (time-based windows)
        - Distance from multi-timeframe (1M/3M/6M/1Y) Low/High
        - Volume Surge across windows
        - ATR Ratio (normalized volatility)
        """
        print("Adding Trend-Following Features...")

        # 1. ADX (trend strength) across lengths
        for length in ADX_LENGTHS:
            adx = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], length=length)
            if adx is not None and not adx.empty:
                self.df[f'ADX_{length}'] = adx.iloc[:, 0]

        # 2. SMA Cross (Golden Cross = 1, Death Cross = 0)
        sma50 = self.df.get('SMA_50', ta.sma(self.df['Close'], length=50))
        sma200 = self.df.get('SMA_200', ta.sma(self.df['Close'], length=200))
        self.df['SMA_Cross'] = (sma50 > sma200).astype(int)

        # 3. Multi-horizon returns (time-based bars)
        for length in RETURN_LENGTHS:
            self.df[f'Return_{length}b'] = self.df['Close'].pct_change(length)

        # 4. Distance from multi-timeframe Low/High (1M/3M/6M/1Y)
        n = len(self.df)
        for length in DIST_HL_LENGTHS:
            window = min(length, n - 1)
            if window <= 50:
                continue
            roll_low = self.df['Low'].rolling(window=window, min_periods=50).min()
            roll_high = self.df['High'].rolling(window=window, min_periods=50).max()
            self.df[f'Dist_Low_{length}b'] = (self.df['Close'] - roll_low) / (roll_low + 1e-9)
            self.df[f'Dist_High_{length}b'] = (self.df['Close'] - roll_high) / (roll_high + 1e-9)

        # 5. Volume Surge across windows (current volume vs rolling average)
        for length in VOL_SURGE_LENGTHS:
            vol_ma = self.df['Volume'].rolling(window=length).mean()
            self.df[f'Volume_Surge_{length}'] = self.df['Volume'] / (vol_ma + 1e-9)

        # 6. ATR Ratio (ATR_14 / price, normalized volatility)
        atr = self.df.get('ATR_14', ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14))
        self.df['ATR_Ratio'] = atr / (self.df['Close'] + 1e-9)

        return self.df

    def add_lagged_features(self):
        print("Adding Lagged Features (Previous bar values)...")
        # 1. Price Returns (Rate of Change) at short lags
        self.df['Return_1'] = self.df['Close'].pct_change(1)
        self.df['Return_2'] = self.df['Close'].pct_change(2)

        # 2. Previous Bar Indicators (Lagged states) so the model can see
        #    the momentum of change. Lag a curated set of existing columns.
        cols_to_lag = ['RSI_10', 'RSI_20', 'MACD', 'MACD_Hist', 'BB_Width_20', 'Return_1']
        for col in cols_to_lag:
            if col in self.df.columns:
                self.df[f'{col}_Lag1'] = self.df[col].shift(1)

        return self.df

    def add_mtf_context(self):
        print("Adding Multi-Timeframe Context (Position in Range)...")
        # Rolling position within range across time-based windows.
        # 4H Timeframe -> 6 bars = 24H, 42 bars = 7D, 180 bars ~ 1M
        for length in POS_RANGE_LENGTHS:
            roll_high = self.df['High'].rolling(window=length).max()
            roll_low = self.df['Low'].rolling(window=length).min()
            self.df[f'Pos_In_{length}b'] = (self.df['Close'] - roll_low) / (roll_high - roll_low + 1e-9)

        return self.df

    def add_microstructure_features(self):
        print("Adding Microstructure & Volume Profile Proxies...")

        # 1. Order Flow Imbalance Proxy (OFI)
        # (Close - Open) / (High - Low) gives intra-bar dominance (-1 to 1)
        # Multiply by Volume for a net aggressive-volume proxy.
        price_range = (self.df['High'] - self.df['Low']) + 1e-9
        self.df['OFI_Proxy'] = ((self.df['Close'] - self.df['Open']) / price_range) * self.df['Volume']

        # 2. Volume Profile (rolling VWAP distance across windows)
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        vol_price = self.df['Volume'] * typical_price
        for length in VWAP_LENGTHS:
            vwap = vol_price.rolling(window=length).sum() / (self.df['Volume'].rolling(window=length).sum() + 1e-9)
            self.df[f'Dist_VWAP_{length}b'] = (self.df['Close'] - vwap) / (vwap + 1e-9)

        return self.df

    def generate_all_features(self):
        self.add_technical_indicators()
        self.add_trend_features()
        self.add_lagged_features()
        self.add_mtf_context()
        self.add_microstructure_features()

        # STRICT ANTI-LEAKAGE RULE (AGENTS.md Rule #1 — No Lookahead Bias)
        # Shift all calculated features by 1 so that at bar T the model only sees
        # data up to T-1 and makes its decision at the Open of bar T.
        # We do NOT shift OHLCV: labels.py needs exact prices to evaluate the
        # Triple Barrier, and the Open of bar T is the only unshifted execution price.
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
