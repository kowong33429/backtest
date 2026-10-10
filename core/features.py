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

# Time-based windows expressed in CALENDAR DAYS, then converted to bars at
# runtime via `bars_per_day` (so the same feature spans the same real time on
# any timeframe):
#   1D, 3D, 7D, 15D, 1M, 3M, 6M, 1Y
# On the 4H timeframe (bars_per_day=6) these reproduce the original bar windows
# EXACTLY: [6, 18, 42, 90, 180, 540, 1080, 2190] (and HL: [180, 540, 1080, 2190]).
# On the 1D timeframe (bars_per_day=1) they become the daily-native windows
# [1, 3, 7, 15, 30, 90, 180, 365] (and HL: [30, 90, 180, 365]).
TIME_WINDOWS_DAYS = [1, 3, 7, 15, 30, 90, 180, 365]  # returns / pos-in-range / VWAP distance
DIST_HL_DAYS = [30, 90, 180, 365]                    # distance from 1M/3M/6M/1Y High & Low
DONCHIAN_LENGTHS = [20, 40, 60, 120]                 # explicit Donchian channel features


class FeatureEngineer:
    def __init__(self, df, bars_per_day=6):
        """
        bars_per_day : bars per calendar day of the input series. Default 6 is the
        4H timeframe (6 bars/day) and reproduces the original windows unchanged;
        pass 1 for daily bars. Indicator lengths (RSI/BB/ADX/ATR/SMA/Volume) are
        kept as-is across timeframes (daily-native); only the time-based windows
        (returns, position-in-range, VWAP distance, Hi/Lo distance) scale with it.
        """
        self.df = df.copy()
        self.bars_per_day = max(1, int(bars_per_day))
        self.return_lengths = [d * self.bars_per_day for d in TIME_WINDOWS_DAYS]
        self.pos_range_lengths = [d * self.bars_per_day for d in TIME_WINDOWS_DAYS]
        self.vwap_lengths = [d * self.bars_per_day for d in TIME_WINDOWS_DAYS]
        self.dist_hl_lengths = [d * self.bars_per_day for d in DIST_HL_DAYS]

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
        for length in self.return_lengths:
            self.df[f'Return_{length}b'] = self.df['Close'].pct_change(length)

        # 4. Distance from multi-timeframe Low/High (1M/3M/6M/1Y)
        n = len(self.df)
        for length in self.dist_hl_lengths:
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
        # Rolling position within range across time-based windows (scaled by
        # bars_per_day: e.g. on 4H 6 bars = 24H, 42 = 7D, 180 ~ 1M).
        for length in self.pos_range_lengths:
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
        for length in self.vwap_lengths:
            vwap = vol_price.rolling(window=length).sum() / (self.df['Volume'].rolling(window=length).sum() + 1e-9)
            self.df[f'Dist_VWAP_{length}b'] = (self.df['Close'] - vwap) / (vwap + 1e-9)

        return self.df

    def add_donchian_features(self):
        print("Adding Explicit Donchian Channel Features...")
        # Add exact Donchian filter metrics the short labeler uses, so the model
        # can natively learn to avoid "price above midline" or "giant candle".
        for length in DONCHIAN_LENGTHS:
            hh = self.df['High'].rolling(window=length, min_periods=length).max()
            ll = self.df['Low'].rolling(window=length, min_periods=length).min()
            mid = (hh + ll) / 2.0
            
            # 1. Distance to Midline (positive = above mid, negative = below mid)
            self.df[f'Dist_Donchian_Mid_{length}'] = (self.df['Close'] - mid) / (mid + 1e-9)
            
            # 2. Channel Width (normalized by price)
            chan_width = hh - ll
            self.df[f'Donchian_Width_{length}'] = chan_width / (self.df['Close'] + 1e-9)
            
            # 3. Bar Range vs Channel Width Fraction
            bar_range = self.df['High'] - self.df['Low']
            frac = np.where(chan_width > 0, bar_range / chan_width, 1.0)
            self.df[f'Bar_Donchian_Frac_{length}'] = pd.Series(frac, index=self.df.index)

        return self.df

    def generate_all_features(self):
        self.add_technical_indicators()
        self.add_trend_features()
        self.add_lagged_features()
        self.add_mtf_context()
        self.add_microstructure_features()
        self.add_donchian_features()

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
