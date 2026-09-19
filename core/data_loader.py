"""
data_loader.py — Centralized Data Fetching (100% Automated, No Hardcoding)
"""
import os
import time
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from pandas_datareader import data as web
from dotenv import load_dotenv

load_dotenv()


class DataLoader:
    def __init__(self, csv_path, symbol):
        self.csv_path = csv_path
        self.symbol = symbol

    def load_base_data(self):
        print(f"[DataLoader] Loading base OHLCV from {self.csv_path} ...")
        df = pd.read_csv(self.csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        print(f"  -> {len(df)} rows, range {df['Date'].min()} ~ {df['Date'].max()}")
        return df

    def fetch_macro_data(self, start_date, end_date):
        print("[DataLoader] Fetching Macro data (yfinance + FRED) ...")
        tickers = {
            "DXY":    "DX-Y.NYB",
            "VIX":    "^VIX",
            "SP500":  "^GSPC",
            "BTC":    "BTC-USD",
            "USDJPY": "JPY=X",
            "GOLD":   "GC=F",
        }

        macro_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
        macro_df.index.name = 'Date'

        for name, ticker in tickers.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    macro_df[f"{name}_Close"] = data['Close'].squeeze()
                    print(f"  -> {name}: OK ({len(data)} rows)")
            except Exception as e:
                print(f"  -> {name}: FAILED ({e})")

        try:
            fedfunds = web.DataReader("FEDFUNDS", "fred", start_date, end_date)
            fedfunds.index = pd.to_datetime(fedfunds.index)
            macro_df = macro_df.join(fedfunds, how="outer")
            print(f"  -> FEDFUNDS: OK")
        except Exception as e:
            print(f"  -> FEDFUNDS: FAILED ({e})")

        macro_df = macro_df.ffill().fillna(0)
        return macro_df

    def fetch_fear_greed(self, start_date, end_date):
        print("[DataLoader] Fetching Fear & Greed Index (Alternative.me) ...")
        try:
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
            url = f"https://api.alternative.me/fng/?limit={days}&format=json"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:
                return None

            rows = [
                {"Date": pd.to_datetime(int(d["timestamp"]), unit="s"), "FearGreed": int(d["value"])}
                for d in data
            ]
            fg_df = pd.DataFrame(rows).sort_values("Date").set_index("Date")
            print(f"  -> Fear & Greed: OK ({len(fg_df)} rows)")
            return fg_df
        except Exception as e:
            print(f"  -> Fear & Greed: FAILED ({e})")
            return None

    def fetch_funding_rate(self, start_date=None, end_date=None):
        print(f"[DataLoader] Fetching Binance Funding Rate ({self.symbol}) ...")
        try:
            all_data = []
            start_ms = int(pd.to_datetime(start_date).timestamp() * 1000) if start_date else 0

            for _ in range(50):
                url = (
                    f"https://fapi.binance.com/fapi/v1/fundingRate"
                    f"?symbol={self.symbol}&limit=1000&startTime={start_ms}"
                )
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                all_data.extend(batch)
                start_ms = batch[-1]["fundingTime"] + 1
                time.sleep(0.15)
                if len(batch) < 1000:
                    break

            if not all_data:
                return None

            fr_df = pd.DataFrame(all_data)
            fr_df['Date'] = pd.to_datetime(fr_df['fundingTime'], unit='ms')
            fr_df['FundingRate'] = fr_df['fundingRate'].astype(float)
            fr_df = fr_df[['Date', 'FundingRate']].sort_values('Date')

            fr_daily = fr_df.set_index('Date').resample('1D').mean()
            print(f"  -> Funding Rate: OK ({len(fr_daily)} daily rows)")
            return fr_daily
        except Exception as e:
            print(f"  -> Funding Rate: FAILED ({e})")
            return None

    def get_full_data(self):
        df = self.load_base_data()
        start_date = df['Date'].min().strftime("%Y-%m-%d")
        end_date = df['Date'].max().strftime("%Y-%m-%d")

        macro_df = self.fetch_macro_data(start_date, end_date)
        if macro_df is not None:
            macro_df.index.name = 'Date'
            macro_reset = macro_df.reset_index()
            macro_reset['Date'] = pd.to_datetime(macro_reset['Date'])
            df = pd.merge_asof(df, macro_reset, on='Date', direction='backward')

        fg_df = self.fetch_fear_greed(start_date, end_date)
        if fg_df is not None:
            fg_reset = fg_df.reset_index()
            fg_reset['Date'] = pd.to_datetime(fg_reset['Date'])
            df = pd.merge_asof(df, fg_reset, on='Date', direction='backward')

        fr_df = self.fetch_funding_rate(start_date, end_date)
        if fr_df is not None:
            fr_reset = fr_df.reset_index()
            fr_reset['Date'] = pd.to_datetime(fr_reset['Date'])
            df = pd.merge_asof(df, fr_reset, on='Date', direction='backward')

        df = df.ffill().fillna(0)
        print(f"[DataLoader] Final dataset: {df.shape[0]} rows x {df.shape[1]} columns")
        return df
