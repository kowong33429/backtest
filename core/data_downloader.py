"""
data_downloader.py — Universal Binance Klines Downloader

Usage:
    python core/data_downloader.py --symbol ZECUSDT --interval 4h
"""
import requests
import pandas as pd
import time
import os
import argparse

def download_funding_rate(symbol, start_time):
    print(f"Downloading {symbol} Funding Rate from Binance Futures...")
    limit = 1000
    all_funding = []
    
    while True:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}&startTime={start_time}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_funding.extend(data)
            print(f"Fetched {len(data)} funding rates. Last date: {pd.to_datetime(data[-1]['fundingTime'], unit='ms')}")
            
            start_time = data[-1]['fundingTime'] + 1
            time.sleep(0.2)
            
            if len(data) < limit:
                break
        except Exception as e:
            print(f"Error fetching funding rate: {e}")
            break
            
    if all_funding:
        df_fund = pd.DataFrame(all_funding)
        df_fund['Date'] = pd.to_datetime(df_fund['fundingTime'], unit='ms')
        df_fund['Funding_Rate'] = df_fund['fundingRate'].astype(float)
        return df_fund[['Date', 'Funding_Rate']]
    return pd.DataFrame(columns=['Date', 'Funding_Rate'])

def download_data(symbol, interval):
    print(f"Downloading {symbol} {interval} data from Binance from the beginning...")
    limit = 1000
    start_time = 0
    all_klines = []

    while True:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&startTime={start_time}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_klines.extend(data)
            print(f"Fetched {len(data)} klines. Last date: {pd.to_datetime(data[-1][0], unit='ms')}")
            
            start_time = data[-1][6] + 1
            time.sleep(0.2)
            
            if len(data) < limit:
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    if all_klines:
        columns = [
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close Time', 'Quote Asset Volume', 'Number of Trades',
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
        ]
        
        df = pd.DataFrame(all_klines, columns=columns)
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df = df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.rename(columns={'Open Time': 'Date'}, inplace=True)
        
        # Download Funding Rate
        first_time_ms = all_klines[0][0]
        df_fund = download_funding_rate(symbol, first_time_ms)
        
        if not df_fund.empty:
            df = pd.merge_asof(df.sort_values('Date'), df_fund.sort_values('Date'), on='Date', direction='backward')
            df['Funding_Rate'] = df['Funding_Rate'].fillna(0) # Fill initial values with 0
        else:
            df['Funding_Rate'] = 0.0

        # Save to specific pair folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(base_dir, "data", symbol.lower())
        os.makedirs(save_dir, exist_ok=True)
        
        filename = os.path.join(save_dir, f"{symbol}_{interval}_full.csv")
        df.to_csv(filename, index=False)
        print(f"Success! Saved {len(df)} rows to {filename}")
    else:
        print("No data fetched.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Binance OHLCV data')
    parser.add_argument('--symbol', type=str, default='ZECUSDT', help='Trading pair symbol (e.g., ZECUSDT)')
    parser.add_argument('--interval', type=str, default='4h', help='Kline interval (e.g., 4h, 1d)')
    args = parser.parse_args()
    
    download_data(args.symbol.upper(), args.interval)
