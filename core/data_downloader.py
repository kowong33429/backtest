"""
data_downloader.py — Universal Binance Klines Downloader (with Incremental Update)

Usage:
    python core/data_downloader.py --symbol ZECUSDT --interval 4h
    
ครั้งแรก: ดาวน์โหลดตั้งแต่เริ่มต้น
ครั้งถัดไป: อ่าน CSV เดิม แล้วดาวน์โหลดต่อจากแท่งล่าสุด -> append เข้าไฟล์เดิม
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


def get_existing_data(filename):
    """Load existing CSV and return the last timestamp (ms) for incremental download."""
    if not os.path.exists(filename):
        return None, 0
    
    try:
        df_existing = pd.read_csv(filename)
        df_existing['Date'] = pd.to_datetime(df_existing['Date'])
        last_date = df_existing['Date'].max()
        # Convert to ms timestamp, add 1 interval to avoid duplicate
        last_ms = int(last_date.timestamp() * 1000) + 1
        print(f"  Existing data found: {len(df_existing)} rows, last date: {last_date}")
        return df_existing, last_ms
    except Exception as e:
        print(f"  Error reading existing file ({e}), re-downloading from scratch...")
        return None, 0


def download_data(symbol, interval):
    # Determine save path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "data", symbol.lower())
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{symbol}_{interval}_full.csv")

    # Check for existing data (incremental update)
    df_existing, start_time = get_existing_data(filename)
    
    if df_existing is not None:
        print(f"[Update Mode] Downloading new data for {symbol} {interval} from last date...")
    else:
        print(f"[Full Download] Downloading {symbol} {interval} data from the beginning...")
        start_time = 0

    limit = 1000
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

    if not all_klines and df_existing is not None:
        print(f"Already up to date! No new data for {symbol}.")
        return

    if all_klines:
        columns = [
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close Time', 'Quote Asset Volume', 'Number of Trades',
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
        ]
        
        df_new = pd.DataFrame(all_klines, columns=columns)
        df_new['Open Time'] = pd.to_datetime(df_new['Open Time'], unit='ms')
        df_new = df_new[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df_new.rename(columns={'Open Time': 'Date'}, inplace=True)
        
        # Cast numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

        # Merge with existing data (if updating)
        if df_existing is not None:
            # Append new rows, remove potential overlap by Date
            df = pd.concat([df_existing.drop(columns=['Funding_Rate'], errors='ignore'), df_new], ignore_index=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.drop_duplicates(subset='Date', keep='last').sort_values('Date').reset_index(drop=True)
            new_rows = len(df) - len(df_existing)
            print(f"  Appended {new_rows} new rows (total: {len(df)})")
        else:
            df = df_new

        # Download Funding Rate (only for new period)
        if df_existing is not None:
            # Only fetch funding from last existing date
            fund_start = int(df_existing['Date'].max().timestamp() * 1000)
        else:
            fund_start = all_klines[0][0]
        
        df_fund = download_funding_rate(symbol, fund_start)
        
        if not df_fund.empty:
            # If updating, we need to re-merge funding for the whole dataset
            df = df.drop(columns=['Funding_Rate'], errors='ignore')
            df = pd.merge_asof(df.sort_values('Date'), df_fund.sort_values('Date'), on='Date', direction='backward')
            df['Funding_Rate'] = df['Funding_Rate'].fillna(0)
        elif 'Funding_Rate' not in df.columns:
            df['Funding_Rate'] = 0.0

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
