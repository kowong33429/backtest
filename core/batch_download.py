"""
batch_download.py — Download 4H OHLCV data from Binance Futures

Usage:
    python core/batch_download.py              # Download ALL Binance Futures coins
    python core/batch_download.py --priority   # Download 10 priority coins only
    python core/batch_download.py --symbol DASHUSDT  # Download a single coin
"""
import subprocess
import sys
import os
import argparse
import requests

# Priority coins for backtesting (from trend scanner analysis)
# This will be dynamically loaded from data/scan_results/scanned_priority_coins.json
PRIORITY_COINS = []


def get_all_binance_futures_symbols():
    """Fetch all active USDT perpetual symbols from Binance Futures."""
    print("[Binance] Fetching all futures symbols...")
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=15)
        resp.raise_for_status()
        symbols = sorted([
            s['symbol'] for s in resp.json().get('symbols', [])
            if s['quoteAsset'] == 'USDT'
            and s['contractType'] == 'PERPETUAL'
            and s['status'] == 'TRADING'
        ])
        print(f"  -> Found {len(symbols)} active USDT perpetual pairs")
        return symbols
    except Exception as e:
        print(f"  -> ERROR: {e}")
        return []


def get_already_downloaded(base_dir):
    """Check which symbols already have data downloaded."""
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(data_dir):
        return set()
    downloaded = set()
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            # Check if CSV exists
            csv_file = os.path.join(folder_path, f"{folder.upper()}_4h_full.csv")
            if os.path.exists(csv_file):
                downloaded.add(folder.upper() + 'USDT' if not folder.upper().endswith('USDT') else folder.upper())
    return downloaded


def main():
    parser = argparse.ArgumentParser(description='Batch download 4H OHLCV data')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Download a single symbol (e.g., DASHUSDT)')
    parser.add_argument('--priority', action='store_true',
                        help='Download only 10 priority coins')
    args = parser.parse_args()

    core_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(core_dir)
    downloader = os.path.join(core_dir, 'data_downloader.py')

    # Determine symbol list
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.priority:
        import json
        priority_file = os.path.join(base_dir, 'data', 'scan_results', 'scanned_priority_coins.json')
        if os.path.exists(priority_file):
            with open(priority_file, 'r', encoding='utf-8') as f:
                symbols = json.load(f)
            print(f"Loaded {len(symbols)} priority coins from trend scanner.")
        else:
            print("Priority file not found! Please run 'python core/trend_scanner.py' first.")
            return
    else:
        symbols = get_all_binance_futures_symbols()
        if not symbols:
            print("Failed to fetch symbols. Exiting.")
            return

    print(f"{'=' * 60}")
    print(f"  Batch Download: {len(symbols)} coins (4H timeframe)")
    print(f"{'=' * 60}\n")

    success = 0
    failed = 0
    skipped = 0

    downloaded = get_already_downloaded(base_dir)
    print(f"  Found {len(downloaded)} already downloaded coins.")

    for i, sym in enumerate(symbols, 1):
        if sym in downloaded:
            print(f"[{i}/{len(symbols)}] Skipping {sym} (Already downloaded)")
            skipped += 1
            continue
            
        print(f"[{i}/{len(symbols)}] Downloading {sym}...")
        result = subprocess.run(
            [sys.executable, downloader, '--symbol', sym, '--interval', '4h'],
            capture_output=True, text=True, encoding='utf-8'
        )
        if result.returncode == 0:
            # Extract the last line showing success
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'Success' in line or 'rows' in line.lower():
                    print(f"  -> {line.strip()}")
                    break
            success += 1
        else:
            print(f"  -> ERROR: {result.stderr.strip()[:200]}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Download complete! Success: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
