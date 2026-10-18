"""
trend_scanner.py — Multi-Exchange Big Trend Scanner
สแกนเหรียญจาก Binance Futures + OKX Swap หา Pattern "Big Rally" (>200% ใน 14-120 วัน)

Usage:
    python core/trend_scanner.py                        # สแกนทั้งหมด ทุกปี
    python core/trend_scanner.py --min-gain 500         # เฉพาะ >500%
    python core/trend_scanner.py --year 2025            # เฉพาะปี 2025
    python core/trend_scanner.py --slow-only            # เฉพาะ ZEC-like pattern (7d momentum < 50%)
    python core/trend_scanner.py --multi-year           # เฉพาะเหรียญที่ rally ซ้ำหลายปี

Output:
    data/scan_results/all_rallies_YYYYMMDD.csv
    data/scan_results/zec_like_YYYYMMDD.csv
    data/scan_results/multi_year_YYYYMMDD.csv
    data/scan_results/by_category_YYYYMMDD.csv
"""
import os
import sys
import time
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================
# COIN CATEGORIES (ใส่ไว้ในโค้ดเลยเพื่อไม่ต้องพึ่ง external file)
# ============================================================
COIN_CATEGORIES = {
    # Privacy
    'ZEC': 'Privacy', 'DASH': 'Privacy', 'XMR': 'Privacy', 'SCRT': 'Privacy',
    'ROSE': 'Privacy', 'BEAM': 'Privacy', 'NIL': 'Privacy',
    # Layer 1
    'BTC': 'L1-Major', 'ETH': 'L1-Major', 'SOL': 'L1', 'ADA': 'L1',
    'AVAX': 'L1', 'DOT': 'L1', 'ATOM': 'L1', 'NEAR': 'L1',
    'FTM': 'L1', 'ALGO': 'L1', 'HBAR': 'L1', 'ICP': 'L1',
    'EOS': 'L1', 'NEO': 'L1', 'XTZ': 'L1', 'EGLD': 'L1',
    'FLOW': 'L1', 'MINA': 'L1', 'ONE': 'L1', 'CELO': 'L1',
    'KAVA': 'L1', 'SUI': 'L1', 'SEI': 'L1', 'TIA': 'L1',
    'APT': 'L1', 'TON': 'L1', 'STRK': 'L1', 'TRX': 'L1',
    'VET': 'L1', 'IOTA': 'L1', 'XLM': 'L1', 'XRP': 'L1',
    'QTUM': 'L1', 'ZIL': 'L1', 'ICX': 'L1', 'WAVES': 'L1',
    'ONT': 'L1', 'THETA': 'L1', 'STX': 'L1-BTC', 'KAS': 'L1',
    'INJ': 'L1', 'CFX': 'L1', 'CKB': 'L1', 'IOST': 'L1',
    'BERA': 'L1', 'INIT': 'L1', 'FORM': 'L1', 'MOVE': 'L1',
    'DYM': 'L1', 'OMNI': 'L1', 'NTRN': 'L1', 'LAYER': 'L1',
    # Layer 2
    'MATIC': 'L2', 'POL': 'L2', 'OP': 'L2', 'ARB': 'L2',
    'IMX': 'L2', 'LRC': 'L2', 'METIS': 'L2', 'MANTA': 'L2',
    'ZK': 'L2', 'ALT': 'L2', 'SCR': 'L2', 'BLAST': 'L2',
    # DeFi
    'UNI': 'DeFi', 'AAVE': 'DeFi', 'COMP': 'DeFi', 'MKR': 'DeFi',
    'SNX': 'DeFi', 'YFI': 'DeFi', 'SUSHI': 'DeFi', 'CRV': 'DeFi',
    'BAL': 'DeFi', '1INCH': 'DeFi', 'DYDX': 'DeFi', 'GMX': 'DeFi',
    'PENDLE': 'DeFi', 'JUP': 'DeFi', 'ENA': 'DeFi', 'ONDO': 'DeFi',
    'RUNE': 'DeFi', 'CAKE': 'DeFi', 'ZRX': 'DeFi', 'HYPE': 'DeFi',
    'AEVO': 'DeFi', 'BANANA': 'DeFi', 'CETUS': 'DeFi', 'COW': 'DeFi',
    'ORCA': 'DeFi', 'BANK': 'DeFi', 'SPELL': 'DeFi', 'REEF': 'DeFi',
    'PERP': 'DeFi', 'ALPHA': 'DeFi', 'LINA': 'DeFi',
    # AI
    'FET': 'AI', 'OCEAN': 'AI', 'RNDR': 'AI', 'RENDER': 'AI',
    'AGIX': 'AI', 'WLD': 'AI', 'TAO': 'AI', 'ARKM': 'AI',
    'IO': 'AI', 'KAITO': 'AI', 'SHELL': 'AI', 'PARTI': 'AI',
    'VANA': 'AI', 'SLF': 'AI', 'TURBO': 'AI-Meme',
    # Gaming
    'AXS': 'Gaming', 'SAND': 'Gaming', 'MANA': 'Gaming',
    'GALA': 'Gaming', 'ENJ': 'Gaming', 'ILV': 'Gaming',
    'YGG': 'Gaming', 'PIXEL': 'Gaming', 'RONIN': 'Gaming',
    'SUPER': 'Gaming', 'PRIME': 'Gaming', 'MAGIC': 'Gaming',
    'ACE': 'Gaming', 'PORTAL': 'Gaming', 'GUNZ': 'Gaming',
    # Meme
    'DOGE': 'Meme', 'SHIB': 'Meme', 'PEPE': 'Meme', 'FLOKI': 'Meme',
    'BONK': 'Meme', 'WIF': 'Meme', 'BOME': 'Meme', 'MEW': 'Meme',
    'NEIRO': 'Meme', 'POPCAT': 'Meme', 'DOGS': 'Meme', 'NOT': 'Meme',
    'PEOPLE': 'Meme', 'HMSTR': 'Meme', 'TRUMP': 'Meme', 'ANIME': 'Meme',
    # Storage
    'FIL': 'Storage', 'AR': 'Storage', 'STORJ': 'Storage',
    # LSD
    'LDO': 'LSD', 'RPL': 'LSD', 'SSV': 'LSD', 'ETHFI': 'LSD',
    'EIGEN': 'LSD', 'HAEDAL': 'LSD', 'KERNEL': 'LSD',
    'LISTA': 'LSD', 'REZ': 'LSD',
    # Oracle / Cross-chain
    'LINK': 'Oracle', 'BAND': 'Oracle', 'API3': 'Oracle', 'PYTH': 'Oracle',
    'W': 'Cross-chain', 'ZRO': 'Cross-chain',
    # Infra
    'GRT': 'Infra', 'ANKR': 'Infra', 'RLC': 'Infra', 'SKL': 'Infra',
    'CELR': 'Infra', 'COTI': 'Infra', 'ACH': 'Infra', 'ENS': 'Infra',
    'GPS': 'Infra', 'SIGN': 'Infra', 'IP': 'Infra', 'BB': 'Infra',
    # Exchange
    'BNB': 'Exchange', 'OKB': 'Exchange', 'CRO': 'Exchange',
    # Classic
    'LTC': 'Classic', 'BCH': 'Classic', 'ETC': 'Classic',
    # Others
    'BAT': 'Utility', 'CHZ': 'Fan-Token', 'BLUR': 'NFT',
    'APE': 'NFT', 'ME': 'NFT', 'MASK': 'Social', 'CYBER': 'Social',
    'RSR': 'Stablecoin', 'USUAL': 'Stablecoin', 'JASMY': 'IoT',
    'SXP': 'Payment', 'ORDI': 'BRC20', 'BIO': 'DeSci',
}


def get_category(symbol_usdt):
    """Extract base asset and return category."""
    base = symbol_usdt.replace('USDT', '').replace('1000', '').replace('1M', '')
    return COIN_CATEGORIES.get(base, 'Unknown'), base


# ============================================================
# DATA FETCHING
# ============================================================

def get_binance_futures_symbols():
    """Get all active USDT perpetual symbols from Binance Futures."""
    print("[Binance] Fetching futures symbols...")
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=15)
        resp.raise_for_status()
        symbols = [
            s['symbol'] for s in resp.json().get('symbols', [])
            if s['quoteAsset'] == 'USDT'
            and s['contractType'] == 'PERPETUAL'
            and s['status'] == 'TRADING'
        ]
        print(f"  -> {len(symbols)} active USDT perpetual pairs")
        return symbols
    except Exception as e:
        print(f"  -> ERROR: {e}")
        return []


def get_okx_futures_symbols():
    """Get all active USDT swap symbols from OKX."""
    print("[OKX] Fetching swap symbols...")
    try:
        resp = requests.get(
            "https://www.okx.com/api/v5/public/instruments?instType=SWAP", timeout=15
        )
        resp.raise_for_status()
        symbols = [
            s['instId'] for s in resp.json().get('data', [])
            if s['settleCcy'] == 'USDT' and s['state'] == 'live'
        ]
        print(f"  -> {len(symbols)} active USDT swap pairs")
        return symbols
    except Exception as e:
        print(f"  -> ERROR: {e}")
        return []


def fetch_binance_daily(symbol):
    """Fetch ALL available daily klines from Binance spot API."""
    all_dfs = []
    start_ms = 0
    for _ in range(20):
        params = f"symbol={symbol}&interval=1d&limit=1000"
        if start_ms:
            params += f"&startTime={start_ms}"
        url = f"https://api.binance.com/api/v3/klines?{params}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            df = pd.DataFrame(data, columns=[
                'OT', 'Open', 'High', 'Low', 'Close', 'Volume',
                'CT', 'QV', 'T', 'TBB', 'TBQ', 'Ign'
            ])
            df['Date'] = pd.to_datetime(df['OT'], unit='ms')
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[c] = df[c].astype(float)
            all_dfs.append(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])
            start_ms = int(data[-1][6]) + 1
            if len(data) < 1000:
                break
            time.sleep(0.1)
        except Exception:
            break
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True).drop_duplicates('Date').sort_values('Date').reset_index(drop=True)
    return None


def fetch_okx_daily(inst_id):
    """Fetch daily candles from OKX (history endpoint)."""
    base = inst_id.split('-')[0]
    spot = f"{base}-USDT"
    all_dfs = []
    after = ''
    for _ in range(20):
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={spot}&bar=1D&limit=300"
        if after:
            url += f"&after={after}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json().get('data', [])
            if not data:
                break
            rows = [{
                'Date': pd.to_datetime(int(d[0]), unit='ms'),
                'Open': float(d[1]), 'High': float(d[2]),
                'Low': float(d[3]), 'Close': float(d[4]),
                'Volume': float(d[5])
            } for d in data]
            all_dfs.append(pd.DataFrame(rows))
            after = data[-1][0]
            if len(data) < 300:
                break
            time.sleep(0.15)
        except Exception:
            break
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True).drop_duplicates('Date').sort_values('Date').reset_index(drop=True)
    return None


# ============================================================
# PATTERN DETECTION
# ============================================================

def find_rallies(df, symbol, exchange, min_gain=200, min_days=14, max_days=120):
    """Find all big rally patterns in a price series."""
    if df is None or len(df) < 60:
        return []

    category, base = get_category(symbol.replace('-USDT-SWAP', '').replace('-USDT', ''))
    highs = df['High'].values
    lows = df['Low'].values
    dates = df['Date'].values
    n = len(df)
    rallies = []

    for i in range(n - min_days):
        entry_low = lows[i]
        if entry_low <= 0:
            continue

        # Check local bottom (lowest in +/- 14 days)
        lb = max(0, i - 14)
        lf = min(n, i + 15)
        if entry_low > np.min(lows[lb:lf]):
            continue

        # Look forward for peak
        end = min(i + max_days + 1, n)
        future_highs = highs[i + 1:end]
        if len(future_highs) < min_days:
            continue

        peak_offset = np.argmax(future_highs)
        peak_price = future_highs[peak_offset]
        peak_idx = i + 1 + peak_offset

        gain = (peak_price - entry_low) / entry_low * 100
        days = (pd.Timestamp(dates[peak_idx]) - pd.Timestamp(dates[i])).days

        if gain < min_gain or days < min_days:
            continue

        # 7-day momentum
        mom_end = min(i + 8, n)
        mom_highs = highs[i + 1:mom_end]
        momentum_7d = (np.max(mom_highs) - entry_low) / entry_low * 100 if len(mom_highs) > 0 else 0

        entry_date = pd.Timestamp(dates[i])

        rallies.append({
            'symbol': symbol,
            'base': base,
            'exchange': exchange,
            'category': category,
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'entry_price': round(float(entry_low), 6),
            'exit_date': pd.Timestamp(dates[peak_idx]).strftime('%Y-%m-%d'),
            'exit_price': round(float(peak_price), 6),
            'gain_pct': round(gain, 1),
            'days': days,
            'momentum_7d_pct': round(momentum_7d, 1),
            'year': int(entry_date.year),
        })

    # De-duplicate: keep best rally per 30-day window
    if not rallies:
        return []
    rallies.sort(key=lambda x: x['gain_pct'], reverse=True)
    filtered = []
    used = []
    for r in rallies:
        ed = pd.Timestamp(r['entry_date'])
        if any(abs((ed - u).days) < 30 for u in used):
            continue
        filtered.append(r)
        used.append(ed)
    return filtered


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Exchange Big Trend Scanner')
    parser.add_argument('--min-gain', type=float, default=200, help='Minimum gain %% (default: 200)')
    parser.add_argument('--year', type=int, default=None, help='Filter by entry year')
    parser.add_argument('--slow-only', action='store_true', help='Only ZEC-like slow accumulation (7d mom < 50%%)')
    parser.add_argument('--multi-year', action='store_true', help='Only coins with rallies in 2+ years')
    args = parser.parse_args()

    today = datetime.now().strftime('%Y%m%d')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'data', 'scan_results')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"  TREND SCANNER  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Min gain: {args.min_gain}%  |  Year filter: {args.year or 'ALL'}")
    print("=" * 80)

    # 1. Get symbols
    binance_syms = get_binance_futures_symbols()
    okx_syms = get_okx_futures_symbols()

    # Build de-duplicated list (Binance priority)
    binance_bases = {}
    for s in binance_syms:
        b = s.replace('USDT', '')
        binance_bases[b] = s
    okx_only = {}
    for s in okx_syms:
        b = s.split('-')[0]
        if b not in binance_bases:
            okx_only[b] = s

    total = len(binance_bases) + len(okx_only)
    print(f"\nTotal coins to scan: {total} (Binance: {len(binance_bases)}, OKX-only: {len(okx_only)})\n")

    # 2. Scan
    all_rallies = []
    count = 0

    for base_asset, symbol in sorted(binance_bases.items()):
        count += 1
        cat, _ = get_category(symbol)
        df = fetch_binance_daily(symbol)
        if df is not None and len(df) > 60:
            found = find_rallies(df, symbol, 'Binance', min_gain=args.min_gain)
            if found:
                all_rallies.extend(found)
                best = max(found, key=lambda x: x['gain_pct'])
                print(f"  [{count}/{total}] {symbol:<16} ({cat:<12}) {len(df):>5}d | {len(found)} rallies | Best: +{best['gain_pct']}%")
            else:
                print(f"  [{count}/{total}] {symbol:<16} ({cat:<12}) {len(df):>5}d | --")
        else:
            print(f"  [{count}/{total}] {symbol:<16} ({cat:<12}) no data")
        time.sleep(0.12)

    for base_asset, inst_id in sorted(okx_only.items()):
        count += 1
        cat, _ = get_category(inst_id.replace('-USDT-SWAP', ''))
        df = fetch_okx_daily(inst_id)
        if df is not None and len(df) > 60:
            found = find_rallies(df, inst_id, 'OKX', min_gain=args.min_gain)
            if found:
                all_rallies.extend(found)
                best = max(found, key=lambda x: x['gain_pct'])
                print(f"  [{count}/{total}] {inst_id:<16} ({cat:<12}) {len(df):>5}d | {len(found)} rallies | Best: +{best['gain_pct']}%")
            else:
                print(f"  [{count}/{total}] {inst_id:<16} ({cat:<12}) {len(df):>5}d | --")
        else:
            print(f"  [{count}/{total}] {inst_id:<16} ({cat:<12}) no data")
        time.sleep(0.15)

    if not all_rallies:
        print("\nNo rallies found.")
        return

    # 3. Build DataFrame
    df_all = pd.DataFrame(all_rallies)
    df_all = df_all.sort_values('gain_pct', ascending=False).reset_index(drop=True)

    # Apply filters
    if args.year:
        df_all = df_all[df_all['year'] == args.year]
    if args.slow_only:
        df_all = df_all[(df_all['momentum_7d_pct'] < 50) & (df_all['days'] >= 30)]

    # 4. Save: all_rallies
    path_all = os.path.join(out_dir, f'all_rallies_{today}.csv')
    df_all.to_csv(path_all, index=False, encoding='utf-8-sig')
    print(f"\n[SAVED] {len(df_all)} rallies -> {path_all}")

    # 5. Save: ZEC-like (slow accumulation)
    df_zec = df_all[(df_all['momentum_7d_pct'] < 50) & (df_all['gain_pct'] >= 300) & (df_all['days'] >= 30)]
    df_zec = df_zec.sort_values('gain_pct', ascending=False)
    path_zec = os.path.join(out_dir, f'zec_like_{today}.csv')
    df_zec.to_csv(path_zec, index=False, encoding='utf-8-sig')
    print(f"[SAVED] {len(df_zec)} ZEC-like rallies -> {path_zec}")

    # 6. Save: multi-year coins
    year_groups = df_all.groupby('symbol')['year'].apply(lambda x: sorted(x.unique().tolist()))
    multi_year_syms = year_groups[year_groups.apply(len) >= 2].index.tolist()
    df_multi = df_all[df_all['symbol'].isin(multi_year_syms)].copy()
    df_multi = df_multi.sort_values(['symbol', 'year', 'gain_pct'], ascending=[True, True, False])
    path_multi = os.path.join(out_dir, f'multi_year_{today}.csv')
    df_multi.to_csv(path_multi, index=False, encoding='utf-8-sig')
    print(f"[SAVED] {len(df_multi)} multi-year rallies ({len(multi_year_syms)} coins) -> {path_multi}")

    # 7. Save: by_category summary
    cat_summary = df_all.groupby('category').agg(
        rally_count=('gain_pct', 'count'),
        coin_count=('symbol', 'nunique'),
        avg_gain=('gain_pct', 'mean'),
        max_gain=('gain_pct', 'max'),
        avg_days=('days', 'mean'),
        avg_momentum_7d=('momentum_7d_pct', 'mean'),
    ).round(1).sort_values('rally_count', ascending=False)
    path_cat = os.path.join(out_dir, f'by_category_{today}.csv')
    cat_summary.to_csv(path_cat, encoding='utf-8-sig')
    print(f"[SAVED] Category summary -> {path_cat}")

    # 7.5 Save: Combined Priority Coins JSON
    import json
    priority_set = set(df_zec['symbol'].tolist() + multi_year_syms)
    clean_priority = set()
    for s in priority_set:
        # Convert OKX format to Binance format if needed
        clean = s.replace('-USDT-SWAP', 'USDT').replace('-USDT', 'USDT')
        if not clean.endswith('USDT'):
            clean += 'USDT'
        clean_priority.add(clean)
        
    priority_list = sorted(list(clean_priority))
    path_priority = os.path.join(out_dir, 'scanned_priority_coins.json')
    with open(path_priority, 'w', encoding='utf-8') as f:
        json.dump(priority_list, f, indent=4)
    print(f"[SAVED] {len(priority_list)} Priority Coins for Batch Download -> {path_priority}")

    # 8. Print summary
    print(f"\n{'=' * 80}")
    print(f"  SCAN COMPLETE  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Total rallies: {len(df_all)} from {df_all['symbol'].nunique()} coins")
    print(f"  ZEC-like (slow): {len(df_zec)}")
    print(f"  Multi-year coins: {len(multi_year_syms)}")
    print(f"{'=' * 80}")

    print(f"\n--- TOP 20 ---")
    cols = ['symbol', 'category', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'gain_pct', 'days', 'momentum_7d_pct']
    print(df_all[cols].head(20).to_string(index=False))

    print(f"\n--- BY CATEGORY ---")
    print(cat_summary.to_string())

    print(f"\n--- BY YEAR ---")
    year_summary = df_all.groupby('year').agg(
        rallies=('gain_pct', 'count'),
        avg_gain=('gain_pct', 'mean'),
        coins=('symbol', 'nunique'),
    ).round(1)
    print(year_summary.to_string())

    print(f"\nOutput directory: {out_dir}")


if __name__ == '__main__':
    main()
