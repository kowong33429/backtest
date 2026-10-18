"""Step B1.1-B1.2: 4H klines with the full Binance column set for every symbol of the survivorship-free universe.

* Futures (USDT-M perp) klines are the primary series; SPOT klines extend the history before the perp was listed
  (source column says which). Columns: Quote_Volume, Trades, Taker_Buy_Volume, Taker_Buy_Quote_Volume.
* Funding_Rate is NaN where there is no futures market yet; has_futures = 0/1.
* Rows at/after FINAL_HOLDOUT_START are written to a SEPARATE tree (data_new_holdout/) and are never read by
  any step before step 10. Rows before it go to data_new/.

Re-uses the archive helpers of core/universe_download.py (LEGACY code is imported, not modified).
"""
import os
import sys
import json
import time
import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'core'))
sys.path.insert(0, ROOT)
from universe_download import s3_list, read_zip_csv, fetch, KLINE_COLS, DL   # noqa: E402
from common_eval.config import load_config, holdout_start                  # noqa: E402

COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote_Volume', 'Trades',
        'Taker_Buy_Volume', 'Taker_Buy_Quote_Volume']


def _klines(market, sym, session):
    base = 'data/futures/um' if market == 'futures' else 'data/spot'
    months = sorted(m[:-4][-7:] for m in s3_list(f'{base}/monthly/klines/{sym}/4h/') if m.endswith('.zip'))
    frames = []
    for m in months:
        c = fetch(f'{DL}/{base}/monthly/klines/{sym}/4h/{sym}-4h-{m}.zip', session)
        if c:
            frames.append(read_zip_csv(c, KLINE_COLS))
    if not frames:
        return None
    k = pd.concat(frames, ignore_index=True)
    k.columns = KLINE_COLS[:len(k.columns)] if k.columns[0] != 'open_time' else k.columns
    t = pd.to_numeric(k['open_time'], errors='coerce')
    t = np.where(t > 1e14, t // 1000, t)                     # spot archive switched to microseconds in 2025
    out = pd.DataFrame({'Date': pd.to_datetime(t, unit='ms'),
                        'Open': k['open'], 'High': k['high'], 'Low': k['low'], 'Close': k['close'],
                        'Volume': k['volume'], 'Quote_Volume': k['quote_volume'], 'Trades': k['count'],
                        'Taker_Buy_Volume': k['taker_buy_volume'], 'Taker_Buy_Quote_Volume': k['taker_buy_quote_volume']})
    for c in COLS[1:]:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out.dropna(subset=['Date']).drop_duplicates('Date', keep='last').sort_values('Date')


def _funding(sym, session):
    fr = []
    for m in sorted(x[:-4][-7:] for x in s3_list(f'data/futures/um/monthly/fundingRate/{sym}/') if x.endswith('.zip')):
        c = fetch(f'{DL}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip', session)
        if c:
            fr.append(read_zip_csv(c, ['calc_time', 'funding_interval_hours', 'last_funding_rate']))
    if not fr:
        return None
    f = pd.concat(fr, ignore_index=True)
    tcol = 'calc_time' if 'calc_time' in f else f.columns[0]
    rcol = 'last_funding_rate' if 'last_funding_rate' in f else f.columns[-1]
    return pd.DataFrame({'Date': pd.to_datetime(pd.to_numeric(f[tcol]), unit='ms'),
                         'Funding_Rate': pd.to_numeric(f[rcol], errors='coerce')}).dropna().sort_values('Date')


def build_symbol(sym, out_dev, out_hold, cut, session):
    fut = _klines('futures', sym, session)
    spot = _klines('spot', sym, session)
    if fut is None and spot is None:
        return sym, 'no klines', None
    parts = []
    if spot is not None:
        s = spot if fut is None else spot[spot['Date'] < fut['Date'].min()]
        parts.append(s.assign(source='spot', has_futures=0))
    if fut is not None:
        parts.append(fut.assign(source='futures', has_futures=1))
    df = pd.concat(parts, ignore_index=True).sort_values('Date').reset_index(drop=True)
    df['Funding_Rate'] = np.nan
    if fut is not None:
        f = _funding(sym, session)
        if f is not None:
            m = pd.merge_asof(df[['Date']], f, on='Date', direction='backward')['Funding_Rate']
            df['Funding_Rate'] = np.where(df['has_futures'] == 1, m, np.nan)   # never invent funding pre-futures
    meta = {'symbol': sym,
            'spot_first': spot['Date'].min() if spot is not None else pd.NaT,
            'futures_first': fut['Date'].min() if fut is not None else pd.NaT}
    dev, hold = df[df['Date'] < cut], df[df['Date'] >= cut]
    for frame, root in ((dev, out_dev), (hold, out_hold)):
        if len(frame):
            os.makedirs(os.path.join(root, sym.lower()), exist_ok=True)
            frame.to_csv(os.path.join(root, sym.lower(), f'{sym}_4h.csv'), index=False)
    # metadata from the DEVELOPMENT side only (no peeking at whether a coin trades after the cut)
    meta['last_bar_before_holdout'] = dev['Date'].max() if len(dev) else pd.NaT
    meta['rows_dev'] = len(dev)
    return sym, 'ok', meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*')
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()
    cfg = load_config()
    cut = holdout_start(cfg)
    syms = args.symbols or json.load(open(os.path.join(ROOT, 'data_all', 'universe_symbols.json')))['symbols']
    out_dev, out_hold = os.path.join(ROOT, 'data_new'), os.path.join(ROOT, 'data_new_holdout')
    metas, t0 = [], time.time()
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers * 2)
    sess.mount('https://', adapter)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(build_symbol, s, out_dev, out_hold, cut, sess): s for s in syms}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                s, msg, meta = f.result()
                if meta:
                    metas.append(meta)
                elif msg != 'ok':
                    print(f'  {s}: {msg}')
            except Exception as e:
                print(f'  {futs[f]}: ERROR {e}', flush=True)
            if i % 50 == 0:
                print(f'  {i}/{len(syms)} ({time.time() - t0:.0f}s)', flush=True)
    pd.DataFrame(metas).to_csv(os.path.join(out_dev, '_download_meta.csv'), index=False)
    print(f'[done] {len(metas)} symbols in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
