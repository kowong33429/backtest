"""
universe_download.py — Survivorship-free 4H data for EVERY Binance USDT-M perpetual that ever
existed (active AND delisted), from the public archive at data.binance.vision.

Why: the research basket (data/scan_results/scanned_priority_coins.json) was picked by
trend_scanner AFTER the fact — coins that already had big rallies. Any backtest on it is
survivorship-inflated. This builds the full, unselected universe so strategies can be tested on
"what you could have known at the time" (see rotation_backtest.py --universe-dir).

Output (kept OUT of data/ on purpose — every other script globs data/*/*_4h_full.csv):
    data_all/<symbol>/<SYMBOL>_4h_full.csv   Date, Open, High, Low, Close, Volume, Quote_Volume, Funding_Rate
    data_all/universe_symbols.json           symbol list + which were excluded (stocks / ETFs / indices)

Prices are PERP prices (data/ holds SPOT prices); funding is merged backward-asof like
data_downloader.py. Completed months come from the monthly archive; the current month from the
daily archive. Re-running only fetches months that are missing.

Usage:
    python core/universe_download.py                 # all symbols
    python core/universe_download.py --symbols ZECUSDT TUTUSDT
    python core/universe_download.py --workers 16
"""
import os
import io
import re
import json
import time
import zipfile
import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

S3 = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
DL = 'https://data.binance.vision'
KLINE_COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume',
              'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']

# Not crypto: tokenised stocks, ETFs, commodities, FX, pre-IPO and index perps.
NON_CRYPTO = set("""
AAOI AAPL ACN ADBE AGPU ALAB AMAT AMC AMD AMZN ANET ANTHROPIC APLD APP ARM ASML ASTS AVGO AXTI BABA BE
BITO BMNR BRKB BWET BX BYD BZ CAT CBRS CIEN CL COHR COIN COPPER COST CRCL CRDO CRM CRML CRWD CRWV CSCO
CSOPSAMSUNG2L CSOPSKHYNIX2L CVNA CXMT DDOG DELL DIS DJT DKNG DRAM EBAY EWJ EWT EWY EWZ FLNC FWDI GDX GEV
GIGADEV GLW GME GOOGL GPRO GS GTLB HANMI HD HIMS HK0625 HK0700 HK0992 HK1810 HOOD HPE HUT HYUNDAI IBM INTC
INTW IONQ IREN IWM JPM KLAC KODEX200 KORU KO KUAISHOU LGELECTRONICS LITE LLY LRCX MARA MDB MDT MEITUAN
MINIMAX MP MRK MRNA MRVL MSFT MSTR MU MUU NATGAS NAVER NBIS NET NFLX NKE NOK NOW NVDA NVDL NVO OKLO ONDS
OPENAI ORCL OURA PANW PATH PDD PLTR POPMART PYPL QCOM QQQ RDDT RIVN RKLB SAMSUNG SAMSUNGEM SHOP SKHYNIX
SKHY SKUU SMCI SMH SNDK SNOW SNXX SOFI SONY SOXL SOXS SPCX SPY SQQQ STRC STXX TEAM TEM TENCENT TER TMF
TQQQ TSLA TSLL TSM TTWO TXN TZA UBER UNH UNITREE URNM USAR UVXY VRT VST V WDC WMT XAG XAU XBI XLE XOM
XPD XPT ZHIPU ZHONGJI ZM ZS TBT USDBRL NFP DEFI BTCDOM ALL USDC FDUSD TUSD BUSD
""".split())


def s3_list(prefix, delimiter='/'):
    """All common prefixes / keys under an S3 prefix (handles pagination)."""
    out, marker = [], ''
    while True:
        r = requests.get(f'{S3}?delimiter={delimiter}&prefix={prefix}' + (f'&marker={marker}' if marker else ''), timeout=60).text
        pref = re.findall(r'<Prefix>([^<]+)</Prefix>', r)[1:]          # first one is the query prefix itself
        keys = re.findall(r'<Key>([^<]+)</Key>', r)
        out += pref + keys
        if '<IsTruncated>true</IsTruncated>' not in r:
            return out
        marker = (pref + keys)[-1]


def read_zip_csv(content, names):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    first = raw.split('\n', 1)[0]
    has_header = not first.split(',')[0].strip().lstrip('-').isdigit()
    return pd.read_csv(io.StringIO(raw), header=0 if has_header else None, names=None if has_header else names)


def fetch(url, session, tries=4):
    for k in range(tries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(1.5 * (k + 1))
    return None


def download_symbol(sym, out_root, session):
    path = os.path.join(out_root, sym.lower(), f'{sym}_4h_full.csv')
    have = set()
    old = None
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=['Date'])
        have = set(old['Date'].dt.strftime('%Y-%m'))
    months = sorted(m.rstrip('.zip')[-7:] for m in s3_list(f'data/futures/um/monthly/klines/{sym}/4h/') if m.endswith('.zip'))
    today = dt.date.today()
    cur = f'{today:%Y-%m}'
    complete = [m for m in months if m not in have or m == max(have, default='')]   # refetch last stored month
    frames = []
    for m in complete:
        c = fetch(f'{DL}/data/futures/um/monthly/klines/{sym}/4h/{sym}-4h-{m}.zip', session)
        if c:
            frames.append(read_zip_csv(c, KLINE_COLS))
    # current (incomplete) month from the daily archive
    if not months or months[-1] < cur:
        for d in pd.date_range(f'{cur}-01', today - dt.timedelta(days=1)):
            c = fetch(f'{DL}/data/futures/um/daily/klines/{sym}/4h/{sym}-4h-{d:%Y-%m-%d}.zip', session)
            if c:
                frames.append(read_zip_csv(c, KLINE_COLS))
    if not frames and old is None:
        return sym, 0, 'no klines'
    k = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KLINE_COLS)
    if len(k):
        k = k.rename(columns={'open_time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                              'volume': 'Volume', 'quote_volume': 'Quote_Volume'})
        k['Date'] = pd.to_datetime(pd.to_numeric(k['Date']), unit='ms')
        k = k[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote_Volume']].apply(
            lambda s: pd.to_numeric(s, errors='coerce') if s.name != 'Date' else s)
    base_df = old.drop(columns=['Funding_Rate'], errors='ignore') if old is not None else None
    df = pd.concat([base_df, k], ignore_index=True) if base_df is not None else k
    df = df.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)

    # funding (monthly archive; small files)
    fr = []
    for m in sorted(x.rstrip('.zip')[-7:] for x in s3_list(f'data/futures/um/monthly/fundingRate/{sym}/') if x.endswith('.zip')):
        c = fetch(f'{DL}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip', session)
        if c:
            f = read_zip_csv(c, ['calc_time', 'funding_interval_hours', 'last_funding_rate'])
            fr.append(f)
    if fr:
        f = pd.concat(fr, ignore_index=True)
        tcol = 'calc_time' if 'calc_time' in f else f.columns[0]
        rcol = 'last_funding_rate' if 'last_funding_rate' in f else f.columns[-1]
        f = pd.DataFrame({'Date': pd.to_datetime(pd.to_numeric(f[tcol]), unit='ms'),
                          'Funding_Rate': pd.to_numeric(f[rcol], errors='coerce')}).dropna().sort_values('Date')
        df = pd.merge_asof(df.sort_values('Date'), f, on='Date', direction='backward')
        df['Funding_Rate'] = df['Funding_Rate'].fillna(0.0)
    else:
        df['Funding_Rate'] = 0.0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return sym, len(df), 'ok'


def main():
    ap = argparse.ArgumentParser(description='Survivorship-free Binance USDT-M 4H universe')
    ap.add_argument('--symbols', nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default='data_all')
    args = ap.parse_args()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_root = os.path.join(base, args.out)
    os.makedirs(out_root, exist_ok=True)

    if args.symbols:
        syms = [s.upper() for s in args.symbols]
    else:
        all_ = [p.rstrip('/').split('/')[-1] for p in s3_list('data/futures/um/monthly/klines/')]
        syms = sorted(s for s in set(all_) if s.endswith('USDT') and '_' not in s)
    excluded = sorted(s for s in syms if s[:-4] in NON_CRYPTO)
    syms = [s for s in syms if s[:-4] not in NON_CRYPTO]
    json.dump({'symbols': syms, 'excluded_non_crypto': excluded, 'built': str(dt.datetime.now())},
              open(os.path.join(out_root, 'universe_symbols.json'), 'w'), indent=1)
    print(f'[Universe] {len(syms)} crypto USDT perps ({len(excluded)} non-crypto excluded) -> {out_root}')

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        sess = requests.Session()
        futs = {ex.submit(download_symbol, s, out_root, sess): s for s in syms}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                s, n, msg = f.result()
                ok += msg == 'ok'
                fail += msg != 'ok'
            except Exception as e:      # keep going; report at the end
                fail += 1
                print(f'  {futs[f]}: ERROR {e}')
            if i % 50 == 0:
                print(f'  {i}/{len(syms)} done ({time.time() - t0:.0f}s) ok={ok} fail={fail}', flush=True)
    print(f'[Done] ok={ok} fail={fail} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
