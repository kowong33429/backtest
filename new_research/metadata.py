"""Step B1.8: data/coin_metadata.csv - listing dates (spot/futures), delisting before the holdout, sector tag.

Sector = first matching CoinGecko category in SECTOR_PRIORITY (a coin belongs to many categories). Tags are
CoinGecko's CURRENT categorisation (not point-in-time); they are static descriptors (privacy coin, meme, ...),
so this is not a look-ahead on prices, but a coin re-tagged later (e.g. into "AI") would be mis-dated.
"""
import os
import re
import sys
import time
import json

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config, holdout_start   # noqa: E402

SECTOR_PRIORITY = [
    ('privacy', 'privacy-coins'), ('meme', 'meme-token'), ('ai', 'artificial-intelligence'),
    ('gaming', 'gaming'), ('oracle', 'oracle'), ('storage', 'storage'), ('exchange', 'exchange-based-tokens'),
    ('rwa', 'real-world-assets-rwa'), ('depin', 'depin'), ('nft_metaverse', 'non-fungible-tokens-nft'),
    ('nft_metaverse', 'metaverse'), ('layer2', 'layer-2'), ('zk', 'zero-knowledge-zk'),
    ('dex', 'decentralized-exchange'), ('lending', 'lending-borrowing'), ('liquid_staking', 'liquid-staking-tokens'),
    ('defi', 'decentralized-finance-defi'), ('payments', 'payment-solutions'), ('interoperability', 'interoperability'),
    ('infrastructure', 'infrastructure'), ('layer1', 'layer-1'), ('smart_contract', 'smart-contract-platform'),
]
CG = 'https://api.coingecko.com/api/v3'


def cg_get(path, params, tries=6):
    for k in range(tries):
        r = requests.get(CG + path, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(20 * (k + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f'CoinGecko rate limit: {path} {params}')


def base_asset(sym):
    b = sym[:-4] if sym.endswith('USDT') else sym
    return re.sub(r'^(1000000|100000|10000|1000|1M)', '', b)


def category_members(pages=4):
    out = {}
    for sector, cat in SECTOR_PRIORITY:
        rows = []
        for p in range(1, pages + 1):
            js = cg_get('/coins/markets', {'vs_currency': 'usd', 'category': cat, 'per_page': 250, 'page': p})
            rows += js
            time.sleep(2.5)
            if len(js) < 250:
                break
        out[(sector, cat)] = pd.DataFrame([{'id': r['id'], 'symbol': r['symbol'].upper(),
                                            'mcap': r.get('market_cap') or 0} for r in rows])
        print(f'  {cat}: {len(rows)} coins', flush=True)
    return out


def main():
    cfg = load_config()
    cut = holdout_start(cfg)
    dm = pd.read_csv(os.path.join(ROOT, 'data_new', '_download_meta.csv'),
                     parse_dates=['spot_first', 'futures_first', 'last_bar_before_holdout'])
    dm['base'] = dm['symbol'].map(base_asset)
    dm['listed_first'] = dm[['spot_first', 'futures_first']].min(axis=1)
    dm['delisted_before_holdout'] = dm['last_bar_before_holdout'] < cut - pd.Timedelta(days=2)
    dm['delist_date'] = dm['last_bar_before_holdout'].where(dm['delisted_before_holdout'])
    mem = category_members()
    sector, cg_id = [], []
    for b in dm['base']:
        s, i = 'other', None
        for (sec, cat), df in mem.items():
            hit = df[df['symbol'] == b]
            if len(hit):
                s, i = sec, hit.sort_values('mcap', ascending=False)['id'].iloc[0]
                break
        sector.append(s)
        cg_id.append(i)
    dm['sector'], dm['coingecko_id'] = sector, cg_id
    out = dm[['symbol', 'base', 'coingecko_id', 'sector', 'spot_first', 'futures_first', 'listed_first',
              'delisted_before_holdout', 'delist_date', 'rows_dev']]
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    out.to_csv(os.path.join(ROOT, 'data', 'coin_metadata.csv'), index=False)
    print(out['sector'].value_counts().to_string())
    print(f"delisted before holdout: {int(out['delisted_before_holdout'].sum())}")


if __name__ == '__main__':
    main()
