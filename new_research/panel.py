"""Builds the daily NEW panel: features (section 6) + labels (section 4, base and robustness variants) for every
coin of the universe, development period only. -> data_new/panel_daily.parquet

Cross-sectional features use only the same day's values across coins (no future). Relative strength vs BTC uses
BTCUSDT from the same data.
"""
import os
import sys
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from new_research.data import load_coin, metadata, CFG, CUT          # noqa: E402
from new_research.features import coin_features, daily_sample        # noqa: E402
from new_research.labels import label_bars                           # noqa: E402
from new_research.macro import regime_daily                          # noqa: E402

H = CFG['horizons']['H_NEW']
LB = CFG['new']['label']
# (name, H, MAE): base + robustness (H_NEW and MAE_LIMIT +-20%)
LABEL_SETS = [('', H, LB['MAE_LIMIT']), ('_h432', int(H * 0.8), LB['MAE_LIMIT']), ('_h648', int(H * 1.2), LB['MAE_LIMIT']),
              ('_mae20', H, LB['MAE_LIMIT'] * 0.8), ('_mae30', H, LB['MAE_LIMIT'] * 1.2)]


def build_one(args):
    sym, listed, fut, delisted, sector = args
    d = load_coin(sym)
    if d is None or len(d) < 6 * 120:
        return None
    f = coin_features(d, listed, fut)
    pos = pd.Series(np.arange(len(d)), index=d.index)
    day = daily_sample(f)
    idx = pos.loc[day['bar_time']].values
    for suf, h, mae in LABEL_SETS:
        up, dn, upk, dnk, peak = label_bars(d, idx, h, LB['UP_TARGET'], LB['DOWN_TARGET'], mae, delisted)
        day['up' + suf], day['down' + suf] = up, dn
        if suf == '':
            day['up_peak'] = peak
            day['up_hit_time'] = np.where(upk >= 0, d.index.values[np.clip(upk, 0, None)], np.datetime64('NaT'))
            day['down_hit_time'] = np.where(dnk >= 0, d.index.values[np.clip(dnk, 0, None)], np.datetime64('NaT'))
    day['close'] = d['Close'].loc[day['bar_time']].values
    day['symbol'], day['sector'] = sym, sector
    num = day.select_dtypes('float64').columns
    day[num] = day[num].astype('float32')
    return day


def add_cross_section(p):
    btc = p[p['symbol'] == 'BTCUSDT'][['ret_7d', 'ret_30d', 'ret_90d']]
    for nd in (7, 30, 90):
        b = btc[f'ret_{nd}d'].reindex(p.index).values
        p[f'rs_{nd}d'] = p[f'ret_{nd}d'] - b
        p[f'rs_{nd}d_rank'] = p.groupby(level=0)[f'rs_{nd}d'].rank(pct=True)
    # sector mean 30d return excluding the coin itself, and the coin's rank within its sector (same day)
    g = p.groupby([p.index, 'sector'])['ret_30d']
    s_sum, s_cnt = g.transform('sum'), g.transform('count')
    p['sector_ret_30d_ex_self'] = ((s_sum - p['ret_30d']) / (s_cnt - 1)).where(s_cnt > 1)
    p['sector_rank_30d'] = g.rank(pct=True)
    return p


def main():
    t0 = time.time()
    md = metadata().set_index('symbol')
    syms = json.load(open(os.path.join(ROOT, 'data_all', 'universe_symbols.json')))['symbols']
    jobs = [(s, md.at[s, 'listed_first'] if s in md.index else None, md.at[s, 'futures_first'] if s in md.index else None,
             bool(md.at[s, 'delisted_before_holdout']) if s in md.index else False,
             md.at[s, 'sector'] if s in md.index else 'other') for s in syms]
    with Pool(12) as pool:
        frames = [x for x in pool.imap_unordered(build_one, jobs, chunksize=4) if x is not None]
    p = pd.concat(frames).sort_index()
    p.index.name = 'day'
    p = add_cross_section(p)
    btc_4h = load_coin('BTCUSDT')['Close']
    days = pd.DatetimeIndex(sorted(p.index.unique()))
    reg = regime_daily(days, btc_4h)
    p = p.join(reg, how='left')
    p.index.name = 'day'
    assert p['bar_time'].max() < CUT
    out = os.path.join(ROOT, 'data_new', 'panel_daily.parquet')
    p.reset_index().to_parquet(out, index=False)
    print(f'panel {len(p):,} rows, {p.symbol.nunique()} coins, {p.index.min().date()}..{p.index.max().date()} '
          f'up rate {p.up.mean():.4f} down rate {p.down.mean():.4f} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
