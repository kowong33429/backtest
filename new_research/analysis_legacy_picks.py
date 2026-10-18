"""Why is LEGACY_FIXED long (L4) consistently worse than RANDOM? Describe what it picks vs what was available.

For every L4 entry and for the trades of 200 RANDOM runs drawn from the same scored pool (rate-matched, native exit):
  liquidity  : qv30 percentile among the eligible coins on the entry bar
  extension  : ret_30d, close_ema300, rsi14, pct_from_ath (NEW features of the PREVIOUS day - strictly before entry)
  sector     : CoinGecko sector (data/coin_metadata.csv)
Outputs tables to new_research/.cache/legacy_picks.pkl (used by the report).
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config          # noqa: E402
from common_eval.data import load_market            # noqa: E402
from common_eval.backtest import Engine             # noqa: E402
from common_eval import benchmarks as bm            # noqa: E402

CFG = load_config()
FEATS = ['ret_30d', 'close_ema300', 'rsi14', 'pct_from_ath', 'rs_30d_rank', 'atr_pct']


def attach(td, mk, panel, md):
    t = td.copy()
    ti = np.fromiter((mk.idx(x) for x in t['entry_time']), int, len(t))
    ji = np.fromiter((mk.sym_idx[s] for s in t['symbol']), int, len(t))
    q = mk.qv30[ti]
    q = np.where(mk.elig[ti], q, np.nan)
    t['qv30'] = mk.qv30[ti, ji]
    t['liq_pct'] = [(np.nansum(row < v) / np.isfinite(row).sum()) for row, v in zip(q, t['qv30'])]
    t['feat_day'] = (pd.to_datetime(t['entry_time']) - pd.Timedelta(hours=4)).dt.normalize() - pd.Timedelta(days=1)
    key = panel.set_index(['symbol', 'day'])[FEATS]
    f = key.reindex(pd.MultiIndex.from_arrays([t['symbol'], t['feat_day']]))
    for c in FEATS:
        t[c] = f[c].values
    t['sector'] = t['symbol'].map(md.set_index('symbol')['sector']).fillna('other')
    return t


def main():
    lr = pickle.load(open(os.path.join(ROOT, 'common_eval', '.cache', 'long_results.pkl'), 'rb'))
    rec = lr['L4_+M4_keep_NaN']
    mk = load_market(CFG)
    eng = Engine(mk, CFG)
    panel = pd.read_parquet(os.path.join(ROOT, 'data_new', 'panel_daily.parquet'))
    panel['day'] = pd.to_datetime(panel['day'])
    md = pd.read_csv(os.path.join(ROOT, 'data', 'coin_metadata.csv'))
    out = {}
    rate = rec['n_signals'] / rec['pool_elig']
    for ex in ('native', 'unified'):
        L = attach(rec[ex]['td'], mk, panel, md)
        R = []
        for s in range(200):
            sig = bm.random_signals(mk, np.random.default_rng(10_000 + s), rate, mode='rate', mask=rec['pool_mask'])
            td, _ = eng.run(sig, ex)
            R.append(td)
        R = attach(pd.concat(R, ignore_index=True), mk, panel, md)
        cmp = pd.DataFrame({'LEGACY_FIXED_median': L[['liq_pct'] + FEATS].median(),
                            'RANDOM_median': R[['liq_pct'] + FEATS].median()})
        # outcome by bucket (quintiles defined on the RANDOM picks)
        buckets = {}
        for c in ['liq_pct', 'ret_30d', 'close_ema300', 'rsi14']:
            edges = np.unique(np.nanquantile(R[c], [0, .2, .4, .6, .8, 1]))
            lb = pd.cut(L[c], edges, include_lowest=True)
            rb = pd.cut(R[c], edges, include_lowest=True)
            buckets[c] = pd.DataFrame({'LEGACY_share': L.groupby(lb, observed=False).size() / len(L),
                                       'RANDOM_share': R.groupby(rb, observed=False).size() / len(R),
                                       'LEGACY_exp': L.groupby(lb, observed=False)['ret'].mean(),
                                       'RANDOM_exp': R.groupby(rb, observed=False)['ret'].mean()})
        sec = pd.DataFrame({'LEGACY_share': L['sector'].value_counts(normalize=True),
                            'RANDOM_share': R['sector'].value_counts(normalize=True),
                            'LEGACY_exp': L.groupby('sector')['ret'].mean(),
                            'RANDOM_exp': R.groupby('sector')['ret'].mean()}).sort_values('LEGACY_share', ascending=False)
        out[ex] = {'compare': cmp, 'buckets': buckets, 'sector': sec, 'legacy_trades': L, 'n_random_trades': len(R),
                   'year_share': pd.DataFrame({'LEGACY': pd.to_datetime(L['entry_time']).dt.year.value_counts(normalize=True),
                                               'RANDOM': pd.to_datetime(R['entry_time']).dt.year.value_counts(normalize=True)})}
        print(ex); print(cmp.round(3).to_string())
        for c, b in buckets.items():
            print(c); print(b.round(3).to_string())
        print(sec.head(10).round(3).to_string())
    pickle.dump(out, open(os.path.join(ROOT, 'new_research', '.cache', 'legacy_picks.pkl'), 'wb'))


if __name__ == '__main__':
    main()
