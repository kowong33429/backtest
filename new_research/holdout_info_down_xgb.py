"""Informational only (NOT a pre-registered hypothesis): RANDOM percentiles of NEW_DOWN_XGB on the holdout."""
import os
import pickle
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from new_research.holdout import hcfg, random_for, pool_mask, CACHE, load_market   # noqa: E402
from common_eval.report_tables import random_summary                               # noqa: E402

if __name__ == '__main__':
    R = pickle.load(open(os.path.join(CACHE, 'holdout_results.pkl'), 'rb'))['res']
    S = pd.read_parquet(os.path.join(CACHE, 'scores_holdout.parquet')); S['day'] = pd.to_datetime(S['day'])
    c = hcfg(new_exits=True)
    mk = load_market(c, start='2025-05-01')
    rnd = random_for(c, mk, R['NEW_DOWN_XGB'], pool_mask(mk, S['day'], S['symbol'], pd.Timedelta(days=1)), -1, 1000)
    pickle.dump(rnd, open(os.path.join(CACHE, 'holdout_random_down_xgb_info.pkl'), 'wb'))
    for (m, ex), rs in rnd.items():
        t = random_summary(rs, R['NEW_DOWN_XGB'][ex]['wf'])
        t = t[t.metric.isin(['expectancy', 'CAGR'])]
        print(m, ex, t[['metric', 'variant_value', 'rand_p50', 'rand_p95', 'percentile_of_variant']].round(4).values.tolist(),
              'rand trades', round(rs.trades.mean()))
