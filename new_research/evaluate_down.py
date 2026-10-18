"""DOWN decision table (walk-forward): RANDOM percentiles, bootstrap vs LEGACY_FIXED short (S4, with funding),
section-11 criteria (criterion 3 'beat BTC_HOLD' is UP-only)."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from new_research.run_all import load, save                 # noqa: E402
from new_research.evaluate import block_boot_diff           # noqa: E402
from common_eval.report_tables import random_summary        # noqa: E402

S4 = pickle.load(open(os.path.join(ROOT, 'common_eval', '.cache', 'short_results.pkl'), 'rb'))['S4_+M4 keep NaN']


def main():
    bt, rnd = load('backtest_down'), load('random_down')
    rows, rtabs = [], []
    for k in [x for x in bt if not x.startswith('_')]:
        model = 'logistic' if 'LOGISTIC' in k else 'xgb' if 'XGB' in k else 'rule'
        for ex in ('native', 'unified'):
            w = bt[k][ex]['wf']
            lo, hi, diff = block_boot_diff(bt[k][ex]['td'], S4[ex]['td'])
            pct_t = pct_r = None
            if k in rnd:
                for mode in ('timing', 'rate'):
                    t = random_summary(rnd[k][(mode, ex)], w)
                    t.insert(0, 'exit', ex); t.insert(0, 'random_mode', mode); t.insert(0, 'variant', k)
                    t['random_mean_trades'] = rnd[k][(mode, ex)]['trades'].mean(); t['variant_trades'] = w['trades']
                    rtabs.append(t)
                    v = float(t.loc[t['metric'] == 'expectancy', 'percentile_of_variant'].iloc[0])
                    pct_t, pct_r = (v, pct_r) if mode == 'timing' else (pct_t, v)
            cost = bt['_robustness'].get(f'{model}_cost_x2') if not k.endswith('_btcdown') else None
            rows.append({'variant': k, 'exit': ex, 'trades': w['trades'], 'expectancy': w['expectancy'], 'CAGR': w['CAGR'],
                         'MaxDD': w['MaxDD'], 'Sharpe': w['Sharpe'], 'Calmar': w['Calmar'],
                         'exp_minus_LEGACY_FIXED_short': diff, 'boot_CI_lo': lo, 'boot_CI_hi': hi,
                         'c1': bool(diff > 0 and lo > 0), 'random_pct_timing': pct_t, 'random_pct_rate': pct_r,
                         'c2': bool(pct_t is not None and pct_t > 95 and pct_r > 95),
                         'exp_ex_top3': w['expectancy_ex_top3'],
                         'exp_cost_x2': cost[ex]['wf']['expectancy'] if cost else np.nan,
                         'c4': bool(w['expectancy_ex_top3'] > 0 and cost is not None and cost[ex]['wf']['expectancy'] > 0),
                         'funding_mean_per_trade': bt[k][ex]['td']['funding_ret'].mean() if len(bt[k][ex]['td']) else np.nan})
    crit = pd.DataFrame(rows)
    crit['passes_all_wf (c1,c2,c4)'] = crit[['c1', 'c2', 'c4']].all(axis=1)
    out = {'criteria': crit, 'random': pd.concat(rtabs, ignore_index=True) if rtabs else None}
    save('evaluate_down', out)
    pd.set_option('display.width', 250)
    print(crit.round(4).to_string())
    print('LEGACY_FIXED short (S4, funding):', {ex: {a: round(S4[ex]['wf'][a], 4) for a in ('trades', 'expectancy', 'CAGR', 'MaxDD')} for ex in ('native', 'unified')})
    rob = load('backtest_down')['_robustness']
    for k, r in rob.items():
        print(k, {ex: (round(r[ex]['wf']['expectancy'] * 100, 2), round(r[ex]['wf']['CAGR'] * 100, 1), r[ex]['wf']['trades']) for ex in ('native', 'unified')})
    for k in ('NEW_DOWN_BASELINE', 'NEW_DOWN_XGB', 'NEW_DOWN_LOGISTIC'):
        print(k, 'yearly native', bt[k]['native']['yearly'].round(3).to_dict(),
              'post-dev', {a: round(bt[k]['native']['post_dev'].get(a, np.nan), 4) for a in ('trades', 'expectancy', 'CAGR')})


if __name__ == '__main__':
    main()
