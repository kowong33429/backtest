"""Holdout verdict against the pre-registered criteria (reports/holdout_hypotheses.md). Reporting only - no tuning."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from new_research.holdout import CACHE                         # noqa: E402
from new_research.evaluate import block_boot_diff              # noqa: E402
from common_eval.report_tables import random_summary           # noqa: E402


def main():
    R = pickle.load(open(os.path.join(CACHE, 'holdout_results.pkl'), 'rb'))
    rnd = pickle.load(open(os.path.join(CACHE, 'holdout_random.pkl'), 'rb'))
    res = R['res']
    btc = res['_bench']['BTC_HOLD']['wf']
    rows, rt = [], []
    for k, ref, is_up in (('NEW_LOGISTIC', 'LEGACY_FIXED_long', True), ('NEW_DOWN_BASELINE', 'LEGACY_FIXED_short', False),
                          ('LEGACY_FIXED_long', None, True), ('LEGACY_FIXED_short', None, False)):
        for ex in ('native', 'unified'):
            w = res[k][ex]['wf']
            pct = {}
            if k in rnd:
                for mode in ('timing', 'rate'):
                    t = random_summary(rnd[k][(mode, ex)], w)
                    t.insert(0, 'exit', ex); t.insert(0, 'random_mode', mode); t.insert(0, 'variant', k)
                    t['random_mean_trades'] = rnd[k][(mode, ex)]['trades'].mean(); t['variant_trades'] = w['trades']
                    rt.append(t)
                    pct[mode] = float(t.loc[t['metric'] == 'expectancy', 'percentile_of_variant'].iloc[0])
            lo = hi = diff = np.nan
            if ref:
                lo, hi, diff = block_boot_diff(res[k][ex]['td'], res[ref][ex]['td'])
            cx = res['_cost_x2'].get(k)
            r = {'variant': k, 'exit': ex, 'trades': w['trades'], 'win_rate': w['win_rate'], 'avg_win': w['avg_win'],
                 'avg_loss': w['avg_loss'], 'expectancy': w['expectancy'], 'exp_ex_top3': w['expectancy_ex_top3'],
                 'CAGR': w['CAGR'], 'MaxDD': w['MaxDD'], 'Sharpe': w['Sharpe'], 'Sortino': w['Sortino'], 'Calmar': w['Calmar'],
                 'vs_ref': ref, 'exp_minus_ref': diff, 'boot_CI_lo': lo, 'boot_CI_hi': hi,
                 'random_pct_timing': pct.get('timing'), 'random_pct_rate': pct.get('rate'),
                 'exp_cost_x2': cx[ex]['wf']['expectancy'] if cx else np.nan,
                 'pnl_total': w['pnl_total'], 'pnl_ex_top3': w['pnl_total_ex_top3']}
            if ref:
                r['c1'] = bool(diff > 0 and lo > 0)
                r['c2'] = bool(pct.get('timing', 0) > 95 and pct.get('rate', 0) > 95)
                r['c3'] = bool((w['Sharpe'] > btc['Sharpe']) or (w['Calmar'] > btc['Calmar'])) if is_up else None
                r['c4'] = bool(w['expectancy_ex_top3'] > 0 and cx is not None and cx[ex]['wf']['expectancy'] > 0)
                r['enough_trades'] = w['trades'] >= 30
                need = ['c1', 'c2', 'c4'] + (['c3'] if is_up else [])
                r['passes_holdout'] = bool(all(r[c] for c in need) and r['enough_trades'])
            rows.append(r)
    crit = pd.DataFrame(rows)
    # all variants table
    allv = []
    for k, r in res.items():
        if k.startswith('_'):
            continue
        for ex in ('native', 'unified'):
            allv.append({'variant': k, 'exit': ex, 'signals': r['n_signals'], **r[ex]['wf']})
    for k, v in res['_bench'].items():
        allv.append({'variant': k, 'exit': '-', **v['wf']})
    out = {'criteria': crit, 'random': pd.concat(rt, ignore_index=True), 'all': pd.DataFrame(allv), 'model': res['_model']}
    pickle.dump(out, open(os.path.join(CACHE, 'holdout_eval.pkl'), 'wb'))
    pd.set_option('display.width', 260)
    print(crit.round(4).to_string())
    print('BTC', {a: round(btc[a], 3) for a in ('CAGR', 'MaxDD', 'Sharpe', 'Calmar')})
    print(res['_model'])
    for k in ('NEW_LOGISTIC', 'NEW_DOWN_BASELINE', 'NEW_DOWN_XGB', 'LEGACY_FIXED_short'):
        td = res[k]['native']['td']
        top = td.nlargest(5, 'pnl')[['symbol', 'entry_time', 'exit_time', 'entry_px', 'exit_px', 'reason', 'ret', 'pnl']]
        print(k, 'top trades'); print(top.round(4).to_string())
        print(k, 'pnl share of top 3:', round(td.nlargest(3, 'pnl')['pnl'].sum() / td['pnl'].sum(), 3) if td['pnl'].sum() else None,
              '| yearly', res[k]['native']['yearly'].round(3).to_dict())


if __name__ == '__main__':
    main()
