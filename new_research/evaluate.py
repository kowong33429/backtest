"""Decision metrics for the walk-forward report: RANDOM percentiles, month-block bootstrap of the expectancy
difference NEW_XGB - LEGACY_FIXED, cross-sectional vs timing skill, and the section-11 criteria table."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from new_research.run_all import load, save, CFG          # noqa: E402
from common_eval.report_tables import random_summary      # noqa: E402

LEG = pickle.load(open(os.path.join(ROOT, 'common_eval', '.cache', 'long_results.pkl'), 'rb'))['L4_+M4_keep_NaN']
BENCH = pickle.load(open(os.path.join(ROOT, 'common_eval', '.cache', 'bench_results.pkl'), 'rb'))


def block_boot_diff(a_td, b_td, n=10000, seed=0):
    """95% CI of mean(ret_a) - mean(ret_b), resampling calendar months of entries within each strategy."""
    rng = np.random.default_rng(seed)

    def prep(td):
        m = pd.to_datetime(td['entry_time']).dt.to_period('M')
        g = td.groupby(m)['ret'].agg(['sum', 'count'])
        return g['sum'].values, g['count'].values

    sa, ca = prep(a_td)
    sb, cb = prep(b_td)
    d = np.empty(n)
    for k in range(n):
        ia, ib = rng.integers(0, len(sa), len(sa)), rng.integers(0, len(sb), len(sb))
        d[k] = sa[ia].sum() / ca[ia].sum() - sb[ib].sum() / cb[ib].sum()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(a_td['ret'].mean() - b_td['ret'].mean())


def xsec(o, col):
    """Skill split: timing (day-mean score) vs cross-sectional (score demeaned within each day)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    d = o[o['y'].notna() & o[col].notna()]
    dm = d.groupby('day')[col].transform('mean')
    return {'roc_all': roc_auc_score(d['y'], d[col]), 'roc_day_mean_only': roc_auc_score(d['y'], dm),
            'roc_within_day': roc_auc_score(d['y'], d[col] - dm), 'pr_all': average_precision_score(d['y'], d[col]),
            'pr_within_day': average_precision_score(d['y'], d[col] - dm), 'base': d['y'].mean()}


def main():
    bt, rnd, mods = load('backtest_up'), load('random_up'), load('models')
    out = {'random': {}, 'bootstrap': {}, 'xsec': {}}
    rnd = dict(rnd)
    rnd.update(load('random_up_NEW_LOGISTIC'))
    for k in ('NEW_XGB', 'NEW_XGB_regime', 'NEW_LOGISTIC', 'NEW_LOGISTIC_regime'):
        for (mode, ex), rs in rnd[k].items():
            t = random_summary(rs, bt[k][ex]['wf'])
            t.insert(0, 'exit', ex); t.insert(0, 'random_mode', mode); t.insert(0, 'variant', k)
            t['random_mean_trades'] = rs['trades'].mean(); t['variant_trades'] = bt[k][ex]['wf']['trades']
            out['random'][(k, mode, ex)] = t
    for k in [x for x in bt if not x.startswith('_')]:
        for ex in ('native', 'unified'):
            out['bootstrap'][(k, ex)] = block_boot_diff(bt[k][ex]['td'], LEG[ex]['td']) if len(bt[k][ex]['td']) else (np.nan,) * 3
    for tgt in ('up', 'down'):
        for m in ('rule', 'logistic', 'xgb'):
            out['xsec'][(tgt, m)] = xsec(mods[tgt]['oof'], m)
    # section 11 criteria (walk-forward part; holdout part comes in step 10)
    crit = []
    btc = BENCH['BTC_HOLD']['wf']
    for k in [x for x in bt if not x.startswith('_')]:
        for ex in ('native', 'unified'):
            w = bt[k][ex]['wf']
            lo, hi, diff = out['bootstrap'][(k, ex)]
            tag = '_regime' if k.endswith('_regime') else ''
            rob = (bt['_robustness'].get(f'cost_x2{tag}') if k.startswith('NEW_XGB') else
                   bt['_robustness'].get(f'logistic_cost_x2{tag}') if k.startswith('NEW_LOGISTIC') else None)
            pct = None
            if (k, 'timing', ex) in out['random']:
                r = out['random'][(k, 'timing', ex)]
                pct = float(r.loc[r['metric'] == 'expectancy', 'percentile_of_variant'].iloc[0])
            crit.append({'variant': k, 'exit': ex, 'trades': w['trades'], 'expectancy': w['expectancy'],
                         'exp_minus_LEGACY_FIXED': diff, 'boot_CI_lo': lo, 'boot_CI_hi': hi,
                         'c1_beats_LEGACY_and_CI_excludes_0': bool(diff > 0 and lo > 0),
                         'random_pct_expectancy_timing': pct, 'c2_beats_RANDOM_P95': (pct is not None and pct > 95),
                         'sharpe': w['Sharpe'], 'calmar': w['Calmar'], 'btc_sharpe': btc['Sharpe'], 'btc_calmar': btc['Calmar'],
                         'c3_sharpe_or_calmar_above_BTC': bool((w['Sharpe'] or -9) > btc['Sharpe'] or (w['Calmar'] or -9) > btc['Calmar']),
                         'exp_ex_top3': w['expectancy_ex_top3'],
                         'exp_cost_x2': (rob[ex]['wf']['expectancy'] if rob else np.nan),
                         'c4_profitable_ex_top3_and_cost_x2': bool(w['expectancy_ex_top3'] > 0 and (rob is not None and rob[ex]['wf']['expectancy'] > 0))})
    out['criteria'] = pd.DataFrame(crit)
    out['criteria']['passes_all_wf'] = out['criteria'][['c1_beats_LEGACY_and_CI_excludes_0', 'c2_beats_RANDOM_P95',
                                                        'c3_sharpe_or_calmar_above_BTC', 'c4_profitable_ex_top3_and_cost_x2']].all(axis=1)
    save('evaluate', out)
    pd.set_option('display.width', 250)
    print(out['criteria'].round(4).to_string())
    for k, v in out['xsec'].items():
        print(k, {a: round(b, 4) for a, b in v.items()})
    for k, t in out['random'].items():
        r = t[t.metric.isin(['expectancy', 'CAGR'])]
        print(k, r[['metric', 'variant_value', 'rand_p50', 'rand_p95', 'percentile_of_variant', 'random_mean_trades', 'variant_trades']].round(4).values.tolist())


if __name__ == '__main__':
    main()
