"""Assembles result tables (DataFrames) from the cached run outputs. Used by the report builder."""
import os
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'common_eval', '.cache')

COLS = ['trades', 'win_rate', 'avg_win', 'avg_loss', 'expectancy', 'expectancy_ex_top3', 'CAGR', 'MaxDD', 'Sharpe',
        'Sortino', 'Calmar', 'final_multiple', 'turnover_per_year', 'cost_per_year_pct_equity', 'avg_hold_days']


def load(name):
    return pickle.load(open(os.path.join(CACHE, name), 'rb'))


def ladder(results, mode, part='wf'):
    rows = []
    for k, r in results.items():
        m = dict(r[mode][part])
        m.update({f'model_{a}': b for a, b in (r.get('model') or {}).items()})
        m['variant'] = k
        m['signals'] = r['n_signals']
        rows.append(m)
    df = pd.DataFrame(rows).set_index('variant')
    return df


def random_summary(rs, ref_metrics):
    """rs: DataFrame of random runs; ref_metrics: dict for the variant. Returns percentile table."""
    out = []
    for k in ['expectancy', 'CAGR', 'Sharpe', 'Calmar', 'MaxDD', 'final_multiple']:
        v = ref_metrics.get(k)
        x = rs[k].dropna().values
        out.append({'metric': k, 'variant_value': v, 'rand_mean': x.mean(), 'rand_p5': np.percentile(x, 5), 'rand_p50': np.percentile(x, 50),
                    'rand_p95': np.percentile(x, 95), 'percentile_of_variant': float((x < v).mean() * 100) if v is not None else np.nan,
                    'beats_p95': bool(v > np.percentile(x, 95)) if v is not None else None})
    return pd.DataFrame(out)
