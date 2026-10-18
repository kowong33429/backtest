"""Builds reports/legacy_vs_benchmarks_<date>.xlsx + charts from cached results (dev + walk-forward only)."""
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.report_tables import load, ladder, random_summary, COLS
from common_eval.config import load_config

DATE = '20261018'
cfg = load_config()
LR, SR, BR = load('long_results.pkl'), load('short_results.pkl'), load('bench_results.pkl')
BRIDGE = pd.read_pickle(os.path.join(ROOT, 'common_eval', '.cache', 'bridge.pkl'))
WF0 = pd.Timestamp(cfg['time']['WF_START'])
os.makedirs(os.path.join(ROOT, 'reports', 'charts'), exist_ok=True)

FIXED_L, FIXED_S = 'L4_+M4_keep_NaN', 'S4_+M4 keep NaN'


def rebased(eq):
    e = eq[eq.index >= WF0]
    return e / e.iloc[0]


# ---------------- charts ----------------
def equity_chart(mode, fname):
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    series = [('LEGACY_AS_IS (long)', LR['L0_AS_IS'][mode]['eq']), ('+H1 universe', LR['L1_+H1_universe'][mode]['eq']),
              ('LEGACY_FIXED long', LR[FIXED_L][mode]['eq']), ('SHORT as-is (reciprocal bug)', SR['S0_AS_IS (reciprocal P&L)'][mode]['eq']),
              ('LEGACY_FIXED short', SR[FIXED_S][mode]['eq'])]
    for n, e in series:
        e = rebased(e)
        ax[0].plot(e.index, e.values, label=n, lw=1.4)
        ax[1].plot(e.index, (e / e.cummax() - 1).values, lw=1)
    for n in ('BTC_HOLD', 'EQUAL_WEIGHT'):
        e = rebased(BR[n]['eq'])
        ax[0].plot(e.index, e.values, '--', label=n, lw=1.4, color='k' if n == 'BTC_HOLD' else 'gray')
        ax[1].plot(e.index, (e / e.cummax() - 1).values, '--', lw=1, color='k' if n == 'BTC_HOLD' else 'gray')
    ax[0].set_yscale('log'); ax[0].set_title(f'Equity (log), walk-forward 2022-01 -> 2025-06, exit = {mode}'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].set_title('Drawdown'); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, 'reports', 'charts', fname), dpi=120); plt.close()


def random_hist(name, fixed_rec, fname):
    fig, axs = plt.subplots(2, 4, figsize=(18, 7))
    for r, metric in enumerate(['CAGR', 'expectancy']):
        for c, (mode, ex) in enumerate([('rate', 'native'), ('rate', 'unified'), ('timing', 'native'), ('timing', 'unified')]):
            rs = BR[name][(mode, ex)][metric].dropna()
            ax = axs[r, c]
            ax.hist(rs, bins=40, color='lightgray', edgecolor='gray')
            v = fixed_rec[ex]['wf'][metric]
            ax.axvline(v, color='red', lw=2, label=f'FIXED {v:.3f}')
            ax.axvline(np.percentile(rs, 95), color='blue', ls='--', label='P95')
            ax.set_title(f'{metric} | RANDOM {mode}-matched | {ex}', fontsize=9); ax.legend(fontsize=7)
    plt.suptitle(f'{name}: 1,000 RANDOM runs vs the variant'); plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'reports', 'charts', fname), dpi=110); plt.close()


for mode in ('native', 'unified'):
    equity_chart(mode, f'equity_drawdown_{mode}_{DATE}.png')
random_hist('LEGACY_FIXED_LONG', LR[FIXED_L], f'random_hist_long_{DATE}.png')
random_hist('LEGACY_FIXED_SHORT', SR[FIXED_S], f'random_hist_short_{DATE}.png')

# ---------------- workbook ----------------
wb = Workbook(); wb.remove(wb.active)


def sheet(name, df, index=True, note=None):
    ws = wb.create_sheet(name[:31])
    d = df.reset_index() if index else df
    ws.append(list(map(str, d.columns)))
    for row in d.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, 'item') else (str(v) if isinstance(v, (pd.Timestamp, np.datetime64)) else v)) for v in row])
    for c in ws[1]:
        c.font = Font(bold=True, color='FFFFFF'); c.fill = PatternFill('solid', fgColor='1F3864'); c.alignment = Alignment(wrap_text=True, vertical='center')
    ws.freeze_panes = 'A2'; ws.auto_filter.ref = ws.dimensions
    for i, h in enumerate(d.columns, 1):
        w = max([len(str(h))] + [len(str(x)) for x in d.iloc[:200, i - 1].tolist()]) + 2
        ws.column_dimensions[get_column_letter(i)].width = min(max(w, 10), 48)
    return ws


def pick(df, extra=()):
    keep = [c for c in list(COLS) + list(extra) if c in df.columns]
    return df[keep]


model_cols = ['signals', 'model_n', 'model_base_rate', 'model_pr_auc', 'model_roc_auc', 'model_precision@1%', 'model_lift@1%', 'model_precision@5%', 'model_lift@5%', 'model_precision@10%', 'model_lift@10%']
cfg_rows = []
def flat(d, p=''):
    for k, v in d.items():
        if isinstance(v, dict):
            flat(v, p + k + '.')
        else:
            cfg_rows.append({'key': p + k, 'value': str(v)})
flat(cfg)
sheet('config', pd.DataFrame(cfg_rows), index=False)

for mode in ('native', 'unified'):
    sheet(f'long_ladder_{mode}', pick(ladder(LR, mode), model_cols))
    sheet(f'long_postdev_{mode}', pick(ladder(LR, mode, 'post_dev')))
    sheet(f'short_ladder_{mode}', pick(ladder(SR, mode), model_cols))
sheet('bridge_AS_IS', BRIDGE, index=False)

bench_rows = []
for n in ('BTC_HOLD', 'EQUAL_WEIGHT'):
    m = dict(BR[n]['wf']); m['variant'] = n; bench_rows.append(m)
for lab, R, k in [('LEGACY_FIXED_LONG', LR, FIXED_L), ('LEGACY_FIXED_SHORT', SR, FIXED_S)]:
    for mode in ('native', 'unified'):
        m = dict(R[k][mode]['wf']); m['variant'] = f'{lab} [{mode}]'; bench_rows.append(m)
bm = pd.DataFrame(bench_rows).set_index('variant')
sheet('benchmarks_vs_fixed', pick(bm))

rand_rows, rand_dist = [], []
for name, R, k in [('LEGACY_FIXED_LONG', LR, FIXED_L), ('LEGACY_FIXED_SHORT', SR, FIXED_S)]:
    for (mode, ex), rs in BR[name].items():
        t = random_summary(rs, R[k][ex]['wf'])
        t.insert(0, 'exit', ex); t.insert(0, 'random_mode', mode); t.insert(0, 'variant', name)
        t['random_mean_trades'] = rs['trades'].mean(); t['variant_trades'] = R[k][ex]['wf']['trades']
        rand_rows.append(t)
        d = rs.copy(); d.insert(0, 'exit', ex); d.insert(0, 'random_mode', mode); d.insert(0, 'variant', name); rand_dist.append(d)
RS = pd.concat(rand_rows, ignore_index=True)
sheet('random_vs_fixed', RS, index=False)
sheet('random_runs', pd.concat(rand_dist, ignore_index=True), index=False)

yr = {}
for k, r in list(LR.items()) + list(SR.items()):
    for mode in ('native', 'unified'):
        yr[f'{k} [{mode}]'] = r[mode]['yearly']
for n in ('BTC_HOLD', 'EQUAL_WEIGHT'):
    e = rebased(BR[n]['eq']); d = e.resample('D').last().dropna(); ye = d.resample('YE').last()
    prev = pd.concat([pd.Series([1.0]), ye.iloc[:-1].reset_index(drop=True)]).values
    yr[n] = pd.Series(ye.values / prev - 1, index=ye.index.year)
sheet('yearly', pd.DataFrame(yr).T)
sheet('ex_2022_from_2023', pd.read_pickle(os.path.join(ROOT, 'common_eval', '.cache', 'ex2022.pkl')), index=False)

for k in (FIXED_L, 'L0_AS_IS'):
    for mode in ('native', 'unified'):
        sheet(f'trades_{k[:2]}_{mode}', LR[k][mode]['td'], index=False)
for mode in ('native',):
    sheet('trades_S4_native', SR[FIXED_S][mode]['td'], index=False)

out = os.path.join(ROOT, 'reports', f'legacy_vs_benchmarks_{DATE}.xlsx')
wb.save(out)
RS.to_pickle(os.path.join(ROOT, 'common_eval', '.cache', 'random_summary.pkl'))
print('saved', out)
