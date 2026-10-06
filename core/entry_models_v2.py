"""
entry_models_v2.py — Baseline entry-point model orchestrator (m1 +100%/60d)

WHY THIS EXISTS
---------------
The BASELINE entry model, built with the ORIGINAL leakage-safe pipeline left
untouched:

  * Model 1 ("m1_100pct_60d") : entry points that run to +100% within 2 months.

(The project also trialled a +50%/30d "m2" and a daily rebuild; those lost on
total P&L and were removed in the cleanup — see the README experiment tables.)

HOW (and why it's a thin orchestrator, not a fork)
--------------------------------------------------
An "entry model" in this repo is defined ENTIRELY by its forward label
(labeler.py: `+tp% within horizon days before -sl%`). The leakage-safe,
audited pipeline — dataset.py -> train_entry_model.py -> entry_backtest.py —
is already fully parameterized by (tp, sl, horizon_days). Re-implementing the
labeling / purged-CV here would risk introducing the exact look-ahead bias
AGENTS.md forbids. So this driver does NOT copy that logic: it runs the same
battle-tested scripts with each model's parameters, writing every artifact into
its OWN folder under data/model/v2/<model>/ so nothing collides with the
existing +100%/30d/-40% model in data/model/.

Stop-loss (user decision: keep reward:risk ~2.5:1 per model):
  * Model 1: +100% target -> -40% stop  (2.5 : 1)
  * Model 2:  +50% target -> -20% stop  (2.5 : 1)

CV settings mirror the original model (n_splits=6, embargo=3d) so the only thing
that differs between the three models is the entry definition — a fair compare.

Usage
-----
  python core/entry_models_v2.py                     # build+train+backtest both, then compare
  python core/entry_models_v2.py --models m1_100pct_60d
  python core/entry_models_v2.py --stage compare     # just re-print the comparison
  python core/entry_models_v2.py --force             # rebuild even if artifacts exist

Each stage is skipped when its output already exists (resumable); --force
rebuilds. The panel build is the slow part (forward labeling over 100+ coins).
"""
import os
import sys
import json
import argparse
import subprocess

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(BASE_DIR, 'core')
PY = sys.executable  # the same interpreter that launched this driver (venv-safe)

# ---------------------------------------------------------------------------
# Model definitions — the ONLY thing that differs between models.
# ---------------------------------------------------------------------------
MODELS = {
    'm1_100pct_60d': {
        'tp': 1.00, 'sl': 0.40, 'horizon_days': 60,
        'desc': '+100% within 60d (2 months), stop -40%  [R:R 2.5]',
    },
}

OUT_ROOT = os.path.join(BASE_DIR, 'data', 'model', 'v2')


# ---------------------------------------------------------------------------
# Stage runners (shell out to the existing, audited pipeline)
# ---------------------------------------------------------------------------
def _run(cmd, title):
    print("\n" + "#" * 78)
    print(f"#  {title}")
    print(f"#  $ {' '.join(str(c) for c in cmd)}")
    print("#" * 78, flush=True)
    subprocess.run(cmd, check=True, cwd=BASE_DIR)


def stage_panel(name, cfg, model_dir, force):
    panel = os.path.join(model_dir, 'entry_panel.parquet')
    if os.path.exists(panel) and not force:
        print(f"[{name}] panel exists -> skip  ({panel})")
        return panel
    _run([PY, os.path.join(CORE_DIR, 'dataset.py'),
          '--tp', str(cfg['tp']), '--sl', str(cfg['sl']),
          '--horizon-days', str(cfg['horizon_days']),
          '--out', panel],
         f"{name}: BUILD PANEL  (+{cfg['tp']*100:.0f}%/{cfg['horizon_days']}d/-{cfg['sl']*100:.0f}%)")
    return panel


def stage_train(name, model_dir, panel, n_splits, embargo, force):
    oof = os.path.join(model_dir, 'entry_oof.parquet')
    if os.path.exists(oof) and not force:
        print(f"[{name}] oof exists -> skip  ({oof})")
        return oof
    _run([PY, os.path.join(CORE_DIR, 'train_entry_model.py'),
          '--panel', panel, '--out-dir', model_dir,
          '--n-splits', str(n_splits), '--embargo-days', str(embargo)],
         f"{name}: TRAIN (purged walk-forward)")
    return oof


def stage_backtest(name, cfg, model_dir, oof, notional, force):
    trades = os.path.join(model_dir, 'entry_backtest_trades.csv')
    if os.path.exists(trades) and not force:
        print(f"[{name}] backtest exists -> skip  ({trades})")
        return trades
    # Use the threshold the trainer selected (precision-oriented, recall>=5%).
    meta_path = os.path.join(model_dir, 'entry_model_meta.json')
    thr = 0.70
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            thr = json.load(f).get('threshold', 0.70)
    _run([PY, os.path.join(CORE_DIR, 'entry_backtest.py'),
          '--oof', oof, '--threshold', str(thr),
          '--tp', str(cfg['tp']), '--sl', str(cfg['sl']),
          '--horizon-days', str(cfg['horizon_days']),
          '--notional', str(notional), '--out-dir', model_dir],
         f"{name}: BACKTEST  (threshold {thr})")
    return trades


# ---------------------------------------------------------------------------
# Summary (mirrors entry_backtest.py's metric definitions, read from the CSV)
# ---------------------------------------------------------------------------
def summarize(name, cfg, model_dir, notional):
    trades_csv = os.path.join(model_dir, 'entry_backtest_trades.csv')
    meta_path = os.path.join(model_dir, 'entry_model_meta.json')
    row = {'model': name, 'label': cfg['desc'],
           'tp_pct': cfg['tp'] * 100, 'sl_pct': cfg['sl'] * 100,
           'horizon_days': cfg['horizon_days']}

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        row.update({
            'threshold': meta.get('threshold'),
            'pr_auc_xgb': meta.get('pr_auc_xgb'),
            'pr_auc_lr': meta.get('pr_auc_lr'),
            'base_rate_pct': (meta.get('base_rate') or 0) * 100,  # % of bars that are positives
        })

    if not os.path.exists(trades_csv):
        row['trades'] = 0
        return row
    td = pd.read_csv(trades_csv)
    if td.empty:
        row['trades'] = 0
        return row

    n = len(td)
    wins = td[td['net_ret'] > 0]
    losses = td[td['net_ret'] <= 0]
    gross_win = wins['net_ret'].sum()
    gross_loss = -losses['net_ret'].sum()
    oc = td['outcome'].value_counts().to_dict()
    bpd = 6  # 4H bars/day (same inference as the rest of the repo)
    row.update({
        'trades': n,
        'tp_hits': int(oc.get('TP', 0)),
        'sl_hits': int(oc.get('SL', 0)),
        'time_hits': int(oc.get('TIME', 0)),
        'win_rate_pct': len(wins) / n * 100,
        'avg_win_pct': (wins['net_ret'].mean() * 100) if len(wins) else float('nan'),
        'avg_loss_pct': (losses['net_ret'].mean() * 100) if len(losses) else float('nan'),
        'expectancy_pct': td['net_ret'].mean() * 100,
        'profit_factor': (gross_win / gross_loss) if gross_loss > 0 else float('inf'),
        'avg_days_held': td['bars_held'].mean() / bpd,
        'total_pnl_usd': td['pnl_usd'].sum(),
        'return_on_deployed_pct': td['pnl_usd'].sum() / (n * notional) * 100,
    })
    return row


def _fmt(v, spec=''):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    try:
        return format(v, spec) if spec else str(v)
    except (ValueError, TypeError):
        return str(v)


def compare(names, notional):
    rows = [summarize(n, MODELS[n], os.path.join(OUT_ROOT, n), notional) for n in names]
    cmp_df = pd.DataFrame(rows)
    csv_out = os.path.join(OUT_ROOT, 'comparison.csv')
    cmp_df.to_csv(csv_out, index=False, encoding='utf-8-sig')

    # Human-readable side-by-side (metrics as rows, models as columns).
    metric_rows = [
        ('Entry definition', 'label', 's'),
        ('Take profit', 'tp_pct', '.0f'),
        ('Stop loss', 'sl_pct', '.0f'),
        ('Horizon (days)', 'horizon_days', 'd'),
        ('Signal threshold', 'threshold', '.2f'),
        ('PR-AUC (XGB, OOS)', 'pr_auc_xgb', '.3f'),
        ('PR-AUC (Logistic)', 'pr_auc_lr', '.3f'),
        ('Positive base rate %', 'base_rate_pct', '.2f'),
        ('— — —', None, None),
        ('Trades', 'trades', 'd'),
        ('TP / SL / TIME', '_oc', 's'),
        ('Win rate %', 'win_rate_pct', '.1f'),
        ('Avg win %', 'avg_win_pct', '+.1f'),
        ('Avg loss %', 'avg_loss_pct', '+.1f'),
        ('Expectancy %/trade', 'expectancy_pct', '+.2f'),
        ('Profit factor', 'profit_factor', '.2f'),
        ('Avg days held', 'avg_days_held', '.0f'),
        (f'Total P&L (${notional:.0f}/trade)', 'total_pnl_usd', '+,.0f'),
        ('Return on deployed %', 'return_on_deployed_pct', '+.1f'),
    ]
    by_model = {r['model']: r for r in rows}

    lines = []
    lines.append("# Entry Models v2 — Comparison\n")
    lines.append("Original model (`data/model/`, untouched): **+100% / 30d / -40%**.\n")
    lines.append("Both v2 models use the same purged walk-forward CV as the original "
                 "(n_splits=6, embargo=3d); only the entry definition differs.\n")
    header = ['Metric'] + list(names)
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
    for title, key, spec in metric_rows:
        if key is None:
            lines.append('| ' + ' | '.join([title] + [''] * len(names)) + ' |')
            continue
        cells = [title]
        for nm in names:
            r = by_model[nm]
            if key == '_oc':
                val = f"{r.get('tp_hits','—')} / {r.get('sl_hits','—')} / {r.get('time_hits','—')}"
            else:
                val = _fmt(r.get(key), spec)
            cells.append(val)
        lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |')
    md = '\n'.join(lines) + '\n'

    md_out = os.path.join(OUT_ROOT, 'comparison.md')
    with open(md_out, 'w', encoding='utf-8') as f:
        f.write(md)

    print("\n" + "=" * 78)
    print("  MODEL COMPARISON")
    print("=" * 78)
    print(md)
    print(f"  Comparison table : {md_out}")
    print(f"  Comparison CSV   : {csv_out}")
    for nm in names:
        print(f"  {nm} artifacts  : {os.path.join(OUT_ROOT, nm)}/")
    print("=" * 78)
    return cmp_df


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build/train/backtest the two v2 entry models and compare")
    ap.add_argument('--models', nargs='*', default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument('--stage', default='all',
                    choices=['all', 'panel', 'train', 'backtest', 'compare'])
    ap.add_argument('--n-splits', type=int, default=6)
    ap.add_argument('--embargo-days', type=int, default=3)
    ap.add_argument('--notional', type=float, default=100.0)
    ap.add_argument('--force', action='store_true', help='rebuild even if artifacts exist')
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    print("=" * 78)
    print("  ENTRY MODELS v2  |  building:", ', '.join(args.models))
    print("=" * 78)
    for nm in args.models:
        print(f"  {nm}: {MODELS[nm]['desc']}")

    if args.stage != 'compare':
        for nm in args.models:
            cfg = MODELS[nm]
            model_dir = os.path.join(OUT_ROOT, nm)
            os.makedirs(model_dir, exist_ok=True)

            panel = os.path.join(model_dir, 'entry_panel.parquet')
            oof = os.path.join(model_dir, 'entry_oof.parquet')

            if args.stage in ('all', 'panel'):
                panel = stage_panel(nm, cfg, model_dir, args.force)
            if args.stage in ('all', 'train'):
                oof = stage_train(nm, model_dir, panel, args.n_splits, args.embargo_days, args.force)
            if args.stage in ('all', 'backtest'):
                stage_backtest(nm, cfg, model_dir, oof, args.notional, args.force)

    # Always finish with the comparison (uses whatever artifacts are present).
    compare(args.models, args.notional)


if __name__ == '__main__':
    main()
