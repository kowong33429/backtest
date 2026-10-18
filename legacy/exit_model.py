"""
exit_model.py — Dedicated EXIT model for long trades opened by the BUY model.

Why not just reuse the short model?  The short model answers "will price halve
before rising 67%?" over EVERY bar of every coin. An exit is a different, easier
question asked only while we are already long, typically inside a strong uptrend:
"from here, does price fall `down` before it rises `up`?". Training on exactly
those in-trade bars teaches the model the rally-vs-top distinction the short model
never sees enough of.

PIPELINE
  1. In-trade bars: replay the BUY model's OOF signals (prob >= --entry-thr) with a
     HOLD policy (hard SL -40%, max --hold-days, one position per coin). Every bar
     j strictly inside a trade is a training row (a decision "exit at Open[j]?").
  2. Label (per bar, from Open[j]): 1 if Low falls to Open*(1-down) BEFORE High
     reaches Open*(1+up) within --horizon-days (same-bar tie -> 1, conservative).
     Label_EndDate = the bar where it resolved (used for purging).
  3. Features (row j sees bars <= j-1 only): the 108 panel features (already
     .shift(1) in features.py) + the short model's OOF prob + the buy model's OOF
     prob on that row (both out-of-sample, produced by time-ordered CV).
     No trade-specific features, so the prob is defined on EVERY bar and plugs
     straight into exits.py (the trade state is handled there via --min-dd).
  4. Purged + embargoed walk-forward CV (train_entry_model.purged_walk_forward).
     Train on in-trade rows of the train window; predict ALL rows of the test
     window. Output: <out-dir>/exit_oof.parquet (Date, symbol, oof_prob_xgb) —
     use it with --exit-mode model --exit-probs ... in entry_backtest.py /
     portfolio_backtest.py. A final model on all in-trade rows is saved for live.

Usage:
    python core/exit_model.py
    python core/exit_model.py --down 0.30 --up 0.50 --horizon-days 180 --out-dir data/model/exit_v1
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_entry_model import purged_walk_forward  # noqa: E402

try:
    from numba import njit
except ImportError:          # pure-python fallback (slow but correct)
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _race_down_first(o, hi, lo, down, up, H):
    """label[i]=1 if, from Open[i], price drops `down` before it rises `up` within H bars."""
    n = len(o)
    lab = np.zeros(n, np.int8)
    end_idx = np.arange(n)
    for i in range(n):
        e = o[i]
        if not (e > 0):
            continue
        dn, upl = e * (1 - down), e * (1 + up)
        last = min(i + H, n - 1)
        end_idx[i] = last
        for j in range(i, last + 1):      # bar i itself counts: we are in the market from Open[i]
            if lo[j] <= dn:
                lab[i] = 1
                end_idx[i] = j
                break
            if hi[j] >= upl:
                end_idx[i] = j
                break
    return lab, end_idx


def in_trade_mask(o, lo, sig_pos, sl, hold_bars):
    """Bars strictly inside HOLD-policy trades (entry Open[s+1], hard SL, max hold)."""
    n = len(o)
    mask = np.zeros(n, bool)
    until = -1
    for s in sig_pos:
        e = s + 1
        if s <= until or e >= n:
            continue
        stop = o[e] * (1 - sl)
        x = min(e + hold_bars, n - 1)
        for j in range(e, x + 1):
            if lo[j] <= stop:
                x = j
                break
        mask[e + 1:x + 1] = True      # decisions "exit at Open[j]" for j after the entry bar
        until = x
    return mask


def main():
    ap = argparse.ArgumentParser(description="Train a dedicated exit model for buy-model trades")
    ap.add_argument('--panel', default='data/model/entry_panel_short.parquet',
                    help='feature panel (same 108 features as the buy panel)')
    ap.add_argument('--buy-oof', default='data/model/v2/m1_100pct_60d/entry_oof.parquet')
    ap.add_argument('--short-oof', default='data/model/short_v2/entry_oof.parquet')
    ap.add_argument('--entry-thr', type=float, default=0.7)
    ap.add_argument('--sl', type=float, default=0.40)
    ap.add_argument('--hold-days', type=int, default=365)
    ap.add_argument('--down', type=float, default=0.30)
    ap.add_argument('--up', type=float, default=0.50)
    ap.add_argument('--horizon-days', type=int, default=180)
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--embargo-days', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='data/model/exit_v1')
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    P = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    out_dir = P(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    bpd = 6  # 4H bars per day

    print("=" * 72)
    print(f"  EXIT MODEL | label: -{args.down*100:.0f}% before +{args.up*100:.0f}% "
          f"within {args.horizon_days}d | in-trade = buy prob >= {args.entry_thr}")
    print("=" * 72)
    panel = pd.read_parquet(P(args.panel))
    panel['Date'] = pd.to_datetime(panel['Date'])
    non = {'Date', 'symbol', 'Entry_Label', 'Label_EndDate', 'Sample_Weight'}
    feats = [c for c in panel.columns if c not in non]
    panel = panel[['Date', 'symbol'] + feats]

    def oof(path, name):
        d = pd.read_parquet(P(path), columns=['Date', 'symbol', 'oof_prob_xgb'])
        d['Date'] = pd.to_datetime(d['Date'])
        return d.rename(columns={'oof_prob_xgb': name})
    buy, short = oof(args.buy_oof, 'Buy_Prob'), oof(args.short_oof, 'Short_Prob')
    panel = panel.merge(buy, on=['Date', 'symbol'], how='left').merge(short, on=['Date', 'symbol'], how='left')
    # Both stacked probs must be out-of-sample on every row we use.
    start = max(buy.dropna().Date.min(), short.dropna().Date.min())
    panel = panel[panel['Date'] >= start]

    csvs = {os.path.basename(c).replace('_4h_full.csv', '').upper(): c
            for c in glob.glob(os.path.join(base, 'data', '*', '*_4h_full.csv'))}
    parts = []
    for sym, g in panel.groupby('symbol'):
        if sym not in csvs:
            continue
        px = pd.read_csv(csvs[sym], usecols=['Date', 'Open', 'High', 'Low'])
        px['Date'] = pd.to_datetime(px['Date'])
        px = px.sort_values('Date').reset_index(drop=True)
        o, hi, lo = (px[c].values.astype(float) for c in ('Open', 'High', 'Low'))
        lab, end = _race_down_first(o, hi, lo, args.down, args.up, args.horizon_days * bpd)
        pos = {d: i for i, d in enumerate(px['Date'].values)}
        sig = g.loc[g['Buy_Prob'] >= args.entry_thr, 'Date'].values
        mask = in_trade_mask(o, lo, sorted(pos[d] for d in sig if d in pos), args.sl, args.hold_days * bpd)
        px['Exit_Label'] = lab
        px['Label_EndDate'] = px['Date'].values[end]
        px['In_Trade'] = mask
        parts.append(g.merge(px[['Date', 'Exit_Label', 'Label_EndDate', 'In_Trade']], on='Date', how='inner'))
    df = pd.concat(parts, ignore_index=True).sort_values(['Date', 'symbol']).reset_index(drop=True)

    X_cols = feats + ['Buy_Prob', 'Short_Prob']
    X = df[X_cols].astype(float).fillna(0.0)
    y = df['Exit_Label'].astype(int).values
    it = df['In_Trade'].values
    print(f"[Data] {len(df)} bars | in-trade {it.sum()} ({it.mean()*100:.1f}%) | "
          f"exit-label rate in-trade {y[it].mean()*100:.1f}% (all bars {y.mean()*100:.1f}%)")

    from xgboost import XGBClassifier
    from sklearn.metrics import average_precision_score
    prm = dict(n_estimators=600, learning_rate=0.03, max_depth=5, min_child_weight=50,
               subsample=0.8, colsample_bytree=0.5, random_state=args.seed,
               n_jobs=4, verbosity=0, eval_metric='aucpr')
    oofp = np.full(len(df), np.nan)
    folds = list(purged_walk_forward(df['Date'].values, df['Label_EndDate'].values,
                                     args.n_splits, args.embargo_days))
    for k, (tr, te) in enumerate(folds, 1):
        tr = tr[it[tr]]                       # learn only from in-trade decisions
        if len(tr) == 0 or y[tr].sum() == 0:
            print(f"  fold {k}: skipped (no in-trade rows)")
            continue
        m = XGBClassifier(**prm)
        m.fit(X.iloc[tr], y[tr])
        oofp[te] = m.predict_proba(X.iloc[te])[:, 1]
        te_it = te[it[te]]
        ap_ = average_precision_score(y[te_it], oofp[te_it]) if y[te_it].sum() else float('nan')
        print(f"  fold {k}: train {len(tr):>7} in-trade | test {len(te):>7} "
              f"[{pd.Timestamp(df['Date'].values[te].min()).date()}..{pd.Timestamp(df['Date'].values[te].max()).date()}] "
              f"| in-trade PR-AUC {ap_:.3f} (base {y[te_it].mean():.3f})")

    ok = ~np.isnan(oofp) & it
    print(f"\n  POOLED in-trade OOS PR-AUC {average_precision_score(y[ok], oofp[ok]):.3f} "
          f"(base {y[ok].mean():.3f}) | OOF from {df.loc[~np.isnan(oofp), 'Date'].min().date()}")
    out = df[['Date', 'symbol', 'Exit_Label', 'In_Trade']].copy()
    out['oof_prob_xgb'] = oofp
    out.to_parquet(os.path.join(out_dir, 'exit_oof.parquet'), index=False)

    final = XGBClassifier(**prm)
    final.fit(X[it], y[it])
    final.save_model(os.path.join(out_dir, 'exit_xgb.json'))
    with open(os.path.join(out_dir, 'exit_model_meta.json'), 'w') as f:
        json.dump({'features': X_cols, 'down': args.down, 'up': args.up,
                   'horizon_days': args.horizon_days, 'entry_thr': args.entry_thr,
                   'sl': args.sl, 'hold_days': args.hold_days, 'buy_oof': args.buy_oof,
                   'short_oof': args.short_oof, 'xgb_params': prm}, f, indent=2)
    print(f"  Saved -> {out_dir}\\exit_oof.parquet, exit_xgb.json, exit_model_meta.json")
    print("=" * 72)


if __name__ == '__main__':
    main()
