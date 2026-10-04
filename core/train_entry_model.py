"""
train_entry_model.py — Pooled Entry-Point classifier with PURGED walk-forward CV

Trains on the stacked panel from dataset.py (many ZEC-like coins) to predict, for
every 4H bar, P(a +100%/30d rally starts here) using ONLY strict n-1 features.

WHY PURGED + EMBARGOED WALK-FORWARD (the part people get wrong)
--------------------------------------------------------------
The label at bar t peeks forward up to `horizon` days (its Label_EndDate). A
plain TimeSeriesSplit (as in optimizer.py) therefore LEAKS at the train/test
boundary: training bars near the split still "know" what happens inside the test
window. We fix it (López de Prado, AFML ch. 7):

  * TEST  = a contiguous FUTURE time block (all coins share the timeline, so the
            test period is strictly future for every coin — no cross-coin leak).
  * PURGE = drop any TRAIN bar whose Label_EndDate falls on/after the test start
            (its outcome overlaps the test period).
  * EMBARGO = additionally drop TRAIN bars within `embargo_days` just before the
            test start, to kill residual serial correlation.

Metrics are reported OUT-OF-SAMPLE only, and PR-AUC (average precision) is the
headline — ROC-AUC flatters imbalanced problems. The LogisticRegression baseline
is the bar XGBoost must beat OOS; if it can't, XGBoost is overfitting and we say so.
"""
import os
import sys
import argparse
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------
# Purged + embargoed walk-forward folds (time-based, cross-coin safe)
# ----------------------------------------------------------------------------
def purged_walk_forward(dates, label_end, n_splits=5, embargo_days=3):
    """
    Yield (train_pos, test_pos) positional index arrays.

    dates      : np.datetime64 array, the entry Date of each row (sorted asc).
    label_end  : np.datetime64 array, each row's Label_EndDate.
    Expanding train window; test blocks tile the back portion of the timeline.
    """
    dates = pd.to_datetime(dates).values
    label_end = pd.to_datetime(label_end).values
    order = np.argsort(dates, kind='stable')
    dmin, dmax = dates[order[0]], dates[order[-1]]

    # Split the timeline into n_splits+1 equal spans; folds 1..n_splits are tests.
    edges = pd.date_range(pd.Timestamp(dmin), pd.Timestamp(dmax), periods=n_splits + 2)
    embargo = pd.Timedelta(days=embargo_days)

    for k in range(1, n_splits + 1):
        test_start = edges[k]
        test_end = edges[k + 1]
        test_mask = (dates >= np.datetime64(test_start)) & (dates < np.datetime64(test_end))
        # Train = everything before test_start, PURGED and EMBARGOED.
        train_mask = dates < np.datetime64(test_start - embargo)
        # Purge: a train bar whose label resolves on/after test_start overlaps test.
        train_mask &= label_end < np.datetime64(test_start)
        train_pos = np.where(train_mask)[0]
        test_pos = np.where(test_mask)[0]
        if len(train_pos) == 0 or len(test_pos) == 0:
            continue
        yield train_pos, test_pos


def drop_correlated(X, threshold=0.75):
    """AGENTS.md Rule #4 — drop one of each >threshold-correlated pair."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return X.drop(columns=to_drop), to_drop


def pr_report(y_true, y_prob, base_rate):
    """Precision/recall/F1 at a few thresholds + PR-AUC + lift over base rate."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    rows = []
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = y_prob >= t
        if pred.sum() == 0:
            rows.append((t, float('nan'), 0.0, float('nan'), 0))
            continue
        p = (y_true[pred] == 1).mean()
        r = (y_true[pred] == 1).sum() / max(1, (y_true == 1).sum())
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append((t, p, r, p / base_rate if base_rate > 0 else float('nan'), int(pred.sum())))
    return ap, rows


def main():
    ap = argparse.ArgumentParser(description="Train pooled entry model (purged walk-forward)")
    ap.add_argument('--panel', default='data/model/entry_panel.parquet')
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--embargo-days', type=int, default=3)
    ap.add_argument('--corr-filter', action='store_true',
                    help='Drop >0.75-correlated features (AGENTS.md Rule #4)')
    ap.add_argument('--no-shap', action='store_true')
    ap.add_argument('--out-dir', default='data/model')
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    panel_path = os.path.join(base_dir, args.panel) if not os.path.isabs(args.panel) else args.panel
    if not os.path.exists(panel_path):
        panel_path = panel_path.replace('.parquet', '.csv')
    out_dir = os.path.join(base_dir, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("  ENTRY MODEL — pooled, purged walk-forward")
    print("=" * 72)
    panel = (pd.read_parquet(panel_path) if panel_path.endswith('.parquet')
             else pd.read_csv(panel_path, parse_dates=['Date', 'Label_EndDate']))
    panel = panel.sort_values(['Date', 'symbol']).reset_index(drop=True)

    non_feat = {'Date', 'symbol', 'Entry_Label', 'Label_EndDate', 'Sample_Weight'}
    feat_cols = [c for c in panel.columns if c not in non_feat]
    X = panel[feat_cols].astype(float).fillna(0.0)
    y = panel['Entry_Label'].astype(int).values
    w = panel['Sample_Weight'].astype(float).values
    dates = panel['Date'].values
    lend = panel['Label_EndDate'].values
    base_rate = y.mean()
    print(f"[Data] {len(panel)} rows | {panel['symbol'].nunique()} coins | "
          f"{len(feat_cols)} features | positives {y.sum()} ({base_rate*100:.2f}%)")

    if args.corr_filter:
        X, dropped = drop_correlated(X)
        feat_cols = list(X.columns)
        print(f"[Filter] dropped {len(dropped)} correlated features -> {len(feat_cols)} kept")

    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scale_pos = max(1.0, (len(y) - y.sum()) / max(1, y.sum()))
    oof_xgb = np.full(len(y), np.nan)
    oof_lr = np.full(len(y), np.nan)
    fold_rows = []
    last_model, last_Xtest = None, None

    folds = list(purged_walk_forward(dates, lend, args.n_splits, args.embargo_days))
    print(f"[CV] {len(folds)} purged walk-forward folds (embargo {args.embargo_days}d)\n")
    for k, (tr, te) in enumerate(folds, 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        wtr = w[tr]
        if ytr.sum() == 0 or yte.sum() == 0:
            print(f"  fold {k}: skipped (no positives in train/test)")
            continue

        # --- XGBoost ---
        xgb = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=scale_pos, random_state=42,
                            n_jobs=4, verbosity=0, eval_metric='aucpr')
        xgb.fit(Xtr, ytr, sample_weight=wtr)
        oof_xgb[te] = xgb.predict_proba(Xte)[:, 1]

        # --- Logistic baseline (standardized) ---
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)
        lr.fit(sc.transform(Xtr), ytr, sample_weight=wtr)
        oof_lr[te] = lr.predict_proba(sc.transform(Xte))[:, 1]

        from sklearn.metrics import average_precision_score
        ap_x = average_precision_score(yte, oof_xgb[te])
        ap_l = average_precision_score(yte, oof_lr[te])
        tr_span = f"{pd.Timestamp(dates[tr].min()).date()}..{pd.Timestamp(dates[tr].max()).date()}"
        te_span = f"{pd.Timestamp(dates[te].min()).date()}..{pd.Timestamp(dates[te].max()).date()}"
        print(f"  fold {k}: train {len(tr):>6} [{tr_span}] | test {len(te):>5} [{te_span}] "
              f"pos {yte.sum():>4} | PR-AUC  xgb {ap_x:.3f}  lr {ap_l:.3f}")
        fold_rows.append((k, ap_x, ap_l))
        last_model, last_Xtest = xgb, Xte

    # --- Pooled OOF report ---
    mask = ~np.isnan(oof_xgb)
    yv = y[mask]
    print("\n" + "-" * 72)
    print(f"POOLED OUT-OF-SAMPLE ({mask.sum()} bars, base rate {yv.mean()*100:.2f}%)")
    ap_x, rows_x = pr_report(yv, oof_xgb[mask], yv.mean())
    ap_l, rows_l = pr_report(yv, oof_lr[mask], yv.mean())
    print(f"  PR-AUC:  XGBoost {ap_x:.3f}   |   Logistic baseline {ap_l:.3f}   "
          f"(random = {yv.mean():.3f})")
    print(f"\n  XGBoost precision/recall/lift by threshold:")
    print(f"  {'thr':>5} {'prec':>7} {'recall':>7} {'lift':>6} {'n_signals':>10}")
    for t, p, r, lift, n in rows_x:
        print(f"  {t:>5.2f} {p:>7.3f} {r:>7.3f} {lift:>6.1f} {n:>10}")
    verdict = ("XGBoost beats the baseline OOS — signal is real."
               if ap_x > ap_l + 0.005 else
               "XGBoost does NOT clearly beat the linear baseline — likely overfitting; "
               "prefer the simpler model or revisit features/labels.")
    print(f"\n  VERDICT: {verdict}")

    # --- Final model on ALL data (for live inference) ---
    print("\n[Final] Refitting XGBoost on ALL data for live inference...")
    final = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=scale_pos, random_state=42,
                          n_jobs=4, verbosity=0, eval_metric='aucpr')
    final.fit(X, y, sample_weight=w)
    model_path = os.path.join(out_dir, 'entry_xgb.json')
    final.save_model(model_path)

    # Pick a live threshold from pooled OOF (precision-oriented, recall >= 5%).
    best_t, best_p = 0.7, 0.0
    for t, p, r, lift, n in rows_x:
        if not np.isnan(p) and r >= 0.05 and p > best_p:
            best_p, best_t = p, t
    meta = {'features': feat_cols, 'threshold': best_t,
            'pr_auc_xgb': float(ap_x), 'pr_auc_lr': float(ap_l),
            'base_rate': float(yv.mean()), 'scale_pos_weight': float(scale_pos),
            'n_splits': args.n_splits, 'embargo_days': args.embargo_days}
    with open(os.path.join(out_dir, 'entry_model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    # Persist pooled OOF probabilities for inspection / backtesting.
    oof_df = panel[['Date', 'symbol', 'Entry_Label']].copy()
    oof_df['oof_prob_xgb'] = oof_xgb
    oof_df['oof_prob_lr'] = oof_lr
    oof_df.to_parquet(os.path.join(out_dir, 'entry_oof.parquet'), index=False)

    print(f"[Final] Saved model -> {model_path}")
    print(f"[Final] Meta -> {os.path.join(out_dir, 'entry_model_meta.json')} "
          f"(live threshold {best_t:.2f}, precision ~{best_p:.3f})")

    # --- SHAP on the last fold ---
    if not args.no_shap and last_model is not None:
        try:
            import shap
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            expl = shap.TreeExplainer(last_model)
            sv = expl.shap_values(last_Xtest)
            shap.summary_plot(sv, last_Xtest, show=False, max_display=20)
            p = os.path.join(out_dir, 'entry_shap.png')
            plt.tight_layout(); plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
            print(f"[SHAP] Saved -> {p}")
        except Exception as e:
            print(f"[SHAP] skipped ({e})")

    print("=" * 72)


if __name__ == '__main__':
    main()
