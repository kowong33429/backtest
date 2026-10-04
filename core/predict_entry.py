"""
predict_entry.py — Live per-bar entry probability from the pooled model

Loads the trained XGBoost entry model and scores every 4H bar of a coin with
P(a +100%/30d rally starts here), using ONLY strict n-1 features (same pipeline
as dataset.py). Outputs:

  * <coin>/entry_signals.csv   — Date, close, entry_prob, signal (prob>=threshold)
  * <coin>/entry_signals.html  — price + probability panel with signal markers
  * prints the latest bar's probability (the live read) and recent signals.

NO LOOK-AHEAD: features are shifted in features.py; funding shifted here; macro
publication-lagged + backward-asof. The probability at bar T uses only data < T.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FeatureEngineer                        # noqa: E402
from dataset import select_scale_free                       # noqa: E402
from entry_finder import MacroSnapshot, fetch_fear_greed, fg_at  # noqa: E402


def build_live_features(df, feature_list, use_macro=True):
    """Reproduce dataset.py's feature frame (no labels) for inference."""
    med = df['Date'].diff().median()
    bpd = max(1, round(pd.Timedelta(days=1) / med))

    feat = FeatureEngineer(df.copy()).generate_all_features()
    if 'Funding_Rate' in feat.columns:
        feat['Funding_Rate'] = feat['Funding_Rate'].shift(1)
    cols = select_scale_free(feat.columns)
    m = feat[['Date'] + cols].copy()
    if 'Funding_Rate' in feat.columns:
        m['Funding_Rate'] = feat['Funding_Rate']

    fg = fetch_fear_greed()
    if fg is not None and not fg.empty:
        m['FearGreed'] = [fg_at(fg, d).get('FearGreed') for d in m['Date']]

    if use_macro and any(c not in m.columns for c in feature_list):
        try:
            macro = MacroSnapshot(df['Date'].min(), df['Date'].max()).build()
            if macro.table is not None and not macro.table.empty:
                mt = macro.table.sort_index().reset_index().rename(columns={'index': 'Date'})
                mt['Date'] = pd.to_datetime(mt['Date'])
                m = pd.merge_asof(m.sort_values('Date'), mt.sort_values('Date'),
                                  on='Date', direction='backward')
        except Exception as e:
            print(f"[Predict] macro unavailable ({e})")

    # Align to the EXACT training feature order; missing -> 0.
    for c in feature_list:
        if c not in m.columns:
            m[c] = 0.0
    X = m[feature_list].astype(float).fillna(0.0)
    return m['Date'].values, X, bpd


def main():
    ap = argparse.ArgumentParser(description="Score a coin with the entry model")
    ap.add_argument('--csv', required=True, help='Path to a *_4h_full.csv')
    ap.add_argument('--model-dir', default='data/model')
    ap.add_argument('--threshold', type=float, default=None,
                    help='Override the model meta threshold')
    ap.add_argument('--no-macro', action='store_true')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(base_dir, args.csv)
    model_dir = os.path.join(base_dir, args.model_dir) if not os.path.isabs(args.model_dir) else args.model_dir
    out_dir = args.out_dir or os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(model_dir, 'entry_model_meta.json')) as f:
        meta = json.load(f)
    feature_list = meta['features']
    threshold = args.threshold if args.threshold is not None else meta['threshold']

    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(os.path.join(model_dir, 'entry_xgb.json'))

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    sym = os.path.basename(csv_path).replace('_4h_full.csv', '').upper()
    print(f"[Predict] {sym}: {len(df)} bars | threshold {threshold:.2f} "
          f"(OOS PR-AUC {meta.get('pr_auc_xgb', float('nan')):.3f})")

    dates, X, bpd = build_live_features(df, feature_list, use_macro=not args.no_macro)
    prob = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({'Date': dates, 'entry_prob': prob})
    out = out.merge(df[['Date', 'Close']], on='Date', how='left')
    out['signal'] = (out['entry_prob'] >= threshold).astype(int)
    sig_path = os.path.join(out_dir, 'entry_signals.csv')
    out.to_csv(sig_path, index=False, encoding='utf-8-sig')

    latest = out.iloc[-1]
    recent_sigs = out[out['signal'] == 1].tail(10)
    print(f"\n  LATEST bar {pd.Timestamp(latest['Date'])}: "
          f"entry_prob = {latest['entry_prob']:.3f} "
          f"{'>>> SIGNAL' if latest['signal'] else '(below threshold)'}")
    print(f"  Signals total: {int(out['signal'].sum())} / {len(out)} bars")
    if len(recent_sigs):
        print("  Most recent signals:")
        for _, r in recent_sigs.iterrows():
            print(f"    {pd.Timestamp(r['Date'])}  prob {r['entry_prob']:.3f}  close ${r['Close']:.4g}")

    # Chart: price + probability panel with signal markers.
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"{sym} price + entry signals", "entry_prob"))
        fig.add_trace(go.Scatter(x=out['Date'], y=out['Close'], mode='lines',
                                 line=dict(color='deepskyblue', width=1), name='Close'), row=1, col=1)
        s = out[out['signal'] == 1]
        fig.add_trace(go.Scatter(x=s['Date'], y=s['Close'], mode='markers',
                                 marker=dict(symbol='triangle-up', size=8, color='lime'),
                                 name='signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=out['Date'], y=out['entry_prob'], mode='lines',
                                 line=dict(color='orange', width=1), name='prob'), row=2, col=1)
        fig.add_hline(y=threshold, line=dict(color='white', dash='dash', width=1), row=2, col=1)
        fig.update_layout(template='plotly_dark', height=700, xaxis_rangeslider_visible=False,
                          title=f"{sym} — entry probability (threshold {threshold:.2f})")
        html_path = os.path.join(out_dir, 'entry_signals.html')
        fig.write_html(html_path)
        print(f"\n  Chart: {html_path}")
    except Exception as e:
        print(f"  [chart skipped: {e}]")
    print(f"  Signals CSV: {sig_path}")


if __name__ == '__main__':
    main()
