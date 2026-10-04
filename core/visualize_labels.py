"""
visualize_labels.py — Eyeball the EntryLabeler output on real price

For a coin, runs the SAME EntryLabeler used to build the training panel and
draws every positive bar (Entry_Label=1 = "+tp% reached before -sl% within
horizon") on the price chart, so you can verify the labels really sit at good
entries BEFORE trusting the model.

What you should SEE if labels are correct:
  * Green bars cluster on the run-ups INTO big rallies (the launch region),
  * They STOP at/near the top (from a top, +100% before -40% rarely happens),
  * They are ABSENT during long declines / chop (no qualifying rally ahead).

Usage:
    python core/visualize_labels.py --csv data/zecusdt/ZECUSDT_4h_full.csv
    python core/visualize_labels.py --csv ... --tp 1.0 --sl 0.4 --horizon-days 30
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labeler import EntryLabeler  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Visualize EntryLabeler labels on price")
    ap.add_argument('--csv', required=True)
    ap.add_argument('--tp', type=float, default=1.00)
    ap.add_argument('--sl', type=float, default=0.40)
    ap.add_argument('--horizon-days', type=int, default=30)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(base_dir, args.csv)
    sym = os.path.basename(csv_path).replace('_4h_full.csv', '').upper()

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    lab = EntryLabeler(df, tp_pct=args.tp, sl_pct=args.sl,
                       horizon_days=args.horizon_days).generate()
    pos = lab[lab['Entry_Label'] == 1]

    # --- Sanity table: a few positives with their realized forward outcome ---
    print(f"\n[Check] {sym}: {len(pos)} positive bars / {len(lab)} "
          f"({len(pos)/len(lab)*100:.1f}%). Sample positives and what followed:")
    print(f"  {'entry_date':<20}{'entry':>10}{'resolve_date':<22}{'days':>5}{'realized_gain':>14}")
    sample = pos.iloc[np.linspace(0, len(pos) - 1, min(12, len(pos))).astype(int)] if len(pos) else pos
    for _, r in sample.iterrows():
        e = r['Open']
        rd = pd.Timestamp(r['Label_EndDate'])
        days = (rd - pd.Timestamp(r['Date'])).days
        # realized gain = High path reached the +tp target by construction
        print(f"  {str(pd.Timestamp(r['Date'])):<20}{e:>10.4g}{str(rd):<22}{days:>5}{args.tp*100:>13.0f}%")

    out = args.out or os.path.join(os.path.dirname(csv_path), 'label_check.html')

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.72, 0.28],
                        subplot_titles=(
                            f"{sym} — green = positive entry label "
                            f"(+{args.tp*100:.0f}% before -{args.sl*100:.0f}% within "
                            f"{args.horizon_days}d)", "Volume"))

    # Price line + positive-label markers (on the Open, the execution price).
    fig.add_trace(go.Scatter(x=lab['Date'], y=lab['Close'], mode='lines',
                             line=dict(color='#5b6b7b', width=1), name='Close'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=pos['Date'], y=pos['Open'], mode='markers',
                             marker=dict(symbol='circle', size=4, color='lime',
                                         line=dict(width=0)),
                             name='Entry_Label=1'), row=1, col=1)
    if 'Volume' in lab.columns:
        fig.add_trace(go.Bar(x=lab['Date'], y=lab['Volume'], marker_color='#30363d',
                             name='Volume'), row=2, col=1)

    fig.update_layout(template='plotly_dark', height=760, xaxis_rangeslider_visible=False,
                      title=f"{sym} label check — {len(pos)} positives "
                            f"({len(pos)/len(lab)*100:.1f}% of bars)",
                      legend=dict(orientation='h', y=1.05, x=1, xanchor='right'))
    fig.write_html(out)
    print(f"\n  Chart: {out}")
    print("  -> Verify: green should cluster on the climbs INTO rallies and vanish "
          "at tops / during declines.")


if __name__ == '__main__':
    main()
