"""
visualize_short_labels.py — Eyeball the ShortEntryLabeler output on real price

Mirror of visualize_labels.py. Draws every positive SHORT bar (Entry_Label=1 =
"short reaches +tp% before -sl% within horizon", i.e. price falls to entry/(1+tp)
before rising to entry/(1-sl)) on the price chart, so you can verify the labels
sit at good SHORT entries BEFORE trusting the model.

What you should SEE if labels are correct (OPPOSITE of the long model):
  * Red bars cluster on the TOPS / roll-overs INTO big declines (the launch of a
    crash), i.e. just before price craters,
  * They are ABSENT during long uptrends / near bottoms (no qualifying decline
    ahead, or price spikes up into the stop first).

Usage:
    python core/visualize_short_labels.py --csv data/zecusdt/ZECUSDT_4h_full.csv
    python core/visualize_short_labels.py --csv ... --tp 1.0 --sl 0.4 --horizon-days 60
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labeler_short import ShortEntryLabeler  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Visualize ShortEntryLabeler labels on price")
    ap.add_argument('--csv', required=True)
    ap.add_argument('--tp', type=float, default=1.00)
    ap.add_argument('--sl', type=float, default=0.40)
    ap.add_argument('--horizon-days', type=int, default=60)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(base_dir, args.csv)
    sym = os.path.basename(csv_path).replace('_4h_full.csv', '').replace('_1d_full.csv', '').upper()

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    lab = ShortEntryLabeler(df, tp_pct=args.tp, sl_pct=args.sl,
                            horizon_days=args.horizon_days).generate()
    pos = lab[lab['Entry_Label'] == 1]

    tp_mult = 1.0 / (1 + args.tp)   # price multiplier at the short TP (down)
    sl_mult = 1.0 / (1 - args.sl)   # price multiplier at the short SL (up)

    # --- Sanity table: a few positives with their realized forward outcome ---
    print(f"\n[Check] {sym}: {len(pos)} positive SHORT bars / {len(lab)} "
          f"({len(pos)/len(lab)*100:.1f}%). Sample positives and what followed:")
    print(f"  {'entry_date':<20}{'entry':>12}{'tp_price':>12}{'resolve_date':<22}{'days':>5}{'short_profit':>13}")
    sample = pos.iloc[np.linspace(0, len(pos) - 1, min(12, len(pos))).astype(int)] if len(pos) else pos
    for _, r in sample.iterrows():
        e = r['Open']
        rd = pd.Timestamp(r['Label_EndDate'])
        days = (rd - pd.Timestamp(r['Date'])).days
        # By construction a positive hit the TP-down target -> short profit = +tp%
        print(f"  {str(pd.Timestamp(r['Date'])):<20}{e:>12.4g}{e*tp_mult:>12.4g}"
              f"{str(rd):<22}{days:>5}{args.tp*100:>12.0f}%")

    out = args.out or os.path.join(os.path.dirname(csv_path), 'label_check_short.html')

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.72, 0.28],
                        subplot_titles=(
                            f"{sym} — red = positive SHORT label "
                            f"(price x{tp_mult:.2f} [+{args.tp*100:.0f}%] before "
                            f"x{sl_mult:.2f} [-{args.sl*100:.0f}%] within "
                            f"{args.horizon_days}d)", "Volume"))

    fig.add_trace(go.Scatter(x=lab['Date'], y=lab['Close'], mode='lines',
                             line=dict(color='#5b6b7b', width=1), name='Close'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=pos['Date'], y=pos['Open'], mode='markers',
                             marker=dict(symbol='triangle-down', size=6, color='red',
                                         line=dict(width=0)),
                             name='Short Entry_Label=1'), row=1, col=1)
    if 'Volume' in lab.columns:
        fig.add_trace(go.Bar(x=lab['Date'], y=lab['Volume'], marker_color='#30363d',
                             name='Volume'), row=2, col=1)

    fig.update_layout(template='plotly_dark', height=760, xaxis_rangeslider_visible=False,
                      title=f"{sym} SHORT label check — {len(pos)} positives "
                            f"({len(pos)/len(lab)*100:.1f}% of bars)",
                      legend=dict(orientation='h', y=1.05, x=1, xanchor='right'))
    fig.write_html(out)
    print(f"\n  Chart: {out}")
    print("  -> Verify: red should cluster on the TOPS / roll-overs INTO big "
          "declines and vanish during uptrends / at bottoms.")


if __name__ == '__main__':
    main()
