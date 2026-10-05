"""
plot_train_entries.py — Plot the EXACT entry points used to train the entry model.

Unlike visualize_labels.py (which RE-RUNS the labeler on one coin), this reads the
positive labels straight from the training panel (data/model/entry_panel.parquet),
so every green dot is a bar that was actually fed to the model as Entry_Label=1.

The panel keeps only scale-free features (no price), so we join each coin's
Entry_Label back onto its 4H price CSV by Date to draw the dots on real price.

Output: one interactive HTML with a coin dropdown (price line + entry markers).

Usage:
    python core/plot_train_entries.py
    python core/plot_train_entries.py --symbols FETUSDT ZECUSDT DOGEUSDT
    python core/plot_train_entries.py --min-pos 50 --out data/model/entry_labels_viz.html
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd


def discover_price_csvs(base_dir):
    """symbol -> 4H price CSV path."""
    found = {}
    for csv in glob.glob(os.path.join(base_dir, 'data', '*', '*_4h_full.csv')):
        sym = os.path.basename(csv).replace('_4h_full.csv', '').upper()
        found[sym] = csv
    return found


def main():
    ap = argparse.ArgumentParser(description="Plot training entry labels on price")
    ap.add_argument('--panel', default='data/model/entry_panel.parquet')
    ap.add_argument('--symbols', nargs='*', default=None,
                    help='Only these symbols (default: all in panel)')
    ap.add_argument('--min-pos', type=int, default=1,
                    help='Skip coins with fewer than this many positive labels')
    ap.add_argument('--out', default='data/model/entry_labels_viz.html')
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    panel_path = os.path.join(base_dir, args.panel) if not os.path.isabs(args.panel) else args.panel
    out_path = os.path.join(base_dir, args.out) if not os.path.isabs(args.out) else args.out

    print(f"[Load] panel labels from {panel_path}")
    panel = pd.read_parquet(panel_path, columns=['Date', 'symbol', 'Entry_Label'])
    panel['Date'] = pd.to_datetime(panel['Date'])

    price_csvs = discover_price_csvs(base_dir)

    # Per-coin positive counts; order coins by most positives first.
    counts = (panel.groupby('symbol')['Entry_Label']
              .agg(n='size', pos='sum').sort_values('pos', ascending=False))
    want = [s.upper() for s in args.symbols] if args.symbols else list(counts.index)

    import plotly.graph_objects as go

    fig = go.Figure()
    buttons = []
    coin_meta = []  # (symbol, pos, n) for coins actually plotted
    n_traces = 0

    for sym in want:
        if sym not in counts.index:
            print(f"  {sym}: not in panel — skipped")
            continue
        pos = int(counts.loc[sym, 'pos'])
        if pos < args.min_pos:
            continue
        if sym not in price_csvs:
            print(f"  {sym}: {pos} pos but no price CSV — skipped")
            continue

        px = pd.read_csv(price_csvs[sym], usecols=['Date', 'Open', 'Close'])
        px['Date'] = pd.to_datetime(px['Date'])
        sub = panel[panel['symbol'] == sym][['Date', 'Entry_Label']]
        merged = px.merge(sub, on='Date', how='inner').sort_values('Date')
        entries = merged[merged['Entry_Label'] == 1]

        visible = (len(coin_meta) == 0)  # first plotted coin visible
        fig.add_trace(go.Scattergl(
            x=merged['Date'], y=merged['Close'], mode='lines',
            line=dict(color='#5b6b7b', width=1), name='Close', visible=visible,
            hovertemplate='%{x}<br>Close %{y:.6g}<extra></extra>'))
        fig.add_trace(go.Scattergl(
            x=entries['Date'], y=entries['Open'], mode='markers',
            marker=dict(symbol='circle', size=5, color='lime'),
            name='Entry_Label=1', visible=visible,
            hovertemplate='%{x}<br>entry @ Open %{y:.6g}<extra></extra>'))
        coin_meta.append((sym, pos, len(merged)))
        n_traces += 2

    if not coin_meta:
        print("No coins to plot (check --symbols / --min-pos / CSVs).")
        return

    # Build dropdown: each coin toggles its own 2 traces visible.
    for i, (sym, pos, n) in enumerate(coin_meta):
        vis = [False] * n_traces
        vis[2 * i] = True
        vis[2 * i + 1] = True
        buttons.append(dict(
            label=f"{sym}  ({pos} pos / {n})", method='update',
            args=[{'visible': vis},
                  {'title': f"{sym} — {pos} training entries "
                            f"({pos / max(1, n) * 100:.1f}% of {n} bars)"}]))

    sym0, pos0, n0 = coin_meta[0]
    fig.update_layout(
        template='plotly_dark', height=760, xaxis_rangeslider_visible=False,
        title=f"{sym0} — {pos0} training entries ({pos0 / max(1, n0) * 100:.1f}% of {n0} bars)",
        updatemenus=[dict(buttons=buttons, direction='down', showactive=True,
                          x=0.0, xanchor='left', y=1.12, yanchor='top')],
        legend=dict(orientation='h', y=1.05, x=1, xanchor='right'))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
    total_pos = sum(p for _, p, _ in coin_meta)
    print(f"\n[Done] {len(coin_meta)} coins, {total_pos} total entry labels")
    print(f"[Chart] {out_path}")
    print("  Open it and use the dropdown (top-left) to switch coins.")


if __name__ == '__main__':
    main()
