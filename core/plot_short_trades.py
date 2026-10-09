"""
plot_short_trades.py — Plot the ACTUAL SHORT orders the backtest opened.

Mirror of plot_m1_trades.py for the SHORT model. Reads the short backtest trade
log (entry_backtest_short.py output) and draws every short order on real price,
colored by outcome:

    TP   (green)  — price FELL to entry x0.5  -> short +100%
    SL   (red)    — price ROSE to entry x1.667 -> short -40%
    TIME (gold)   — neither; covered at the 60-day time stop

Entry markers are DOWN triangles (a short is opened by selling). Each entry links
to its exit so the realized move is visible. A dropdown switches coins (listed
with per-coin P&L); the first page is an ALL-COINS overview scattering every
short's net return over time.

Usage:
    python core/plot_short_trades.py
    python core/plot_short_trades.py --symbols ZECUSDT DASHUSDT
    python core/plot_short_trades.py --trades data/model/short/entry_backtest_trades.csv
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

OUTCOME_COLORS = {'TP': '#3fb950', 'SL': '#f85149', 'TIME': '#d29922'}


def discover_price_csvs(base_dir):
    found = {}
    for csv in glob.glob(os.path.join(base_dir, 'data', '*', '*_4h_full.csv')):
        sym = os.path.basename(csv).replace('_4h_full.csv', '').upper()
        found[sym] = csv
    return found


def main():
    ap = argparse.ArgumentParser(description="Plot the SHORT model's actual backtest orders on price")
    ap.add_argument('--trades', default='data/model/short/entry_backtest_trades.csv')
    ap.add_argument('--symbols', nargs='*', default=None,
                    help='Only these symbols (default: all traded coins)')
    ap.add_argument('--out', default='data/model/short/short_trades_viz.html')
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trades_path = os.path.join(base_dir, args.trades) if not os.path.isabs(args.trades) else args.trades
    out_path = os.path.join(base_dir, args.out) if not os.path.isabs(args.out) else args.out

    td = pd.read_csv(trades_path)
    td['entry_date'] = pd.to_datetime(td['entry_date'])
    td['exit_date'] = pd.to_datetime(td['exit_date'])
    print(f"[Load] {len(td)} SHORT orders across {td['symbol'].nunique()} coins from {trades_path}")

    price_csvs = discover_price_csvs(base_dir)
    pnl_by_coin = td.groupby('symbol')['pnl_usd'].sum()
    want = ([s.upper() for s in args.symbols] if args.symbols
            else sorted(pnl_by_coin.index))

    import plotly.graph_objects as go

    fig = go.Figure()
    pages = []
    trace_pages = []

    # ---- Page 0: ALL-COINS overview (net return of every short over time) ----
    for oc, color in OUTCOME_COLORS.items():
        sub = td[td['outcome'] == oc]
        fig.add_trace(go.Scattergl(
            x=sub['entry_date'], y=sub['net_ret'] * 100, mode='markers',
            marker=dict(size=7, color=color, line=dict(width=0.5, color='#0e1117')),
            name=f'{oc} ({len(sub)})', legendgroup=oc,
            customdata=np.stack([sub['symbol'], sub['bars_held'] / 6.0], axis=-1),
            hovertemplate='%{customdata[0]}<br>%{x|%Y-%m-%d}<br>'
                          f'{oc} · net %{{y:.1f}}%%<br>held %{{customdata[1]:.0f}}d<extra></extra>',
            visible=True))
        trace_pages.append(0)
    fig.add_hline(y=0, line=dict(color='#6e7681', width=1, dash='dot'))
    win = (td['net_ret'] > 0).mean() * 100
    pages.append(('★ ALL COINS (overview)',
                  f"SHORT model — all {len(td)} orders · win {win:.1f}% · "
                  f"net P&amp;L ${td['pnl_usd'].sum():,.0f} (@$100/trade, reciprocal P&amp;L)",
                  3))

    # ---- One page per coin: price + short entry/exit markers + links ----
    for sym in want:
        if sym not in price_csvs:
            print(f"  {sym}: no price CSV — skipped")
            continue
        g = td[td['symbol'] == sym].sort_values('entry_date')
        if g.empty:
            continue
        px = pd.read_csv(price_csvs[sym], usecols=['Date', 'Close'])
        px['Date'] = pd.to_datetime(px['Date'])

        start = g['entry_date'].min() - pd.Timedelta(days=60)
        stop = g['exit_date'].max() + pd.Timedelta(days=30)
        pxw = px[(px['Date'] >= start) & (px['Date'] <= stop)]

        base_n = len(trace_pages)
        page_idx = len(pages)

        fig.add_trace(go.Scattergl(
            x=pxw['Date'], y=pxw['Close'], mode='lines',
            line=dict(color='#5b6b7b', width=1), name='Close', visible=False,
            hovertemplate='%{x|%Y-%m-%d}<br>Close %{y:.6g}<extra></extra>'))
        trace_pages.append(page_idx)

        lx, ly = [], []
        for _, r in g.iterrows():
            lx += [r['entry_date'], r['exit_date'], None]
            ly += [r['entry_price'], r['exit_price'], None]
        fig.add_trace(go.Scattergl(
            x=lx, y=ly, mode='lines',
            line=dict(color='#30363d', width=1), name='entry→exit',
            visible=False, hoverinfo='skip', showlegend=False))
        trace_pages.append(page_idx)

        # Short entry markers (DOWN triangle = sell to open), colored by outcome
        fig.add_trace(go.Scattergl(
            x=g['entry_date'], y=g['entry_price'], mode='markers',
            marker=dict(symbol='triangle-down', size=11,
                        color=[OUTCOME_COLORS[o] for o in g['outcome']],
                        line=dict(width=1, color='#e6e6e6')),
            name='short entry', visible=False,
            customdata=np.stack([g['outcome'], g['net_ret'] * 100,
                                 g['bars_held'] / 6.0], axis=-1),
            hovertemplate='SHORT %{x|%Y-%m-%d} @ %{y:.6g}<br>'
                          '%{customdata[0]} · net %{customdata[1]:.1f}% · '
                          'held %{customdata[2]:.0f}d<extra></extra>'))
        trace_pages.append(page_idx)

        fig.add_trace(go.Scattergl(
            x=g['exit_date'], y=g['exit_price'], mode='markers',
            marker=dict(symbol='x', size=7,
                        color=[OUTCOME_COLORS[o] for o in g['outcome']]),
            name='cover', visible=False, hoverinfo='skip', showlegend=False))
        trace_pages.append(page_idx)

        n_tp = (g['outcome'] == 'TP').sum()
        n_sl = (g['outcome'] == 'SL').sum()
        n_tm = (g['outcome'] == 'TIME').sum()
        pages.append((
            f"{sym}  (${pnl_by_coin[sym]:,.0f} · {len(g)} shorts)",
            f"{sym} — {len(g)} shorts · TP {n_tp} / SL {n_sl} / TIME {n_tm} · "
            f"P&amp;L ${pnl_by_coin[sym]:,.0f}",
            base_n))

    n_traces = len(trace_pages)
    buttons = []
    for pidx, (label, title, _) in enumerate(pages):
        vis = [tp == pidx for tp in trace_pages]
        buttons.append(dict(label=label, method='update',
                            args=[{'visible': vis}, {'title': title}]))

    for i in range(n_traces):
        fig.data[i].visible = (trace_pages[i] == 0)

    fig.update_layout(
        template='plotly_dark', height=780, xaxis_rangeslider_visible=False,
        title=pages[0][1],
        updatemenus=[dict(buttons=buttons, direction='down', showactive=True,
                          x=0.0, xanchor='left', y=1.12, yanchor='top')],
        legend=dict(orientation='h', y=1.05, x=1, xanchor='right'))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
    print(f"\n[Done] {len(pages)-1} coins + overview, {len(td)} short orders")
    print(f"[Chart] {out_path}")
    print("  Open it; use the dropdown (top-left) to switch between the overview and each coin.")
    print("  Green ▽ = short that hit +100% TP · Red ▽ = -40% SL · Gold ▽ = 60-day time stop.")


if __name__ == '__main__':
    main()
