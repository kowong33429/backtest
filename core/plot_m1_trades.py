"""
plot_m1_trades.py — Plot the ACTUAL entry points Model 1 traded in the backtest.

Unlike plot_train_entries.py (which draws every Entry_Label=1 bar fed to training),
this reads the backtest trade log — the bars where the trained model's OOS
probability crossed the live threshold and a trade was actually opened — and draws
each one on real price, colored by outcome:

    TP   (green)  — hit +100% take-profit
    SL   (red)    — hit -40% stop
    TIME (gold)   — neither; exited at the 60-day time stop

Each entry is linked to its exit so the realized move is visible. A dropdown
switches coins (ordered by P&L); the first page is an ALL-COINS overview that
scatters every trade's net return over time.

Usage:
    python core/plot_m1_trades.py
    python core/plot_m1_trades.py --symbols FETUSDT CELRUSDT
    python core/plot_m1_trades.py --trades data/model/v2/m1_100pct_60d/entry_backtest_trades.csv
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
    ap = argparse.ArgumentParser(description="Plot Model 1's actual backtest entries on price")
    ap.add_argument('--trades',
                    default='data/model/v2/m1_100pct_60d/entry_backtest_trades.csv')
    ap.add_argument('--symbols', nargs='*', default=None,
                    help='Only these symbols (default: all traded coins)')
    ap.add_argument('--out', default='data/model/v2/m1_100pct_60d/m1_trades_viz.html')
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trades_path = os.path.join(base_dir, args.trades) if not os.path.isabs(args.trades) else args.trades
    out_path = os.path.join(base_dir, args.out) if not os.path.isabs(args.out) else args.out

    td = pd.read_csv(trades_path)
    td['entry_date'] = pd.to_datetime(td['entry_date'])
    td['exit_date'] = pd.to_datetime(td['exit_date'])
    print(f"[Load] {len(td)} trades across {td['symbol'].nunique()} coins from {trades_path}")

    price_csvs = discover_price_csvs(base_dir)

    # P&L per coin (shown in each dropdown label); coins listed alphabetically.
    pnl_by_coin = td.groupby('symbol')['pnl_usd'].sum()
    want = ([s.upper() for s in args.symbols] if args.symbols
            else sorted(pnl_by_coin.index))

    import plotly.graph_objects as go

    fig = go.Figure()
    pages = []          # (label, title, n_traces_on_this_page)
    trace_pages = []    # page index for each trace (for dropdown visibility)

    # ---- Page 0: ALL-COINS overview (net return of every trade over time) ----
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
                  f"Model 1 — all {len(td)} trades · win {win:.1f}% · "
                  f"net P&amp;L ${td['pnl_usd'].sum():,.0f} (@$100/trade)",
                  3))

    # ---- One page per coin: price + entry/exit markers + entry→exit links ----
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

        # Price line
        fig.add_trace(go.Scattergl(
            x=pxw['Date'], y=pxw['Close'], mode='lines',
            line=dict(color='#5b6b7b', width=1), name='Close', visible=False,
            hovertemplate='%{x|%Y-%m-%d}<br>Close %{y:.6g}<extra></extra>'))
        trace_pages.append(page_idx)

        # Entry→exit connecting segments (one trace, gaps via None)
        lx, ly, lc = [], [], []
        for _, r in g.iterrows():
            lx += [r['entry_date'], r['exit_date'], None]
            ly += [r['entry_price'], r['exit_price'], None]
        fig.add_trace(go.Scattergl(
            x=lx, y=ly, mode='lines',
            line=dict(color='#30363d', width=1), name='entry→exit',
            visible=False, hoverinfo='skip', showlegend=False))
        trace_pages.append(page_idx)

        # Entry markers, colored by outcome
        fig.add_trace(go.Scattergl(
            x=g['entry_date'], y=g['entry_price'], mode='markers',
            marker=dict(symbol='triangle-up', size=11,
                        color=[OUTCOME_COLORS[o] for o in g['outcome']],
                        line=dict(width=1, color='#e6e6e6')),
            name='entry', visible=False,
            customdata=np.stack([g['outcome'], g['net_ret'] * 100,
                                 g['bars_held'] / 6.0], axis=-1),
            hovertemplate='ENTRY %{x|%Y-%m-%d} @ %{y:.6g}<br>'
                          '%{customdata[0]} · net %{customdata[1]:.1f}% · '
                          'held %{customdata[2]:.0f}d<extra></extra>'))
        trace_pages.append(page_idx)

        # Exit markers
        fig.add_trace(go.Scattergl(
            x=g['exit_date'], y=g['exit_price'], mode='markers',
            marker=dict(symbol='x', size=7,
                        color=[OUTCOME_COLORS[o] for o in g['outcome']]),
            name='exit', visible=False, hoverinfo='skip', showlegend=False))
        trace_pages.append(page_idx)

        n_tp = (g['outcome'] == 'TP').sum()
        n_sl = (g['outcome'] == 'SL').sum()
        n_tm = (g['outcome'] == 'TIME').sum()
        pages.append((
            f"{sym}  (${pnl_by_coin[sym]:,.0f} · {len(g)} trades)",
            f"{sym} — {len(g)} entries · TP {n_tp} / SL {n_sl} / TIME {n_tm} · "
            f"P&amp;L ${pnl_by_coin[sym]:,.0f}",
            base_n))

    n_traces = len(trace_pages)

    # Dropdown: each page shows only its own traces.
    buttons = []
    for pidx, (label, title, _) in enumerate(pages):
        vis = [tp == pidx for tp in trace_pages]
        buttons.append(dict(label=label, method='update',
                            args=[{'visible': vis}, {'title': title}]))

    # Start on the overview page.
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
    print(f"\n[Done] {len(pages)-1} coins + overview, {len(td)} trades")
    print(f"[Chart] {out_path}")
    print("  Open it; use the dropdown (top-left) to switch between the overview and each coin.")


if __name__ == '__main__':
    main()
