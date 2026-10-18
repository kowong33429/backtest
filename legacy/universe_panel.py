"""
universe_panel.py — Entry-model panel on the survivorship-free universe (data_all/).

Same per-coin pipeline as dataset.py (EntryLabeler + FeatureEngineer + macro/F&G, strict N-1),
but a row is KEPT only while its coin is in the point-in-time top-N by trailing 30-day median
daily quote volume — ranked with volume up to the PREVIOUS day, i.e. the universe a trader
could actually have known. Rows are stored as float32 to fit in memory.

Usage:
    python core/universe_panel.py                      # top-120, label +100%/60d/-40% (m1)
    python core/universe_panel.py --top 150 --out data_all/model/entry_panel_univ.parquet
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_coin                                  # noqa: E402
from entry_finder import MacroSnapshot, fetch_fear_greed        # noqa: E402


def volume_rank(base, syms, root):
    """Daily point-in-time rank (1 = most liquid) by 30d median quote volume, known at day start."""
    vols = {}
    for s in syms:
        p = os.path.join(base, root, s.lower(), f'{s}_4h_full.csv')
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, usecols=['Date', 'Quote_Volume'], parse_dates=['Date'])
        vols[s] = d.set_index('Date')['Quote_Volume'].resample('D').sum(min_count=1)
    V = pd.DataFrame(vols).sort_index()
    med = V.rolling(30, min_periods=20).median().shift(1)          # yesterday's info only
    return med.rank(axis=1, ascending=False)


def main():
    ap = argparse.ArgumentParser(description='Point-in-time top-N volume universe panel')
    ap.add_argument('--root', default='data_all')
    ap.add_argument('--top', type=int, default=120)
    ap.add_argument('--tp', type=float, default=1.00)
    ap.add_argument('--sl', type=float, default=0.40)
    ap.add_argument('--horizon-days', type=int, default=60)
    ap.add_argument('--min-rows', type=int, default=540, help='4H bars of history a coin needs (90 days)')
    ap.add_argument('--exclude', nargs='*', default=['BTCUSDT'], help='never part of the alt universe')
    ap.add_argument('--out', default='data_all/model/entry_panel_univ.parquet')
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(base, args.root)
    syms = [s for s in json.load(open(os.path.join(root, 'universe_symbols.json')))['symbols'] if s not in args.exclude]
    rank = volume_rank(base, syms, args.root)
    rank.to_parquet(os.path.join(root, 'volume_rank.parquet'))
    in_univ = (rank <= args.top)
    ever = [s for s in syms if s in in_univ.columns and in_univ[s].any()]
    print(f'[Universe] {len(syms)} coins | {len(ever)} were ever in the daily top-{args.top} by volume')

    csvs = {s: os.path.join(root, s.lower(), f'{s}_4h_full.csv') for s in ever}
    gmin = min(pd.read_csv(p, usecols=['Date'], nrows=1, parse_dates=['Date'])['Date'].iloc[0] for p in csvs.values())
    macro = MacroSnapshot(gmin, pd.Timestamp.now()).build()
    fg = fetch_fear_greed()

    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames, n_rows = [], 0
    for k, s in enumerate(ever, 1):
        try:
            m = build_coin(s, csvs[s], macro, fg, args.tp, args.sl, args.horizon_days, min_rows=args.min_rows)
        except Exception as e:
            print(f'  -> {s}: FAILED ({e})')
            continue
        if m is None:
            continue
        day = m['Date'].dt.normalize()
        keep = in_univ[s].reindex(day).fillna(False).values
        m = m[keep]
        if m.empty:
            continue
        num = m.select_dtypes(include=['float64']).columns
        m[num] = m[num].astype('float32')
        frames.append(m)
        n_rows += len(m)
        if k % 25 == 0:
            print(f'[{k}/{len(ever)}] rows so far {n_rows:,}', flush=True)
    panel = pd.concat(frames, ignore_index=True).sort_values(['Date', 'symbol']).reset_index(drop=True)
    panel.to_parquet(out_path, index=False)
    pos = int(panel['Entry_Label'].sum())
    print(f'\n  PANEL {len(panel):,} rows | {panel.symbol.nunique()} coins | positives {pos:,} ({pos / len(panel):.2%})')
    print(f'  {panel.Date.min()} -> {panel.Date.max()} | saved {out_path}')


if __name__ == '__main__':
    main()
