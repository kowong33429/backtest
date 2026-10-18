"""Builds the SHORT-label panel on the survivorship-free universe (legacy build_coin, direction='short', 30d).

Same recipe as core/universe_panel.py (PIT top-120 rows only) but with the short labeler, so the short model can be
retrained on the same universe as the long model. Uses legacy code unchanged. Labels look forward (they are Y);
the evaluation harness drops everything at/after the holdout start before training or scoring.
"""
import os
import sys
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'core'))
from dataset import build_coin                                   # noqa: E402
from entry_finder import MacroSnapshot, fetch_fear_greed          # noqa: E402

G = {}


def _init(macro, fg, in_univ):
    G['macro'], G['fg'], G['in_univ'] = macro, fg, in_univ


def _one(args):
    s, path = args
    try:
        m = build_coin(s, path, G['macro'], G['fg'], 1.0, 0.4, 30, min_rows=540, direction='short')
    except Exception as e:
        return s, None, str(e)
    if m is None:
        return s, None, 'short history'
    day = m['Date'].dt.normalize()
    m = m[G['in_univ'][s].reindex(day).fillna(False).values]
    if m.empty:
        return s, None, 'never in universe'
    num = m.select_dtypes(include=['float64']).columns
    m[num] = m[num].astype('float32')
    return s, m, ''


def main():
    root = os.path.join(ROOT, 'data_all')
    rank = pd.read_parquet(os.path.join(root, 'volume_rank.parquet'))
    in_univ = rank <= 120
    ever = [s for s in in_univ.columns if in_univ[s].any()]
    csvs = {s: os.path.join(root, s.lower(), f'{s}_4h_full.csv') for s in ever}
    csvs = {s: p for s, p in csvs.items() if os.path.exists(p)}
    gmin = min(pd.read_csv(p, usecols=['Date'], nrows=1, parse_dates=['Date'])['Date'].iloc[0] for p in csvs.values())
    macro = MacroSnapshot(gmin, pd.Timestamp.now()).build()
    fg = fetch_fear_greed()
    print(f'[short panel] {len(csvs)} coins, F&G rows: {None if fg is None else len(fg)}', flush=True)
    frames = []
    with Pool(8, initializer=_init, initargs=(macro, fg, in_univ)) as pool:
        for k, (s, m, why) in enumerate(pool.imap_unordered(_one, list(csvs.items())), 1):
            if m is not None:
                frames.append(m)
            if k % 25 == 0:
                print(f'[{k}/{len(csvs)}] frames {len(frames)}', flush=True)
    panel = pd.concat(frames, ignore_index=True).sort_values(['Date', 'symbol']).reset_index(drop=True)
    out = os.path.join(root, 'model', 'entry_panel_univ_short.parquet')
    panel.to_parquet(out, index=False)
    print(f'saved {out} rows {len(panel):,} coins {panel.symbol.nunique()} pos {panel.Entry_Label.mean():.3%}')


if __name__ == '__main__':
    main()
