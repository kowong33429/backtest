"""Raw funding settlements (time, rate) per USDT-M perp -> data_funding/<SYM>.csv (development period only).

The per-bar Funding_Rate column in the kline files is an as-of value; it cannot tell how many settlements fell in
a bar (intervals are 8h, 4h or 1h depending on the coin and date). Funding P&L needs the settlement events.
Settlements at/after FINAL_HOLDOUT_START go to data_funding_holdout/ and are not read before step 10.
"""
import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from new_research.download import _funding          # noqa: E402
from common_eval.config import load_config, holdout_start   # noqa: E402


def one(sym, cut, session):
    f = _funding(sym, session)
    if f is None:
        return sym, 0
    for frame, root in ((f[f['Date'] < cut], 'data_funding'), (f[f['Date'] >= cut], 'data_funding_holdout')):
        if len(frame):
            os.makedirs(os.path.join(ROOT, root), exist_ok=True)
            frame.to_csv(os.path.join(ROOT, root, f'{sym}.csv'), index=False)
    return sym, len(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--symbols', nargs='*')
    a = ap.parse_args()
    cut = holdout_start(load_config())
    syms = a.symbols or json.load(open(os.path.join(ROOT, 'data_all', 'universe_symbols.json')))['symbols']
    s = requests.Session()
    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(one, x, cut, s) for x in syms]):
            n += f.result()[1] > 0
    print(f'funding events saved for {n}/{len(syms)} symbols')


if __name__ == '__main__':
    main()
