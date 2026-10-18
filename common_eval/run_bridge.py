"""Bridge: how LEGACY_AS_IS (basket m1, thr 0.60, native exit) changes when each common_eval rule is switched on.
Per-trade expectancy is the comparable number (CAGR only exists once capital is constrained)."""
import copy
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config
from common_eval.data import load_market
from common_eval.backtest import Engine, Signals
from common_eval.metrics import window, trade_metrics
from common_eval import legacy_adapter as ad

cfg0 = load_config()
o = ad.read_oof(os.path.join(ROOT, 'data/model/v2/m1_100pct_60d/entry_oof.parquet'))
rows = []


def step(label, c, start, rand_priority=False, root='data', exit_mode='native'):
    mk = load_market(c, root=root, start=start)
    eng = Engine(mk, c)
    sd = ad.to_signals(o, 0.60, 'long', c)
    if rand_priority:
        sd = sd.assign(score=np.random.default_rng(1).random(len(sd)))
    sig = Signals.from_frame(mk, sd)
    td, eq = eng.run(sig, exit_mode)
    w = window(eq, td, start=c['time']['WF_START'])
    rows.append({'step': label, 'trades': w['trades'], 'expectancy': w['expectancy'], 'win_rate': w['win_rate'],
                 'CAGR': w['CAGR'] if c['portfolio']['MAX_POSITIONS'] < 1000 else np.nan,
                 'MaxDD': w['MaxDD'] if c['portfolio']['MAX_POSITIONS'] < 1000 else np.nan})
    print(rows[-1], flush=True)


a = copy.deepcopy(cfg0)
a['portfolio']['MAX_POSITIONS'] = 100000
a['universe']['MIN_QUOTE_VOL_30D'] = 0
a['costs'].update(SLIPPAGE_BASE=0.0005, SLIP_REF_QV=0, SLIP_MAX_MULT=1)
a['universe']['drop_zero_volume_bars'] = False
a['time']['WF_START'] = '2019-05-01'
step('B0 LEGACY rules (2019+, flat 0.05% slip, no liquidity filter, unlimited slots)', a, '2019-04-01')
a['time']['WF_START'] = cfg0['time']['WF_START']
step('B1 + window 2022-01..2025-06', a, '2021-10-01')
a['costs'] = copy.deepcopy(cfg0['costs'])
step('B2 + common costs (fee 0.10% + liquidity-based slippage >=0.20%)', a, '2021-10-01')
a['universe']['MIN_QUOTE_VOL_30D'] = cfg0['universe']['MIN_QUOTE_VOL_30D']
a['universe']['drop_zero_volume_bars'] = True
step('B3 + liquidity filter (30d qv >= 500k) + drop zero-volume bars', a, '2021-10-01')
a['portfolio']['MAX_POSITIONS'] = cfg0['portfolio']['MAX_POSITIONS']
step('B4 + MAX_POSITIONS=10, highest score first, equal slots', a, '2021-10-01')
step('B4r same, but random priority instead of score', a, '2021-10-01', rand_priority=True)
step('B5 B4 with unified exit (trail 35% / 540 bars)', a, '2021-10-01', exit_mode='unified')
pd.DataFrame(rows).to_pickle(os.path.join(ROOT, 'common_eval', '.cache', 'bridge.pkl'))
