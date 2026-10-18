"""Harness validation: configure common_eval like LEGACY's entry_backtest (unlimited positions, flat 0.05% slippage,
no liquidity filter, 2019+ window) and compare to the trade log LEGACY produced. Not part of the ladder."""
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config
from common_eval.data import load_market
from common_eval.backtest import Engine, Signals
from common_eval import legacy_adapter as ad

cfg = load_config()
v = copy.deepcopy(cfg)
v['portfolio']['MAX_POSITIONS'] = 100000
v['universe']['MIN_QUOTE_VOL_30D'] = 0
v['costs'].update(SLIPPAGE_BASE=0.0005, SLIP_REF_QV=0, SLIP_MAX_MULT=1)
v['time']['WF_START'] = '2019-05-01'
v['universe']['drop_zero_volume_bars'] = False
mk = load_market(v, root='data', start='2019-04-01')
eng = Engine(mk, v)
o = ad.read_oof(os.path.join(ROOT, 'data/model/v2/m1_100pct_60d/entry_oof.parquet'))
sig = Signals.from_frame(mk, ad.to_signals(o, 0.60, 'long', v))
td, eq = eng.run(sig, 'native')
# sizing is irrelevant for per-trade stats; unlimited slots means every signal is taken (one per coin at a time)
leg = pd.read_csv(os.path.join(ROOT, 'data/model/v2/m1_100pct_60d/thr06/entry_backtest_trades.csv'), parse_dates=['entry_date', 'exit_date'])
cut = pd.Timestamp('2025-04-30')
leg = leg[leg['exit_date'] < cut]
td = td[td['exit_time'] < cut]
print(f'LEGACY log : {len(leg):5d} trades  win {(leg.net_ret > 0).mean():.3f}  exp {leg.net_ret.mean() * 100:+.2f}%  (exits before {cut.date()})')
print(f'common_eval: {len(td):5d} trades  win {(td.ret > 0).mean():.3f}  exp {td.ret.mean() * 100:+.2f}%')
print('reasons', td.reason.value_counts().to_dict(), 'legacy', leg.outcome.value_counts().to_dict())
