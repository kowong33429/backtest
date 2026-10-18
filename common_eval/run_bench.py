"""BTC_HOLD, EQUAL_WEIGHT and RANDOM (1,000 runs) for LEGACY_FIXED (long L4 and short S4). Dev + walk-forward only."""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config
from common_eval.data import load_market
from common_eval.backtest import Engine, Signals
from common_eval.metrics import window, full_metrics
from common_eval import benchmarks as bm

CACHE = os.path.join(ROOT, 'common_eval', '.cache')
G = {}
KEYS = ['trades', 'expectancy', 'CAGR', 'MaxDD', 'Sharpe', 'Sortino', 'Calmar', 'win_rate', 'final_multiple']


def _init(ref_sig, pool, rate, side, short_pnl):
    cfg = load_config()
    mk = load_market(cfg)
    G.update(cfg=cfg, mk=mk, eng=Engine(mk, cfg, short_pnl=short_pnl), ref=Signals.from_frame(mk, ref_sig),
             pool=pool, rate=rate, side=side)


def _run(args):
    seed, mode, exit_mode = args
    mk, cfg = G['mk'], G['cfg']
    rng = np.random.default_rng(seed)
    sig = bm.random_signals(mk, rng, G['rate'], mode=mode, ref=G['ref'], mask=G['pool'])
    if G['side'] == -1:
        sig.side[:] = -1
    td, eq = G['eng'].run(sig, exit_mode)
    m = window(eq, td, start=cfg['time']['WF_START'])
    return {k: m.get(k, np.nan) for k in KEYS}


def random_set(rec, side, short_pnl, cfg, n, base_seed, workers=8):
    pool = rec['pool_mask']
    mk_n = rec['n_signals']
    rate = mk_n / max(1, rec['pool_elig'])
    out = {}
    with Pool(workers, initializer=_init, initargs=(rec['sig_df'].assign(side='long' if side == 1 else 'short'), pool, rate, side, short_pnl)) as p:
        for mode in ('rate', 'timing'):
            for ex in ('native', 'unified'):
                t0 = time.time()
                res = p.map(_run, [(base_seed + i, mode, ex) for i in range(n)], chunksize=10)
                out[(mode, ex)] = pd.DataFrame(res)
                print(f'  random {mode}/{ex}: {n} runs in {time.time() - t0:.0f}s, mean trades {out[(mode, ex)].trades.mean():.0f}', flush=True)
    return out


def main():
    cfg = load_config()
    n = cfg['random']['N_RUNS']
    mk = load_market(cfg)
    res = {}
    # benchmarks
    t0 = bm.btc_hold(mk, cfg)
    res['BTC_HOLD'] = {'eq': t0, 'wf': window(t0, None, start=cfg['time']['WF_START'])}
    ew = bm.equal_weight(mk, cfg)
    res['EQUAL_WEIGHT'] = {'eq': ew, 'wf': window(ew, None, start=cfg['time']['WF_START']), 'avg_names': ew.attrs.get('avg_names')}
    for k in ('BTC_HOLD', 'EQUAL_WEIGHT'):
        w = res[k]['wf']
        print(f"{k:13s} CAGR {w['CAGR']*100:+6.1f}% DD {w['MaxDD']*100:6.1f}% Sharpe {w['Sharpe']:.2f}", flush=True)
    for name, pk, key, side, spnl in [('LEGACY_FIXED_LONG', 'long_results.pkl', 'L4_+M4_keep_NaN', 1, 'true'),
                                      ('LEGACY_FIXED_SHORT', 'short_results.pkl', 'S4_+M4 keep NaN', -1, 'true')]:
        results = pickle.load(open(os.path.join(CACHE, pk), 'rb'))
        rec = results[key]
        print(f'RANDOM for {name}: signals {rec["n_signals"]}, pool {rec["pool_elig"]}, rate {rec["n_signals"] / rec["pool_elig"]:.4f}', flush=True)
        res[name] = random_set(rec, side, spnl, cfg, n, cfg['random']['SEED'])
    with open(os.path.join(CACHE, 'bench_results.pkl'), 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    main()
