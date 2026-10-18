"""LEGACY long (m1) ladder through common_eval: AS_IS -> +H1 -> (+H1 retrain control) -> +H2 -> +M4.
Development + walk-forward period only. Results -> common_eval/.cache/long_results.pkl"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config
from common_eval.data import load_market
from common_eval.backtest import Engine, Signals
from common_eval.metrics import window, model_metrics, yearly
from common_eval import legacy_adapter as ad
from common_eval import walkforward as wf

CACHE = os.path.join(ROOT, 'common_eval', '.cache')
os.makedirs(CACHE, exist_ok=True)


def oof_cached(name, fn):
    p = os.path.join(CACHE, f'oof_{name}.parquet')
    if os.path.exists(p):
        return pd.read_parquet(p)
    o = fn()
    o.to_parquet(p, index=False)
    return o


def evaluate(name, mk, eng, o, thr, side, cfg, results):
    sigdf = ad.to_signals(o, thr, side, cfg)
    sig = Signals.from_frame(mk, sigdf)
    pool = ad.pool_mask(mk, o, cfg)
    rec = {'name': name, 'n_signals': sig.n, 'pool_elig': int((pool & mk.elig).sum()), 'pool_mask': pool, 'thr': thr,
           'sig_df': sigdf}
    lab = ad.label_cut_rows(o, cfg)
    if 'Entry_Label' in lab.columns and len(lab):
        rec['model'] = model_metrics(lab['Entry_Label'].values, lab['prob'].values)
    start, dev_end = cfg['time']['WF_START'], cfg['time']['DEV_END']
    for mode in ('native', 'unified'):
        td, eq = eng.run(sig, mode)
        rec[mode] = {'td': td, 'eq': eq,
                     'wf': window(eq, td, start=start),
                     'post_dev': window(eq, td, start=pd.Timestamp(dev_end) + pd.Timedelta(days=1)),
                     'yearly': yearly(eq[eq.index >= pd.Timestamp(start)] / eq[eq.index >= pd.Timestamp(start)].iloc[0])}
    results[name] = rec
    w = rec['native']['wf']; u = rec['unified']['wf']
    print(f"{name:28s} signals {sig.n:7d} | native: n={w['trades']:4d} exp={w['expectancy']*100:+6.2f}% CAGR={w['CAGR']*100:+7.1f}% DD={w['MaxDD']*100:6.1f}%"
          f" | unified: n={u['trades']:4d} exp={u['expectancy']*100:+6.2f}% CAGR={u['CAGR']*100:+7.1f}% DD={u['MaxDD']*100:6.1f}%", flush=True)


def main():
    cfg = load_config()
    thr = cfg['signals']['LEGACY_THRESHOLD']
    t0 = time.time()
    mk_all, mk_bk = load_market(cfg), load_market(cfg, root='data')
    eng_all, eng_bk = Engine(mk_all, cfg), Engine(mk_bk, cfg)
    print(f'universe: data_all {mk_all.N} coins, avg eligible/bar {mk_all.elig.sum(1)[mk_all.times >= pd.Timestamp(cfg["time"]["WF_START"])].mean():.0f}'
          f' | basket {mk_bk.N} coins, avg eligible {mk_bk.elig.sum(1).mean():.0f}', flush=True)
    results = {}

    # L0  LEGACY_AS_IS: basket + basket-trained OOF
    o_bk = ad.read_oof(os.path.join(ROOT, 'data/model/v2/m1_100pct_60d/entry_oof.parquet'))
    evaluate('L0_AS_IS', mk_bk, eng_bk, o_bk, thr, 'long', cfg, results)
    # L1  +H1: point-in-time universe, existing m1_univ OOF
    o_un = ad.read_oof(os.path.join(ROOT, 'data_all/model/m1_univ/entry_oof.parquet'))
    evaluate('L1_+H1_universe', mk_all, eng_all, o_un, thr, 'long', cfg, results)

    panel = wf.load_panel(os.path.join(ROOT, 'data_all/model/entry_panel_univ.parquet'), cfg)
    feats_all = [c for c in panel.columns if c not in wf.NON_FEAT]
    feats_h2 = [c for c in feats_all if c not in wf.SLOW_MACRO]
    print(f'panel rows {len(panel):,} (< holdout), features {len(feats_all)} -> {len(feats_h2)} after H2', flush=True)
    o = oof_cached('long_ctrl', lambda: wf.retrain_oof(panel, feats_all, cfg, fillna0=True))
    evaluate('L1c_retrain_control', mk_all, eng_all, o, thr, 'long', cfg, results)
    o = oof_cached('long_h2', lambda: wf.retrain_oof(panel, feats_h2, cfg, fillna0=True))
    evaluate('L2_+H2_no_slow_macro', mk_all, eng_all, o, thr, 'long', cfg, results)
    o = oof_cached('long_m4', lambda: wf.retrain_oof(panel, feats_h2, cfg, fillna0=False))
    evaluate('L4_+M4_keep_NaN', mk_all, eng_all, o, thr, 'long', cfg, results)

    with open(os.path.join(CACHE, 'long_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
