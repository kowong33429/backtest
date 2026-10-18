"""LEGACY short (short_v2) ladder: AS_IS (reciprocal P&L bug H3 reproduced) -> +H3 -> universe retrain (H1) -> +H2 -> +M4.
Development + walk-forward only. Results -> common_eval/.cache/short_results.pkl"""
import os
import pickle
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config
from common_eval.data import load_market
from common_eval.backtest import Engine, Signals
from common_eval import legacy_adapter as ad
from common_eval import walkforward as wf
from common_eval.run_long import oof_cached, CACHE
from common_eval.metrics import window, model_metrics, yearly

SHORT_XGB = dict(n_estimators=600, learning_rate=0.03, max_depth=5, min_child_weight=50.0, subsample=0.8,
                 colsample_bytree=0.5, random_state=42)           # LEGACY short_v2 hyper-parameters


def evaluate(name, mk, o, thr, cfg, short_pnl, results):
    eng = Engine(mk, cfg, short_pnl=short_pnl)
    sigdf = ad.to_signals(o, thr, 'short', cfg)
    sig = Signals.from_frame(mk, sigdf)
    rec = {'name': name, 'n_signals': sig.n, 'thr': thr, 'sig_df': sigdf, 'pool_mask': ad.pool_mask(mk, o, cfg), 'short_pnl': short_pnl}
    rec['pool_elig'] = int((rec['pool_mask'] & mk.elig).sum())
    lab = ad.label_cut_rows(o, cfg)
    if 'Entry_Label' in lab.columns and len(lab):
        rec['model'] = model_metrics(lab['Entry_Label'].values, lab['prob'].values)
    start, dev_end = cfg['time']['WF_START'], cfg['time']['DEV_END']
    for mode in ('native', 'unified'):
        td, eq = eng.run(sig, mode)
        e = eq[eq.index >= pd.Timestamp(start)]
        rec[mode] = {'td': td, 'eq': eq, 'wf': window(eq, td, start=start),
                     'post_dev': window(eq, td, start=pd.Timestamp(dev_end) + pd.Timedelta(days=1)),
                     'yearly': yearly(e / e.iloc[0])}
    results[name] = rec
    w, u = rec['native']['wf'], rec['unified']['wf']
    print(f"{name:26s} signals {sig.n:7d} | native: n={w['trades']:4d} exp={w['expectancy']*100:+6.2f}% CAGR={w['CAGR']*100:+7.1f}% DD={w['MaxDD']*100:6.1f}%"
          f" | unified: n={u['trades']:4d} exp={u['expectancy']*100:+6.2f}% CAGR={u['CAGR']*100:+7.1f}% DD={u['MaxDD']*100:6.1f}%", flush=True)


def main():
    cfg = load_config()
    thr = 0.5                                                   # LEGACY short baseline (README threshold sweep, save-threshold 0.5)
    mk_bk, mk_all = load_market(cfg, root='data'), load_market(cfg)
    results = {}
    o_bk = ad.read_oof(os.path.join(ROOT, 'data/model/short_v2/entry_oof.parquet'))
    evaluate('S0_AS_IS (reciprocal P&L)', mk_bk, o_bk, thr, cfg, 'legacy_reciprocal', results)
    evaluate('S1_+H3 true short P&L', mk_bk, o_bk, thr, cfg, 'true', results)
    panel = wf.load_panel(os.path.join(ROOT, 'data_all/model/entry_panel_univ_short.parquet'), cfg)
    feats_all = [c for c in panel.columns if c not in wf.NON_FEAT]
    feats_h2 = [c for c in feats_all if c not in wf.SLOW_MACRO]
    print(f'short panel rows {len(panel):,}, features {len(feats_all)}', flush=True)
    # retrained models have a different probability scale, so the literal 0.5 gives 0.8k-7k signals. Rule fixed in advance:
    # keep S0's SIGNAL SHARE (signals / scored rows) and set the threshold from DEVELOPMENT-period scores only (<= DEV_END).
    share = results['S0_AS_IS (reciprocal P&L)']['n_signals'] / len(ad.in_window(o_bk, cfg))
    dev_end = pd.Timestamp(cfg['time']['DEV_END']) + pd.Timedelta(days=1)

    def thr_for(o):
        d = ad.in_window(o, cfg)
        return float(d[d['Date'] < dev_end]['prob'].quantile(1 - share))
    print(f'S0 signal share {share:.4f}', flush=True)
    for nm, key, fe, f0 in [('S2_+H1 universe retrain', 'short_ctrl', feats_all, True),
                            ('S3_+H2 no slow macro', 'short_h2', feats_h2, True),
                            ('S4_+M4 keep NaN', 'short_m4', feats_h2, False)]:
        o = oof_cached(key, lambda: wf.retrain_oof(panel, fe, cfg, f0, params=SHORT_XGB))
        evaluate(nm, mk_all, o, thr_for(o), cfg, 'true', results)
    with open(os.path.join(CACHE, 'short_results.pkl'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
