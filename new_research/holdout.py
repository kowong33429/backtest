"""Step 10: ONE run of every frozen variant on the final holdout (2025-07-01 -> 2026-09-30). Nothing is re-fit or tuned.

common_eval is used unchanged; the holdout window is passed as an in-memory config (FINAL_HOLDOUT_START = end of the
window, HOLDOUT_LOCKED = false, WF_START = 2025-07-01, FUNDING_DIR = development + holdout settlements).
Models / thresholds come from new_research/models_frozen (tag new-frozen).
"""
import copy
import glob
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config                                    # noqa: E402
from common_eval.data import load_market                                      # noqa: E402
from common_eval.backtest import Engine, Signals                              # noqa: E402
from common_eval.metrics import window, yearly, model_metrics                 # noqa: E402
from common_eval import benchmarks as bm                                       # noqa: E402
from new_research.features import coin_features, daily_sample                 # noqa: E402
from new_research.labels import label_bars                                    # noqa: E402
from new_research.models import rule_score                                    # noqa: E402

CFG = load_config()
H0, H_END = pd.Timestamp('2025-07-01'), pd.Timestamp('2026-10-01')          # [start, end)
FROZEN = os.path.join(ROOT, 'new_research', 'models_frozen')
CACHE = os.path.join(ROOT, 'new_research', '.cache', 'holdout')
os.makedirs(CACHE, exist_ok=True)
BAR = pd.Timedelta(hours=4)


def hcfg(new_exits=False, fee_mult=1.0):
    c = copy.deepcopy(CFG)
    c['time'].update(FINAL_HOLDOUT_START=str(H_END.date()), HOLDOUT_LOCKED=False, WF_START=str(H0.date()), DEV_END=str(H0.date()))
    c['costs']['FUNDING_DIR'] = 'data_funding_full'
    if new_exits:
        c['exits']['native_long'] = dict(CFG['new']['exits']['native_up'])
        c['exits']['native_short'] = dict(CFG['new']['exits']['native_down'])
    c['costs']['FEE_PER_SIDE'] *= fee_mult
    c['costs']['SLIPPAGE_BASE'] *= fee_mult
    return c


def merge_funding():
    out = os.path.join(ROOT, 'data_funding_full')
    os.makedirs(out, exist_ok=True)
    for p in glob.glob(os.path.join(ROOT, 'data_funding', '*.csv')) + glob.glob(os.path.join(ROOT, 'data_funding_holdout', '*.csv')):
        s = os.path.basename(p)
        parts = [pd.read_csv(q, parse_dates=['Date']) for q in (os.path.join(ROOT, 'data_funding', s), os.path.join(ROOT, 'data_funding_holdout', s)) if os.path.exists(q)]
        pd.concat(parts).drop_duplicates('Date').sort_values('Date').to_csv(os.path.join(out, s), index=False)


# ---------------------------------------------------------------- NEW holdout panel
def load_full(sym):
    parts = [pd.read_csv(p, parse_dates=['Date']) for p in (os.path.join(ROOT, 'data_new', sym.lower(), f'{sym}_4h.csv'),
                                                           os.path.join(ROOT, 'data_new_holdout', sym.lower(), f'{sym}_4h.csv'))
             if os.path.exists(p)]
    if not parts:
        return None
    d = pd.concat(parts).drop_duplicates('Date').sort_values('Date')
    d = d[(d['Volume'] > 0) & (d['Date'] < H_END)]
    bh, bl = d[['Open', 'Close']].max(axis=1), d[['Open', 'Close']].min(axis=1)       # same wick cap as new_research.data
    d['High'] = d['High'].clip(upper=3 * bh)
    d['Low'] = d['Low'].clip(lower=bl / 3)
    return d.set_index('Date')


def build_one(args):
    sym, listed, fut, sector = args
    d = load_full(sym)
    if d is None or len(d) < 720 or d.index.max() < H0:
        return None
    f = coin_features(d, listed, fut)
    pos = pd.Series(np.arange(len(d)), index=d.index)
    day = daily_sample(f)
    day = day[day.index >= H0 - pd.Timedelta(days=1)]
    if day.empty:
        return None
    idx = pos.loc[day['bar_time']].values
    delisted = d.index.max() < H_END - pd.Timedelta(days=2)
    lb = CFG['new']['label']
    up, dn, upk, dnk, peak = label_bars(d, idx, CFG['horizons']['H_NEW'], lb['UP_TARGET'], lb['DOWN_TARGET'], lb['MAE_LIMIT'], delisted)
    day['up'], day['down'], day['up_peak'] = up, dn, peak
    day['close'] = d['Close'].loc[day['bar_time']].values
    day['symbol'], day['sector'] = sym, sector
    return day


def holdout_panel():
    p_path = os.path.join(CACHE, 'panel_holdout.parquet')
    if os.path.exists(p_path):
        p = pd.read_parquet(p_path); p['day'] = pd.to_datetime(p['day']); return p
    from new_research.panel import add_cross_section
    md = pd.read_csv(os.path.join(ROOT, 'data', 'coin_metadata.csv'), parse_dates=['spot_first', 'futures_first', 'listed_first'])
    md = md.set_index('symbol')
    syms = sorted({os.path.basename(x).upper() for x in glob.glob(os.path.join(ROOT, 'data_new_holdout', '*'))})
    jobs = [(s, md['listed_first'].get(s), md['futures_first'].get(s), md['sector'].get(s, 'other')) for s in syms]
    with Pool(12) as pool:
        frames = [x for x in pool.imap_unordered(build_one, jobs, chunksize=4) if x is not None]
    p = pd.concat(frames).sort_index()
    p.index.name = 'day'
    p['sector'] = p['sector'].fillna('other')
    p = add_cross_section(p)
    btc = load_full('BTCUSDT')['Close']
    tr = btc / btc.ewm(span=1200, adjust=False, min_periods=600).mean() - 1
    p['btc_trend'] = tr.groupby(tr.index.normalize()).last().reindex(p.index).values
    p = p.reset_index()
    p.to_parquet(p_path, index=False)
    return p


# ---------------------------------------------------------------- scoring
def score_logistic(m, X):
    Z = X.fillna(m['impute_median']).fillna(0.0)
    for c in m['impute_cols']:
        if c.endswith('_isna'):
            Z[c] = X[c[:-5]].isna().astype(float)
    Z = Z[m['impute_cols']]
    z = m['scaler'].transform(Z)
    return m['logistic'].predict_proba(z)[:, 1], pd.DataFrame(z * m['logistic'].coef_[0], columns=m['impute_cols'], index=X.index)


def new_signals(p):
    up, dn = pickle.load(open(os.path.join(FROZEN, 'new_up.pkl'), 'rb')), pickle.load(open(os.path.join(FROZEN, 'new_down.pkl'), 'rb'))
    s = p[(p['day'] >= H0 - pd.Timedelta(days=1)) & (p['day'] < H_END - pd.Timedelta(days=1))].copy()
    s['up_logistic'], _ = score_logistic(up, s[up['feats']])
    s['up_rule'] = rule_score(s, up['rule'], up['rule_thresholds'])
    s['down_rule'] = rule_score(s, dn['rule'], dn['rule_thresholds'])
    s['down_logistic'], _ = score_logistic(dn, s[dn['feats']])
    s['down_xgb'] = dn['xgb'].predict_proba(s[dn['feats']])[:, 1]
    thr = {'up_logistic': up['thr_logistic'], 'up_rule': 10.0, 'down_rule': 10.0, 'down_logistic': dn['thr_logistic'], 'down_xgb': dn['thr_xgb']}
    sig = {}
    for name, col, side in (('NEW_LOGISTIC', 'up_logistic', 'long'), ('NEW_BASELINE', 'up_rule', 'long'),
                            ('NEW_DOWN_BASELINE', 'down_rule', 'short'), ('NEW_DOWN_LOGISTIC', 'down_logistic', 'short'),
                            ('NEW_DOWN_XGB', 'down_xgb', 'short')):
        x = s[s[col] >= thr[col]]
        sig[name] = pd.DataFrame({'ts': x['day'] + pd.Timedelta(days=1), 'symbol': x['symbol'].values, 'score': x[col].values, 'side': side})
    return s, sig, thr, up, dn


def legacy_signals():
    from common_eval import legacy_adapter as ad
    from common_eval import walkforward as wf
    out = {}
    o = ad.read_oof(os.path.join(ROOT, 'data/model/v2/m1_100pct_60d/entry_oof.parquet'))
    o = o[(o['Date'] >= H0 - BAR) & (o['Date'] < H_END - BAR) & (o['prob'] >= CFG['signals']['LEGACY_THRESHOLD'])]
    out['LEGACY_AS_IS_long'] = pd.DataFrame({'ts': o['Date'] + BAR, 'symbol': o['symbol'].values, 'score': o['prob'].values, 'side': 'long'})
    o = ad.read_oof(os.path.join(ROOT, 'data/model/short_v2/entry_oof.parquet'))
    o = o[(o['Date'] >= H0 - BAR) & (o['Date'] < H_END - BAR) & (o['prob'] >= 0.5)]
    out['LEGACY_AS_IS_short'] = pd.DataFrame({'ts': o['Date'] + BAR, 'symbol': o['symbol'].values, 'score': o['prob'].values, 'side': 'short'})
    pools = {}
    for key, fn, side in (('LEGACY_FIXED_long', 'legacy_fixed_long', 'long'), ('LEGACY_FIXED_short', 'legacy_fixed_short', 'short')):
        m = pickle.load(open(os.path.join(FROZEN, f'{fn}.pkl'), 'rb'))
        path = 'data_all/model/entry_panel_univ.parquet' if side == 'long' else 'data_all/model/entry_panel_univ_short.parquet'
        p = pd.read_parquet(os.path.join(ROOT, path))
        p = p[(p['Date'] >= H0 - BAR) & (p['Date'] < H_END - BAR)]
        prob = m['xgb'].predict_proba(p[m['feats']].astype('float32'))[:, 1]
        sc = pd.DataFrame({'Date': p['Date'].values, 'symbol': p['symbol'].values, 'prob': prob, 'Entry_Label': p['Entry_Label'].values,
                           'Label_EndDate': p['Label_EndDate'].values})
        pools[key] = sc
        x = sc[sc['prob'] >= m['thr']]
        out[key] = pd.DataFrame({'ts': x['Date'] + BAR, 'symbol': x['symbol'].values, 'score': x['prob'].values, 'side': side})
    return out, pools


# ---------------------------------------------------------------- run
def run(mk, cfg, sigdf, short_pnl='true', funding=True):
    eng = Engine(mk, cfg, short_pnl=short_pnl, funding=funding)
    sig = Signals.from_frame(mk, sigdf)
    rec = {'n_signals': sig.n, 'sig_df': sigdf}
    for mode in ('native', 'unified'):
        td, eq = eng.run(sig, mode)
        e = eq[eq.index >= H0]
        rec[mode] = {'td': td, 'eq': eq, 'wf': window(eq, td, start=H0), 'yearly': yearly(e / e.iloc[0])}
    return rec


G = {}


def _init(cfg, ref_sig, pool, rate, side):
    from common_eval import run_bench as rb
    mk = load_market(cfg, start='2025-05-01')
    rb.G.update(cfg=cfg, mk=mk, eng=Engine(mk, cfg), ref=Signals.from_frame(mk, ref_sig), pool=pool, rate=rate, side=side)


def random_for(cfg, mk, rec, pool, side, n):
    from common_eval.run_bench import _run
    rate = rec['n_signals'] / max(1, int((pool & mk.elig).sum()))
    res = {}
    ref = rec['sig_df'].assign(side='long')
    with Pool(8, initializer=_init, initargs=(cfg, ref, pool, rate, side)) as P:
        for mode in ('timing', 'rate'):
            for ex in ('native', 'unified'):
                res[(mode, ex)] = pd.DataFrame(P.map(_run, [(CFG['random']['SEED'] + 7 + i, mode, ex) for i in range(n)], chunksize=10))
    return res


def pool_mask(mk, days, syms, shift):
    m = np.zeros((mk.T, mk.N), bool)
    t = np.fromiter((mk.idx(x) for x in pd.to_datetime(days) + shift), int, len(days))
    j = np.fromiter((mk.sym_idx.get(s, -1) for s in syms), int, len(syms))
    ok = (t >= 0) & (j >= 0)
    m[t[ok], j[ok]] = True
    return m


def main():
    t0 = time.time()
    merge_funding()
    c_leg, c_new = hcfg(), hcfg(new_exits=True)
    mk = load_market(c_leg, start='2025-05-01')
    mk_bk = load_market(c_leg, root='data', start='2025-05-01')
    print(f'holdout market: {mk.N} coins, bars {mk.times[0]} .. {mk.times[-1]}, avg eligible {mk.elig[mk.times >= H0].sum(1).mean():.0f}', flush=True)
    p = holdout_panel()
    print(f'NEW holdout panel {len(p):,} rows, {p.symbol.nunique()} coins, {p.day.min().date()}..{p.day.max().date()}', flush=True)
    s, nsig, thr, up_m, dn_m = new_signals(p)
    lsig, lpools = legacy_signals()
    res = {}
    for k in ('NEW_LOGISTIC', 'NEW_BASELINE', 'NEW_DOWN_BASELINE', 'NEW_DOWN_LOGISTIC', 'NEW_DOWN_XGB'):
        res[k] = run(mk, c_new, nsig[k])
    res['LEGACY_AS_IS_long'] = run(mk_bk, c_leg, lsig['LEGACY_AS_IS_long'])
    res['LEGACY_AS_IS_short'] = run(mk_bk, c_leg, lsig['LEGACY_AS_IS_short'], short_pnl='legacy_reciprocal', funding=False)
    res['LEGACY_FIXED_long'] = run(mk, c_leg, lsig['LEGACY_FIXED_long'])
    res['LEGACY_FIXED_short'] = run(mk, c_leg, lsig['LEGACY_FIXED_short'])
    for k, r in res.items():
        w, u = r['native']['wf'], r['unified']['wf']
        print(f"{k:20s} sig {r['n_signals']:6d} | native n={w.get('trades')} exp={w.get('expectancy', np.nan)*100:+.2f}% CAGR={w['CAGR']*100:+.1f}% DD={w['MaxDD']*100:.1f}% "
              f"| unified n={u.get('trades')} exp={u.get('expectancy', np.nan)*100:+.2f}% CAGR={u['CAGR']*100:+.1f}%", flush=True)
    # robustness asked by section 11 for the hypotheses (cost x2)
    res['_cost_x2'] = {'NEW_LOGISTIC': run(mk, hcfg(True, 2.0), nsig['NEW_LOGISTIC']),
                       'NEW_DOWN_BASELINE': run(mk, hcfg(True, 2.0), nsig['NEW_DOWN_BASELINE'])}
    bench = {'BTC_HOLD': bm.btc_hold(mk, c_leg), 'EQUAL_WEIGHT': bm.equal_weight(mk, c_leg)}
    res['_bench'] = {k: {'eq': v, 'wf': window(v, None, start=H0)} for k, v in bench.items()}
    for k, v in res['_bench'].items():
        print(f"{k:13s} CAGR {v['wf']['CAGR']*100:+.1f}% DD {v['wf']['MaxDD']*100:.1f}% Sharpe {v['wf']['Sharpe']:.2f} Calmar {v['wf']['Calmar']:.2f}", flush=True)
    # model metrics on labelled holdout rows
    lab = s[s['up'].notna()]
    res['_model'] = {'up_logistic': model_metrics(lab['up'].values, lab['up_logistic'].values),
                     'down_rule_pass': float(s.loc[s['down'].notna() & (s['down_rule'] >= 10), 'down'].mean()),
                     'down_base': float(s['down'].mean()), 'up_base': float(s['up'].mean())}
    dl = s[s['down'].notna()]
    for m in ('down_logistic', 'down_xgb'):
        res['_model'][m] = model_metrics(dl['down'].values, dl[m].values)
    pickle.dump({'res': res, 'thr': thr}, open(os.path.join(CACHE, 'holdout_results.pkl'), 'wb'))
    s.to_parquet(os.path.join(CACHE, 'scores_holdout.parquet'), index=False)
    for k, v in lpools.items():
        v.to_parquet(os.path.join(CACHE, f'pool_{k}.parquet'), index=False)
    print(f'backtests done in {time.time() - t0:.0f}s', flush=True)
    # RANDOM (1,000 each) for the two hypotheses and LEGACY_FIXED
    n = CFG['random']['N_RUNS']
    rnd = {}
    sp = pool_mask(mk, s['day'], s['symbol'], pd.Timedelta(days=1))
    rnd['NEW_LOGISTIC'] = random_for(c_new, mk, res['NEW_LOGISTIC'], sp, 1, n)
    rnd['NEW_DOWN_BASELINE'] = random_for(c_new, mk, res['NEW_DOWN_BASELINE'], sp, -1, n)
    for key, side in (('LEGACY_FIXED_long', 1), ('LEGACY_FIXED_short', -1)):
        pl = lpools[key]
        rnd[key] = random_for(c_leg, mk, res[key], pool_mask(mk, pl['Date'], pl['symbol'], BAR), side, n)
    for k, d in rnd.items():
        print(k, {f'{a}/{b}': round(v['trades'].mean()) for (a, b), v in d.items()}, flush=True)
    pickle.dump(rnd, open(os.path.join(CACHE, 'holdout_random.pkl'), 'wb'))
    print(f'all done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
