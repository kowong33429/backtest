"""NEW pipeline driver (steps 5-9), development + walk-forward only. Stages cache into new_research/.cache/.

    python -m new_research.run_all profiles models similarity backtest random
"""
import copy
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config                       # noqa: E402
from common_eval.walkforward import blocks                        # noqa: E402

CFG = load_config()
CACHE = os.path.join(ROOT, 'new_research', '.cache')
os.makedirs(CACHE, exist_ok=True)
WF0 = pd.Timestamp(CFG['time']['WF_START'])
DEV_END = pd.Timestamp(CFG['time']['DEV_END'])
POST0 = DEV_END + pd.Timedelta(days=1)


def save(name, obj):
    with open(os.path.join(CACHE, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f)


def load(name):
    with open(os.path.join(CACHE, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def panel():
    p = pd.read_parquet(os.path.join(ROOT, 'data_new', 'panel_daily.parquet'))
    p['day'] = pd.to_datetime(p['day'])
    return p


# ---------------------------------------------------------------- step 5
def stage_profiles():
    from new_research import profiles as pr
    from new_research.models import FEATS
    p = panel()
    up, dn = pr.event_tables(p)
    up_dev = up[up['period'] == 'development']
    feats = FEATS['up']
    res = {'events_up': up, 'events_down': dn,
           'up_by_year': pr.counts(up)[0], 'up_by_sector': pr.counts(up)[1],
           'down_by_year': pr.counts(dn)[0], 'down_by_sector': pr.counts(dn)[1],
           'lag_profile': pr.lag_profile(p, up_dev, feats),
           'separation_up': pr.separation(p, up_dev, feats)}
    dn_dev = dn[dn['period'] == 'development']
    p2 = p.assign(up=p['down'])                                    # separation() compares to label==0 rows of 'up'
    res['separation_down'] = pr.separation(p2, dn_dev, FEATS['down'])
    res['rule_up'] = pr.derive_rule(res['separation_up'], p)
    res['rule_down'] = pr.derive_rule(res['separation_down'], p2)
    cl = pr.clusters(p, up_dev)
    res['clusters'] = {k: v for k, v in cl.items() if k not in ('model',)}
    res['cluster_hit_rates'] = pr.cluster_hit_rates(p, cl)
    # ZEC-like group (features only)
    dev = p[(p['day'] <= DEV_END) & p['up'].notna()]
    z = pr.zec_like_mask(dev)
    key = dev.set_index(['symbol', 'day'])
    zev = key.reindex(pd.MultiIndex.from_arrays([up_dev['symbol'], pd.to_datetime(up_dev['start'])]))
    res['zec_like'] = {'share_of_coin_days': float(z.mean()), 'up_rate_in_group': float(dev.loc[z, 'up'].mean()),
                       'up_rate_all': float(dev['up'].mean()), 'n_coin_days': int(z.sum()),
                       'events_starting_in_group': int(pr.zec_like_mask(zev).sum()), 'events_total': len(up_dev)}
    save('profiles', res)
    print(f"UP events {len(up)} (dev {len(up_dev)}), DOWN events {len(dn)}; k={cl['k']} silhouette {cl['silhouette']}")
    print('rule_up', res['rule_up']); print('rule_down', res['rule_down']); print('zec_like', res['zec_like'])
    return res


# ---------------------------------------------------------------- step 8
def weekly_precision(o, col, ks):
    d = o[o['y'].notna()].copy()
    d['week'] = d['day'].dt.to_period('W-SUN')
    first = d.groupby('week')['day'].transform('min')
    d = d[d['day'] == first]
    rows = []
    for k in ks:
        pr, lf = [], []
        for _, g in d.groupby('week'):
            base = g['y'].mean()
            top = g.nlargest(k, col)['y'].mean()
            pr.append(top)
            if base > 0:
                lf.append(top / base)
        rows.append({'k': k, 'precision': float(np.mean(pr)), 'lift': float(np.mean(lf)) if lf else np.nan, 'weeks': len(pr)})
    return rows


def model_report(o, models, ks, period=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    d = o if period is None else o[(o['day'] >= period[0]) & (o['day'] <= period[1])]
    lab = d[d['y'].notna()]
    out = []
    for m in models:
        if m not in lab or lab[m].isna().all():
            continue
        r = {'model': m, 'n': len(lab), 'base_rate': float(lab['y'].mean()),
             'pr_auc': float(average_precision_score(lab['y'], lab[m])), 'roc_auc': float(roc_auc_score(lab['y'], lab[m]))}
        for w in weekly_precision(d, m, ks):
            r[f"prec@{w['k']}/wk"] = w['precision']
            r[f"lift@{w['k']}/wk"] = w['lift']
        out.append(r)
    return pd.DataFrame(out)


def shap_importance(info, o, p, feats, n=20000):
    import shap
    rows = []
    for fi in info:
        m = fi.get('_xgb_model')
        if m is None:
            continue
        te = p[(p['day'] >= fi['test_start']) & (p['day'] < fi['test_end'])]
        X = te[feats].sample(min(n, len(te)), random_state=0)
        sv = shap.TreeExplainer(m).shap_values(X)
        rows.append(pd.Series(np.abs(sv).mean(0), index=feats, name=str(fi['test_start'].date())))
    imp = pd.concat(rows, axis=1)
    imp['mean_abs_shap'] = imp.mean(axis=1)
    return imp.sort_values('mean_abs_shap', ascending=False)


def stage_models(targets=('up', 'down'), robustness=True):
    from new_research.models import run_walkforward, FEATS
    p = panel()
    prof = load('profiles')
    ks = CFG['new']['precision_at_k_per_week']
    res = {}
    for tgt in targets:
        t0 = time.time()
        o, info = run_walkforward(p, tgt, rule=prof[f'rule_{tgt}'])
        imp = shap_importance(info, o, p, FEATS[tgt])
        for fi in info:
            fi.pop('_xgb_model', None)
        res[tgt] = {'oof': o, 'info': info, 'shap': imp,
                    'metrics_wf': model_report(o, ['rule', 'logistic', 'xgb'], ks),
                    'metrics_post_dev': model_report(o, ['rule', 'logistic', 'xgb'], ks, (POST0, pd.Timestamp('2025-06-30')))}
        print(f'[{tgt}] done in {time.time() - t0:.0f}s'); print(res[tgt]['metrics_wf'].round(4).to_string())
    if robustness:
        res['robust_labels'] = {}
        for suf in ('_h432', '_h648', '_mae20', '_mae30'):
            o, info = run_walkforward(p, 'up', models=('xgb',), label_col='up' + suf, verbose=False)
            for fi in info:
                fi.pop('_xgb_model', None)
            res['robust_labels'][suf] = {'oof': o, 'info': info, 'metrics_wf': model_report(o, ['xgb'], ks)}
            print('robust', suf, res['robust_labels'][suf]['metrics_wf'].round(4).to_dict('records'))
    save('models', res)
    return res


# ---------------------------------------------------------------- step 7
def stage_similarity():
    from new_research import similarity as sm
    from new_research import profiles as pr
    p = panel()
    prof = load('profiles')
    cl = dict(prof['clusters'])
    cand, tp = sm.search(p, cl, WF0, pd.Timestamp('2025-06-30'))
    res = {'templates_n': {c: v[2] for c, v in tp.items()}}
    for col in ('feat_dist', 'dtw_dist', 'combined_rank'):
        res[f'top20_{col}'] = sm.top_matches(cand, col)
        res[f'curve_{col}'] = sm.hit_curve(cand, col)
        post = cand[cand['day'] >= POST0]
        res[f'curve_post_{col}'] = sm.hit_curve(post, col)
    save('similarity', res)
    for col in ('feat_dist', 'dtw_dist', 'combined_rank'):
        print(res[f'curve_{col}'].round(4).to_string())
    return res


# ---------------------------------------------------------------- step 9
def new_cfg(trailing=None, fee_mult=1.0, h_bars=None):
    c = copy.deepcopy(CFG)
    c['exits']['native_long'] = dict(CFG['new']['exits']['native_up'])
    c['exits']['native_short'] = dict(CFG['new']['exits']['native_down'])
    if trailing is not None:
        c['exits']['unified']['TRAILING_STOP'] = trailing
    if h_bars is not None:
        c['exits']['native_long']['H_BARS'] = h_bars
        c['exits']['unified']['H_BARS'] = h_bars
    c['costs']['FEE_PER_SIDE'] *= fee_mult
    c['costs']['SLIPPAGE_BASE'] *= fee_mult
    return c


def signals(o, info, model, regime=False, side='long', top_per_day=None):
    """Pre-declared rule: score >= the fold's validation (1 - SIGNAL_TOP_FRAC) quantile.
    top_per_day (POST-HOC secondary, added after seeing the primary's low signal count): the best k coins of each day."""
    if top_per_day:
        s = o[o[model].notna()].copy()
        if regime:
            s = s[s['btc_trend'] > CFG['new']['BTC_TREND_MIN']]
        s = s.sort_values(model, ascending=False).groupby('day').head(top_per_day)
        return pd.DataFrame({'ts': s['day'] + pd.Timedelta(days=1), 'symbol': s['symbol'].values,
                             'score': s[model].values, 'side': side})
    rows = []
    for fi in info:
        t = o[(o['day'] >= fi['test_start']) & (o['day'] < fi['test_end'])]
        thr = fi.get(f'{model}_thr')
        if thr is None:
            continue
        s = t[t[model] >= thr]
        if regime:
            s = s[s['btc_trend'] > CFG['new']['BTC_TREND_MIN']]
        rows.append(s)
    s = pd.concat(rows) if rows else o.iloc[:0]
    return pd.DataFrame({'ts': s['day'] + pd.Timedelta(days=1), 'symbol': s['symbol'].values,
                         'score': s[model].values, 'side': side})


def run_variant(mk, cfg, sigdf, short_pnl='true'):
    from common_eval.backtest import Engine, Signals
    from common_eval.metrics import window, yearly
    eng = Engine(mk, cfg, short_pnl=short_pnl)
    sig = Signals.from_frame(mk, sigdf)
    rec = {'n_signals': sig.n, 'sig_df': sigdf}
    for mode in ('native', 'unified'):
        td, eq = eng.run(sig, mode)
        e = eq[eq.index >= WF0]
        rec[mode] = {'td': td, 'eq': eq, 'wf': window(eq, td, start=WF0), 'post_dev': window(eq, td, start=POST0),
                     'yearly': yearly(e / e.iloc[0])}
    return rec


def stage_backtest():
    from common_eval.data import load_market
    mods = load('models')
    mk = load_market(CFG)
    o, info = mods['up']['oof'], mods['up']['info']
    res = {}
    base = new_cfg()
    variants = [(m, n, r, None) for m, n in (('rule', 'NEW_BASELINE'), ('xgb', 'NEW_XGB'), ('logistic', 'NEW_LOGISTIC'))
                for r in (False, True)]
    variants += [('xgb', 'POSTHOC_XGB_top2day', r, 2) for r in (False, True)]
    for model, name, regime, k in variants:
        key = f"{name}{'_regime' if regime else ''}"
        if True:
            res[key] = run_variant(mk, base, signals(o, info, model, regime, top_per_day=k))
            w, u = res[key]['native']['wf'], res[key]['unified']['wf']
            print(f"{key:22s} sig {res[key]['n_signals']:6d} | native n={w['trades']} exp={w['expectancy']*100:+.2f}% CAGR={w['CAGR']*100:+.1f}% DD={w['MaxDD']*100:.1f}% "
                  f"| unified n={u['trades']} exp={u['expectancy']*100:+.2f}% CAGR={u['CAGR']*100:+.1f}% DD={u['MaxDD']*100:.1f}%", flush=True)
    # robustness on NEW_XGB (+ regime variant)
    rob = {}
    for regime in (False, True):
        sx = signals(o, info, 'xgb', regime)
        tag = '_regime' if regime else ''
        rob[f'trail_0.28{tag}'] = run_variant(mk, new_cfg(trailing=0.28), sx)
        rob[f'trail_0.42{tag}'] = run_variant(mk, new_cfg(trailing=0.42), sx)
        rob[f'cost_x2{tag}'] = run_variant(mk, new_cfg(fee_mult=2.0), sx)
        for suf, h in (('_h432', 432), ('_h648', 648), ('_mae20', None), ('_mae30', None)):
            r = mods['robust_labels'][suf]
            c = new_cfg(h_bars=h) if h else copy.deepcopy(base)
            if suf.startswith('_mae'):
                c['exits']['native_long']['SL'] = 0.20 if suf == '_mae20' else 0.30
            rob[f'label{suf}{tag}'] = run_variant(mk, c, signals(r['oof'], r['info'], 'xgb', regime))
    res['_robustness'] = rob
    save('backtest_up', res)
    return res


def stage_random(n=None, key='NEW_XGB'):
    """RANDOM matched to NEW_XGB: same scored pool (coin-days in the panel), same signal rate / same per-bar count."""
    from multiprocessing import Pool
    from common_eval.data import load_market
    from common_eval import legacy_adapter as ad
    from common_eval.run_bench import _init, _run, KEYS
    n = n or CFG['random']['N_RUNS']
    bt = load('backtest_up')
    mods = load('models')
    mk = load_market(CFG)
    o = mods['up']['oof']
    pool = np.zeros((mk.T, mk.N), bool)
    t = np.fromiter((mk.idx(x) for x in o['day'] + pd.Timedelta(days=1)), int, len(o))
    j = np.fromiter((mk.sym_idx.get(s, -1) for s in o['symbol']), int, len(o))
    ok = (t >= 0) & (j >= 0)
    pool[t[ok], j[ok]] = True
    out = {}
    for k in (key, key + '_regime'):
        rec = bt[k]
        rate = rec['n_signals'] / max(1, int((pool & mk.elig).sum()))
        res = {}
        # common_eval.run_bench workers rebuild the Market from config.yaml; NEW native exits are passed via a cfg override
        with Pool(8, initializer=_init_new, initargs=(rec['sig_df'], pool, rate)) as P:
            for mode in ('rate', 'timing'):
                for ex in ('native', 'unified'):
                    res[(mode, ex)] = pd.DataFrame(P.map(_run, [(CFG['random']['SEED'] + i, mode, ex) for i in range(n)], chunksize=10))
                    print(f'  RANDOM {k} {mode}/{ex}: mean trades {res[(mode, ex)].trades.mean():.0f}', flush=True)
        out[k] = res
    save('random_up' if key == 'NEW_XGB' else f'random_up_{key}', out)
    return out


def _init_new(ref_sig, pool, rate):
    from common_eval import run_bench as rb
    from common_eval.data import load_market
    from common_eval.backtest import Engine, Signals
    cfg = new_cfg()
    mk = load_market(cfg)
    rb.G.update(cfg=cfg, mk=mk, eng=Engine(mk, cfg), ref=Signals.from_frame(mk, ref_sig), pool=pool, rate=rate, side=1)


if __name__ == '__main__':
    stages = sys.argv[1:] or ['profiles', 'models', 'similarity', 'backtest', 'random']
    for s in stages:
        t0 = time.time()
        globals()[f'stage_{s}']()
        print(f'== stage {s} done in {time.time() - t0:.0f}s', flush=True)
