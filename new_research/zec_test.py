"""Step 10 ZEC test (definitions pre-registered in reports/holdout_hypotheses.md).

run       = from ZEC's lowest daily close between 2025-07-01 and its holdout peak, up to that peak
'early'   = a signal while the close is still < 2x that low (window starts 30 days before the low)
For every variant: first ZEC signal (date, execution price, multiple of the low), whether common_eval opened the trade
and its result; NEW logistic -> top-5 feature contributions, NEW rules -> which conditions failed; LEGACY_FIXED -> SHAP
top-5. False positives: other coins signalled from 30 days before the low to the peak, and how many of them did a 2X.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from new_research.holdout import CACHE, FROZEN, H0, H_END, load_full, score_logistic, BAR    # noqa: E402

UP_VARIANTS = ['NEW_LOGISTIC', 'NEW_BASELINE', 'LEGACY_AS_IS_long', 'LEGACY_FIXED_long']
DOWN_VARIANTS = ['NEW_DOWN_BASELINE', 'NEW_DOWN_LOGISTIC', 'NEW_DOWN_XGB', 'LEGACY_AS_IS_short', 'LEGACY_FIXED_short']


def zec_run():
    d = load_full('ZECUSDT')
    c = d['Close'][(d.index >= H0)].groupby(lambda x: x.normalize()).last()
    peak_day = c.idxmax()
    low_day = c[:peak_day].idxmin()
    low, peak = c[low_day], c[peak_day]
    x2 = c[(c.index > low_day) & (c >= 2 * low)]
    x2_day = x2.index[0] if len(x2) else None
    return {'low_day': low_day, 'low': float(low), 'peak_day': peak_day, 'peak': float(peak), 'x2_day': x2_day,
            'multiple': float(peak / low), 'daily_close': c}


def signal_day(k, ts):
    ts = pd.to_datetime(ts)
    return (ts - pd.Timedelta(days=1)) if k.startswith('NEW') else (ts - BAR)


def main():
    R = pickle.load(open(os.path.join(CACHE, 'holdout_results.pkl'), 'rb'))
    res, thr = R['res'], R['thr']
    s = pd.read_parquet(os.path.join(CACHE, 'scores_holdout.parquet')); s['day'] = pd.to_datetime(s['day'])
    z = zec_run()
    w0, w_early, w_end = z['low_day'] - pd.Timedelta(days=30), z['x2_day'], z['peak_day']
    print(f"ZEC run: low {z['low']:.2f} on {z['low_day'].date()} -> 2x on {z['x2_day'].date() if z['x2_day'] is not None else '-'} "
          f"-> peak {z['peak']:.2f} on {z['peak_day'].date()} ({z['multiple']:.1f}x)")
    rows, fps, reasons = [], [], {}
    up_m = pickle.load(open(os.path.join(FROZEN, 'new_up.pkl'), 'rb'))
    zs = s[s['symbol'] == 'ZECUSDT'].set_index('day').sort_index()
    for k in UP_VARIANTS + DOWN_VARIANTS:
        r = res[k]
        sd = r['sig_df'].copy()
        sd['sday'] = signal_day(k, sd['ts'])
        zsig = sd[(sd['symbol'] == 'ZECUSDT') & (sd['sday'] >= w0) & (sd['sday'] <= w_end)].sort_values('ts')
        early = zsig[zsig['sday'] <= (w_early if w_early is not None else w_end)]
        td = r['native']['td']
        ztr = td[(td['symbol'] == 'ZECUSDT') & (td['entry_time'] >= w0) & (td['entry_time'] <= w_end + pd.Timedelta(days=1))] if len(td) else td
        first = early.iloc[0] if len(early) else (zsig.iloc[0] if len(zsig) else None)
        px = None
        if first is not None:
            dd = load_full('ZECUSDT')
            px = float(dd['Open'].asof(pd.Timestamp(first['ts'])))
        rows.append({'variant': k, 'side': 'long' if k in UP_VARIANTS else 'short',
                     'zec_signals_in_window': len(zsig), 'zec_signals_before_2x': len(early),
                     'first_signal_day': first['sday'] if first is not None else None,
                     'first_exec_price': px, 'multiple_of_low_at_entry': px / z['low'] if px else None,
                     'caught_before_2x': bool(len(early)) if k in UP_VARIANTS else None,
                     'trade_opened': bool(len(ztr)), 'trade_entry': ztr['entry_time'].min() if len(ztr) else None,
                     'trade_entry_px': float(ztr['entry_px'].iloc[0]) if len(ztr) else None,
                     'trade_exit_px': float(ztr['exit_px'].iloc[0]) if len(ztr) else None,
                     'trade_reason': ztr['reason'].iloc[0] if len(ztr) else None,
                     'trade_ret': float(ztr['ret'].iloc[0]) if len(ztr) else None})
        # false positives (UP variants): other coins signalled in [w0, peak]
        if k in UP_VARIANTS:
            other = sd[(sd['symbol'] != 'ZECUSDT') & (sd['sday'] >= w0) & (sd['sday'] <= w_end)]
            firsts = other.sort_values('ts').groupby('symbol').head(1)
            if k.startswith('NEW'):
                lab = s.set_index(['symbol', 'day'])['up']
                hit = lab.reindex(pd.MultiIndex.from_arrays([firsts['symbol'], firsts['sday'].dt.normalize()]))
            else:
                pool = pd.read_parquet(os.path.join(CACHE, f'pool_LEGACY_FIXED_long.parquet')) if k == 'LEGACY_FIXED_long' else None
                hit = pd.Series(np.nan, index=firsts.index)
                if pool is not None:
                    hit = pool.set_index(['symbol', 'Date'])['Entry_Label'].reindex(pd.MultiIndex.from_arrays([firsts['symbol'], firsts['sday']]))
            fps.append({'variant': k, 'signals_other_coins': len(other), 'distinct_other_coins': firsts['symbol'].nunique(),
                        'with_known_outcome': int(hit.notna().sum()), 'did_2x_within_90d': int(np.nansum(hit.values)),
                        'hit_rate': float(np.nanmean(hit.values)) if hit.notna().any() else np.nan,
                        'note': 'NEW label: 2x within 90d before -25%' if k.startswith('NEW') else 'LEGACY label: +100% within 60d before -40% (FIXED only)'})
    # reasons for NEW_LOGISTIC / rules on ZEC
    zw = zs[(zs.index >= w0) & (zs.index <= w_end)]
    sc, contrib = score_logistic(up_m, zw[up_m['feats']])
    zl = pd.DataFrame({'score': sc, 'threshold': thr['up_logistic'], 'close': zw['close']}, index=zw.index)
    reasons['NEW_LOGISTIC_scores'] = zl
    best = zl['score'].idxmax()
    reasons['NEW_LOGISTIC_top5_at_best_day'] = (best, contrib.loc[best].sort_values(key=np.abs, ascending=False).head(5))
    lowd = z['low_day']
    if lowd in contrib.index:
        reasons['NEW_LOGISTIC_top5_at_low'] = (lowd, contrib.loc[lowd].sort_values(key=np.abs, ascending=False).head(5))
    rule_rows = []
    for d in [lowd, w_early] if w_early is not None else [lowd]:
        if d in zs.index:
            for f, op, q in up_m['rule']:
                t = up_m['rule_thresholds'][f]
                v = zs.at[d, f]
                rule_rows.append({'day': d, 'feature': f, 'op': op, 'threshold': t, 'zec_value': v,
                                  'passes': bool((v >= t) if op == '>=' else (v <= t))})
    reasons['NEW_BASELINE_rule_on_zec'] = pd.DataFrame(rule_rows)
    # LEGACY_FIXED long SHAP top-5 on its first ZEC signal (or best bar)
    try:
        import shap
        m = pickle.load(open(os.path.join(FROZEN, 'legacy_fixed_long.pkl'), 'rb'))
        pl = pd.read_parquet(os.path.join(ROOT, 'data_all/model/entry_panel_univ.parquet'))
        zp = pl[(pl['symbol'] == 'ZECUSDT') & (pl['Date'] >= w0) & (pl['Date'] <= w_end)]
        if len(zp):
            X = zp[m['feats']].astype('float32')
            pr = m['xgb'].predict_proba(X)[:, 1]
            i = int(np.argmax(pr))
            sv = shap.TreeExplainer(m['xgb']).shap_values(X.iloc[[i]])[0]
            reasons['LEGACY_FIXED_long_best'] = (zp['Date'].iloc[i], float(pr[i]), pd.Series(sv, index=m['feats']).sort_values(key=np.abs, ascending=False).head(5))
            reasons['LEGACY_FIXED_long_zec_rows_in_universe'] = len(zp)
        else:
            reasons['LEGACY_FIXED_long_zec_rows_in_universe'] = 0
    except Exception as e:
        reasons['shap_error'] = str(e)
    out = {'run': {k: v for k, v in z.items() if k != 'daily_close'}, 'zec_close': z['daily_close'], 'table': pd.DataFrame(rows),
           'false_positives': pd.DataFrame(fps), 'reasons': reasons}
    pickle.dump(out, open(os.path.join(CACHE, 'zec_test.pkl'), 'wb'))
    pd.set_option('display.width', 250)
    print(out['table'].to_string()); print(out['false_positives'].to_string())
    for k, v in reasons.items():
        print(k, v if not isinstance(v, pd.DataFrame) else v.round(4).to_string())


if __name__ == '__main__':
    main()
