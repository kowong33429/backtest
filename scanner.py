"""NEW scanner (spec Output 4): coins that pass the UP / DOWN model now, ranked by score, with the top-3 reasons.

Usable only after step 10: while config.yaml has time.HOLDOUT_LOCKED = true it refuses to score live data.
`--dev-dry-run` scores the LAST DEVELOPMENT DAY (2025-06-30) instead, to check the pipeline without the holdout.

    python scanner.py --dev-dry-run --target up --top 20
    python scanner.py --target up            # after step 10 only
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from common_eval.config import load_config                    # noqa: E402

FROZEN = os.path.join(ROOT, 'new_research', 'models_frozen')


def fit_final(target):
    """Final model on every labeled development row (label window inside development); early stopping on the
    last VALID_FRAC of time with the same purge as the walk-forward. Saved for the scanner / step 10."""
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from new_research.models import FEATS, fold_masks, _impute
    from new_research.run_all import panel
    cfg = load_config()
    p = panel()
    feats = FEATS[target]
    ts = pd.Timestamp(cfg['time']['FINAL_HOLDOUT_START'])
    fit, valid = fold_masks(p['day'], ts, cfg['new']['model_params']['VALID_FRAC'])
    fit &= p[target].notna().values
    valid &= p[target].notna().values
    prm = dict(cfg['new']['model_params']['xgb']); es = prm.pop('early_stopping_rounds')
    yf, yv = p.loc[fit, target].astype(int).values, p.loc[valid, target].astype(int).values
    xgb = XGBClassifier(**prm, scale_pos_weight=(len(yf) - yf.sum()) / max(1, yf.sum()), n_jobs=12, verbosity=0,
                        eval_metric='aucpr', early_stopping_rounds=es)
    xgb.fit(p.loc[fit, feats], yf, eval_set=[(p.loc[valid, feats], yv)], verbose=False)
    Zf, [Zv] = _impute(p.loc[fit, feats], [p.loc[valid, feats]])
    sc = StandardScaler().fit(Zf)
    lr = LogisticRegression(class_weight='balanced', **cfg['new']['model_params']['logistic']).fit(sc.transform(Zf), yf)
    top = cfg['new']['SIGNAL_TOP_FRAC']
    obj = {'target': target, 'feats': feats, 'xgb': xgb, 'logistic': lr, 'scaler': sc,
           'impute_median': p.loc[fit, feats].median(), 'impute_cols': list(Zf.columns),
           'thr_xgb': float(np.quantile(xgb.predict_proba(p.loc[valid, feats])[:, 1], 1 - top)),
           'thr_logistic': float(np.quantile(lr.predict_proba(sc.transform(Zv))[:, 1], 1 - top)),
           'fit_last_day': p.loc[fit, 'day'].max(), 'valid_days': (p.loc[valid, 'day'].min(), p.loc[valid, 'day'].max())}
    os.makedirs(FROZEN, exist_ok=True)
    pickle.dump(obj, open(os.path.join(FROZEN, f'dryrun_{target}.pkl'), 'wb'))
    return obj


def score(rows, m, model):
    X = rows[m['feats']]
    if model == 'xgb':
        import shap
        s = m['xgb'].predict_proba(X)[:, 1]
        sv = shap.TreeExplainer(m['xgb']).shap_values(X)
        contrib = pd.DataFrame(sv, columns=m['feats'], index=rows.index)
    else:
        Z = X.fillna(m['impute_median']).fillna(0.0)
        for c in m['impute_cols']:
            if c.endswith('_isna'):
                Z[c] = X[c[:-5]].isna().astype(float)
        Z = Z[m['impute_cols']]
        z = m['scaler'].transform(Z)
        s = m['logistic'].predict_proba(z)[:, 1]
        contrib = pd.DataFrame(z * m['logistic'].coef_[0], columns=m['impute_cols'], index=rows.index)
    top3 = contrib.apply(lambda r: ', '.join(f'{k} {v:+.2f}' for k, v in r.abs().sort_values(ascending=False).head(3)
                                             .rename(lambda k: k).items()), axis=1)
    return s, top3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['up', 'down'], default='up')
    ap.add_argument('--model', choices=['xgb', 'logistic'], default='logistic')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--dev-dry-run', action='store_true')
    ap.add_argument('--refit', action='store_true')
    a = ap.parse_args()
    cfg = load_config()
    if not a.dev_dry_run and cfg['time']['HOLDOUT_LOCKED']:
        raise SystemExit('scanner.py is disabled until step 10 (config.yaml time.HOLDOUT_LOCKED = true). '
                         'Use --dev-dry-run to check it on the last development day.')
    if not a.dev_dry_run:
        raise SystemExit('live scoring needs data after the holdout start; it is implemented in step 10 together with '
                         'the holdout data loader.')
    path = os.path.join(FROZEN, f'new_{a.target}.pkl')                # frozen at tag new-frozen (new_research/freeze.py)
    m = fit_final(a.target) if a.refit else pickle.load(open(path, 'rb'))
    from new_research.run_all import panel
    p = panel()
    day = p['day'].max()
    rows = p[p['day'] == day].set_index('symbol')
    if a.target == 'up':
        btc_ok = bool((rows['btc_trend'] > cfg['new']['BTC_TREND_MIN']).iloc[0])
        print(f'BTC regime filter (close > EMA1200): {"ON - UP allowed" if btc_ok else "OFF - UP signals would be suppressed"}')
    s, why = score(rows, m, a.model)
    out = pd.DataFrame({'score': s, 'passes': s >= m[f'thr_{a.model}'], 'top3_reasons': why,
                        'sector': rows['sector'], 'close': rows['close']}, index=rows.index).sort_values('score', ascending=False)
    print(f'{a.target.upper()} {a.model} on {day.date()} (dev dry run) | threshold {m[f"thr_{a.model}"]:.4f} | '
          f'{int(out.passes.sum())} of {len(out)} coins pass')
    print(out.head(a.top).to_string())


if __name__ == '__main__':
    main()
