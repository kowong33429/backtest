"""Freeze every model that will be scored on the holdout (fit on development data only, nothing after 2025-06-30).

NEW  : UP rule + logistic, DOWN rule + logistic + xgb (scanner.fit_final logic: early stopping / thresholds from the
       last VALID_FRAC of development, rule thresholds from train positives).
LEGACY_FIXED long  (L4): legacy XGB params, 101 features (no slow macro), NaN kept, threshold 0.60 (literal).
LEGACY_FIXED short (S4): legacy short_v2 params, same feature rule, threshold = the S4 walk-forward threshold.
Train rows for LEGACY: Date < cut - EMBARGO and Label_EndDate < cut (same purge as common_eval.walkforward).
-> new_research/models_frozen/*.pkl + manifest.json
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config                      # noqa: E402
from common_eval import walkforward as wf                       # noqa: E402
from new_research.models import FEATS, fold_masks, fit_rule_thresholds   # noqa: E402

CFG = load_config()
OUT = os.path.join(ROOT, 'new_research', 'models_frozen')
CUT = pd.Timestamp(CFG['time']['FINAL_HOLDOUT_START'])


def freeze_new(target, models):
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from new_research.models import _impute
    from new_research.run_all import panel, load
    p = panel()
    feats = FEATS[target]
    fit, valid = fold_masks(p['day'], CUT, CFG['new']['model_params']['VALID_FRAC'])
    fit &= p[target].notna().values
    valid &= p[target].notna().values
    top = CFG['new']['SIGNAL_TOP_FRAC']
    yf, yv = p.loc[fit, target].astype(int).values, p.loc[valid, target].astype(int).values
    obj = {'target': target, 'feats': feats, 'fit_last_day': str(p.loc[fit, 'day'].max().date()),
           'valid_days': [str(p.loc[valid, 'day'].min().date()), str(p.loc[valid, 'day'].max().date())]}
    if 'rule' in models:
        rule = load('profiles')[f'rule_{target}']
        tr = p.loc[fit | valid].assign(_y=p[target])
        obj['rule'] = rule
        obj['rule_thresholds'] = fit_rule_thresholds(tr, rule)
        obj['thr_rule'] = 10.0
    if 'logistic' in models:
        Zf, [Zv] = _impute(p.loc[fit, feats], [p.loc[valid, feats]])
        sc = StandardScaler().fit(Zf)
        lr = LogisticRegression(class_weight='balanced', **CFG['new']['model_params']['logistic']).fit(sc.transform(Zf), yf)
        obj.update(logistic=lr, scaler=sc, impute_median=p.loc[fit, feats].median(), impute_cols=list(Zf.columns),
                   thr_logistic=float(np.quantile(lr.predict_proba(sc.transform(Zv))[:, 1], 1 - top)))
    if 'xgb' in models:
        prm = dict(CFG['new']['model_params']['xgb']); es = prm.pop('early_stopping_rounds')
        m = XGBClassifier(**prm, scale_pos_weight=(len(yf) - yf.sum()) / max(1, yf.sum()), n_jobs=12, verbosity=0,
                          eval_metric='aucpr', early_stopping_rounds=es)
        m.fit(p.loc[fit, feats], yf, eval_set=[(p.loc[valid, feats], yv)], verbose=False)
        obj.update(xgb=m, thr_xgb=float(np.quantile(m.predict_proba(p.loc[valid, feats])[:, 1], 1 - top)))
    return obj


def freeze_legacy(panel_path, params, thr, fillna0=False):
    from xgboost import XGBClassifier
    p = wf.load_panel(panel_path, CFG)                       # rows < cut only
    feats = [c for c in p.columns if c not in wf.NON_FEAT and c not in wf.SLOW_MACRO]
    emb = pd.Timedelta(hours=4 * CFG['horizons']['EMBARGO_BARS'])
    tr = (p['Date'] < CUT - emb) & (p['Label_EndDate'] < CUT)
    X = p.loc[tr, feats].astype('float32')
    if fillna0:
        X = X.fillna(0.0)
    y = p.loc[tr, 'Entry_Label'].astype(int).values
    m = XGBClassifier(**params, scale_pos_weight=(len(y) - y.sum()) / max(1, y.sum()), n_jobs=12, verbosity=0, eval_metric='aucpr')
    m.fit(X, y, sample_weight=p.loc[tr, 'Sample_Weight'].values)
    return {'xgb': m, 'feats': feats, 'thr': thr, 'train_last_day': str(p.loc[tr, 'Date'].max().date()), 'n_train': int(tr.sum())}


def main():
    from common_eval.run_short import SHORT_XGB
    os.makedirs(OUT, exist_ok=True)
    s4 = pickle.load(open(os.path.join(ROOT, 'common_eval', '.cache', 'short_results.pkl'), 'rb'))['S4_+M4 keep NaN']
    objs = {
        'new_up': freeze_new('up', ('rule', 'logistic')),
        'new_down': freeze_new('down', ('rule', 'logistic', 'xgb')),
        'legacy_fixed_long': freeze_legacy(os.path.join(ROOT, 'data_all/model/entry_panel_univ.parquet'),
                                           dict(CFG['model']['xgb']), CFG['signals']['LEGACY_THRESHOLD']),
        'legacy_fixed_short': freeze_legacy(os.path.join(ROOT, 'data_all/model/entry_panel_univ_short.parquet'),
                                            SHORT_XGB, float(s4['thr'])),
    }
    man = {}
    for k, o in objs.items():
        pickle.dump(o, open(os.path.join(OUT, f'{k}.pkl'), 'wb'))
        man[k] = {a: (v if isinstance(v, (str, int, float, list)) else str(v)) for a, v in o.items()
                  if a.startswith(('thr', 'fit_', 'valid', 'train', 'n_train', 'rule'))}
    json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1, default=str)
    print(json.dumps(man, indent=1, default=str))


if __name__ == '__main__':
    main()
