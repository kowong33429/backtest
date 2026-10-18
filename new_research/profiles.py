"""Section 5: profile of past UP / DOWN events (development period, day <= DEV_END) instead of using ZEC as a template.

* event table (section 4 b) for the whole pre-holdout panel, with year / sector counts
* feature values at 1/7/30/90/180 days before each event start, as percentiles of all coin-days
* univariate separation (AUC of each feature, Mann-Whitney p-value, Bonferroni) event-start vs ordinary coin-days
* "old coin far below ATH in a long base" group, defined only by features
* k-means clusters of the pre-move state (standardized), k chosen by silhouette, size and hit rate of each
* the rule baseline form (3-5 conditions) derived automatically from the separation table
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from new_research.labels import events              # noqa: E402
from new_research.models import FEATS               # noqa: E402
from common_eval.config import load_config          # noqa: E402

CFG = load_config()
DEV_END = pd.Timestamp(CFG['time']['DEV_END'])
LAGS = [1, 7, 30, 90, 180]
CLUSTER_FEATS = ['pct_from_ath', 'pct_from_low_1y', 'range_width_180d', 'base_duration_days', 'close_ema1200',
                 'breakout_540', 'atr_pct_rank_1y', 'bb_width_rank_1y', 'qv_20d_180d', 'taker_buy_chg', 'obv_slope_30d',
                 'rs_30d_rank', 'rs_90d_rank', 'coin_age_days', 'funding_30d']


def event_tables(panel):
    cd = CFG['new']['label']['EVENT_COOLDOWN'] / 6
    d = panel.set_index('day')
    up, dn = events(d, 'up', cd, 'up'), events(d, 'down', cd, 'down')
    for e in (up, dn):
        e['period'] = np.where(pd.to_datetime(e['start']) <= DEV_END, 'development', 'post_dev')
    return up, dn


def counts(ev):
    by_year = ev.groupby('year').size().rename('events')
    by_sector = ev.groupby('sector').size().rename('events').sort_values(ascending=False)
    return by_year, by_sector


def zec_like_mask(df):
    """Old coin, far below ATH, long base - defined by features only (no ZEC prices)."""
    return ((df['coin_age_days'] >= 730) & (df['pct_from_ath'] <= -0.80) & (df['base_duration_days'] >= 90))


def lag_profile(panel, ev, feats):
    """Median percentile (vs all development coin-days) of each feature at each lag before event start."""
    dev = panel[panel['day'] <= DEV_END]
    ref = {f: np.sort(dev[f].dropna().values) for f in feats}
    key = panel.set_index(['symbol', 'day'])
    rows = []
    for lag in LAGS:
        idx = pd.MultiIndex.from_arrays([ev['symbol'], pd.to_datetime(ev['start']) - pd.Timedelta(days=lag)])
        sub = key.reindex(idx)
        for f in feats:
            v = sub[f].dropna().values
            if len(v) == 0 or len(ref[f]) == 0:
                continue
            pct = np.searchsorted(ref[f], v) / len(ref[f])
            rows.append({'lag_days': lag, 'feature': f, 'n': len(v), 'median_value': float(np.median(v)),
                         'median_percentile': float(np.median(pct))})
    return pd.DataFrame(rows)


def separation(panel, ev, feats, n_ref=200000, seed=0):
    """Feature at the event start (decision day = start) vs a random sample of other development coin-days."""
    from scipy.stats import mannwhitneyu
    dev = panel[panel['day'] <= DEV_END]
    key = dev.set_index(['symbol', 'day'])
    idx = pd.MultiIndex.from_arrays([ev['symbol'], pd.to_datetime(ev['start'])])
    pos = key.reindex(idx)
    neg = dev[dev['up'] == 0].sample(min(n_ref, int((dev['up'] == 0).sum())), random_state=seed)
    rows = []
    for f in feats:
        a, b = pos[f].dropna().values, neg[f].dropna().values
        if len(a) < 10:
            continue
        u, p = mannwhitneyu(a, b, alternative='two-sided')
        rows.append({'feature': f, 'n_event': len(a), 'event_median': float(np.median(a)), 'other_median': float(np.median(b)),
                     'auc': float(u / (len(a) * len(b))), 'p_value': float(p)})
    s = pd.DataFrame(rows)
    s['p_bonferroni'] = (s['p_value'] * len(s)).clip(upper=1)
    s['strength'] = (s['auc'] - 0.5).abs()
    return s.sort_values('strength', ascending=False)


def derive_rule(sep, panel, max_terms=4, corr_max=0.7, min_strength=0.05):
    """Greedy 3-5 condition rule from the separation table: strongest significant features, not collinear with
    one already chosen. Direction from the AUC; threshold re-fit per fold (quantile 0.25 / 0.75 of train positives)."""
    dev = panel[panel['day'] <= DEV_END]
    chosen = []
    for _, r in sep[(sep['p_bonferroni'] < 0.01) & (sep['strength'] >= min_strength)].iterrows():
        f = r['feature']
        if any(abs(dev[[f, g]].corr().iloc[0, 1]) > corr_max for g, _, _ in chosen):
            continue
        chosen.append((f, '>=', 0.25) if r['auc'] > 0.5 else (f, '<=', 0.75))
        if len(chosen) >= max_terms:
            break
    return chosen


def clusters(panel, ev, k_range=range(2, 9), seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    dev = panel[panel['day'] <= DEV_END]
    key = dev.set_index(['symbol', 'day'])
    X = key.reindex(pd.MultiIndex.from_arrays([ev['symbol'], pd.to_datetime(ev['start'])]))[CLUSTER_FEATS]
    ok = X.notna().mean(axis=1) >= 0.8
    X, e = X[ok.values], ev[ok.values].reset_index(drop=True)
    med = dev[CLUSTER_FEATS].median()
    lo, hi = dev[CLUSTER_FEATS].quantile(0.01), dev[CLUSTER_FEATS].quantile(0.99)   # winsorise (bad-tick tails)
    clipf = lambda D: D.clip(lo, hi, axis=1).fillna(med)
    sc = StandardScaler().fit(clipf(dev[CLUSTER_FEATS]).sample(min(300000, len(dev)), random_state=seed))
    Z = sc.transform(clipf(X))
    best, sil = None, {}
    for k in k_range:
        km = KMeans(k, n_init=20, random_state=seed).fit(Z)
        sil[k] = float(silhouette_score(Z, km.labels_))
        if best is None or sil[k] > sil[best[0]]:
            best = (k, km)
    k, km = best
    e['cluster'] = km.labels_
    cent = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=CLUSTER_FEATS)
    cent['n_events'] = e.groupby('cluster').size()
    cent_z = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_FEATS)
    return {'k': k, 'silhouette': sil, 'events': e, 'centroids': cent, 'centroids_z': cent_z, 'scaler': sc,
            'median': med, 'clip_lo': lo, 'clip_hi': hi, 'model': km}


def cluster_hit_rates(panel, cl, radius_q=0.10):
    """How often each profile occurs: the development coin-days nearest to a centroid (closest `radius_q` share of
    all coin-days) and the share of them that become UP events, vs the base rate."""
    dev = panel[(panel['day'] <= DEV_END) & panel['up'].notna()]
    Z = cl['scaler'].transform(dev[CLUSTER_FEATS].clip(cl['clip_lo'], cl['clip_hi'], axis=1).fillna(cl['median']))
    rows = []
    base = float(dev['up'].mean())
    for c in range(cl['k']):
        dist = np.sqrt(((Z - cl['centroids_z'].iloc[c].values) ** 2).sum(axis=1))
        r = np.quantile(dist, radius_q)
        near = dist <= r
        rows.append({'cluster': c, 'radius': float(r), 'coin_days_near': int(near.sum()),
                     'share_of_all_coin_days': float(near.mean()), 'up_rate_near': float(dev.loc[near, 'up'].mean()),
                     'base_rate': base, 'lift': float(dev.loc[near, 'up'].mean() / base)})
    return pd.DataFrame(rows)
