"""Section 7: similarity search against the section-5 cluster centroids (never ZEC).

(a) Euclidean distance on the standardized cluster feature vector to the nearest centroid.
(b) DTW on the 90-day log price path and log quote-volume path (5-day steps, 18 points) to each cluster's median
    pre-event path. Vectorised over all candidate coin-days.
Matches are ranked per method; the report keeps the best day per coin (top 20) and the hit rate of the top-N.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from new_research.data import load_coin                  # noqa: E402
from new_research.profiles import CLUSTER_FEATS           # noqa: E402

STEP, NPTS = 5, 18          # 90 days


def daily_close_qv(symbols):
    out = {}
    for s in symbols:
        d = load_coin(s)
        if d is None:
            continue
        g = d.groupby(d.index.normalize())
        out[s] = pd.DataFrame({'close': g['Close'].last(), 'qv': g['Quote_Volume'].sum()})
    return out


def paths(dq, days):
    """(n, NPTS) log price path relative to the day itself and log qv (5-day sums) relative to the window median."""
    c = dq['close'].reindex(pd.date_range(dq.index.min(), dq.index.max())).ffill()
    q = dq['qv'].reindex(c.index).fillna(0)
    q5 = q.rolling(STEP).sum()
    P, Q = [], []
    for d in days:
        pts = [d - pd.Timedelta(days=STEP * k) for k in range(NPTS - 1, -1, -1)]
        pc, pq = c.reindex(pts).values, q5.reindex(pts).values
        P.append(np.log(pc / c.get(d, np.nan)))
        Q.append(np.log((pq + 1) / (np.nanmedian(pq) + 1)))
    return np.array(P), np.array(Q)


def dtw(A, t):
    """DTW distance between each row of A (n, m) and template t (m,), vectorised over rows."""
    n, m = A.shape
    prev = np.full((n, m + 1), np.inf)                 # row i-1 of the DP table (O(n*m) memory)
    prev[:, 0] = 0
    for i in range(1, m + 1):
        cur = np.full((n, m + 1), np.inf)
        for j in range(1, m + 1):
            cost = np.abs(A[:, i - 1] - t[j - 1])
            cur[:, j] = cost + np.minimum(np.minimum(prev[:, j], cur[:, j - 1]), prev[:, j - 1])
        prev = cur
    return prev[:, m]


def templates(cl, dq_map):
    ev = cl['events']
    tp = {}
    for c in range(cl['k']):
        P, Q = [], []
        for _, r in ev[ev['cluster'] == c].iterrows():
            if r['symbol'] in dq_map:
                p, q = paths(dq_map[r['symbol']], [pd.Timestamp(r['start'])])
                if np.isfinite(p).all() and np.isfinite(q).all():
                    P.append(p[0]); Q.append(q[0])
        tp[c] = (np.median(P, axis=0), np.median(Q, axis=0), len(P))
    return tp


def search(panel, cl, start, end, dq_map=None):
    cand = panel[(panel['day'] >= pd.Timestamp(start)) & (panel['day'] <= pd.Timestamp(end))].copy()
    Z = cl['scaler'].transform(cand[CLUSTER_FEATS].clip(cl['clip_lo'], cl['clip_hi'], axis=1).fillna(cl['median']))
    dist = np.stack([np.sqrt(((Z - cl['centroids_z'].iloc[c].values) ** 2).sum(axis=1)) for c in range(cl['k'])], 1)
    cand['feat_dist'] = dist.min(1)
    cand['feat_cluster'] = dist.argmin(1)
    dq_map = dq_map or daily_close_qv(sorted(cand['symbol'].unique()))
    tp = templates(cl, dq_map)
    P, Q, keep = [], [], []
    for s, g in cand.groupby('symbol'):
        if s not in dq_map:
            continue
        p, q = paths(dq_map[s], pd.DatetimeIndex(g['day']))
        P.append(p); Q.append(q); keep.append(g.index.values)
    P, Q, keep = np.vstack(P), np.vstack(Q), np.concatenate(keep)
    ok = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
    best = np.full(len(P), np.inf)
    for c, (tpp, tpq, n) in tp.items():
        d = np.full(len(P), np.inf)
        d[ok] = dtw(P[ok], tpp) + dtw(Q[ok], tpq)
        best = np.minimum(best, d)
    cand['dtw_dist'] = np.inf
    cand.loc[keep, 'dtw_dist'] = best
    cand['combined_rank'] = cand['feat_dist'].rank(pct=True) + cand['dtw_dist'].rank(pct=True)
    return cand, tp


def top_matches(cand, col, n=20):
    best = cand.sort_values(col).groupby('symbol').head(1).head(n)
    return best[['day', 'symbol', 'sector', 'close', col, 'up', 'up_peak']]


def hit_curve(cand, col, ns=(20, 100, 500, 2000)):
    lab = cand[cand['up'].notna()].sort_values(col)
    base = float(lab['up'].mean())
    return pd.DataFrame([{'method': col, 'top_n': k, 'hit_rate': float(lab['up'].head(k).mean()), 'base_rate': base,
                          'lift': float(lab['up'].head(k).mean() / base)} for k in ns])
