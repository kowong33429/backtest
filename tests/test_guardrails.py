"""Section 9 unit tests (must pass before any NEW backtest). Run: python -m pytest tests -q"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config, guard_holdout          # noqa: E402
from common_eval.backtest import Engine                            # noqa: E402
from new_research.features import coin_features                    # noqa: E402
from new_research.labels import label_bars                         # noqa: E402
from new_research import models as M                               # noqa: E402

CFG = load_config()


class _MK:                                   # minimal stand-in for common_eval.data.Market
    def __init__(self):
        z = np.ones((3, 1))
        self.O = self.H = self.L = self.C = z
        self.T, self.N = 3, 1


def _engine(fee=0.0, short_pnl='true'):
    c = copy.deepcopy(CFG)
    c['costs']['FEE_PER_SIDE'] = fee
    return Engine(_MK(), c, short_pnl=short_pnl)


# ---------------- P&L formulas ----------------
def test_long_pnl_hand_numbers():
    e = _engine()
    assert e.net_ret(1, 100, 150, 0, 0) == pytest.approx(0.50)
    assert e.net_ret(1, 100, 80, 0, 0) == pytest.approx(-0.20)


def test_short_pnl_hand_numbers():
    e = _engine()
    assert e.net_ret(-1, 100, 80, 0, 0) == pytest.approx(0.20)       # short 100 -> 80 = +20%
    assert e.net_ret(-1, 100, 150, 0, 0) == pytest.approx(-0.50)     # short 100 -> 150 = -50%
    assert e.net_ret(-1, 100, 50, 0, 0) == pytest.approx(0.50)       # a halving is +50%, not +100% (bug H3)


def test_legacy_reciprocal_is_flagged_as_wrong():
    e = _engine(short_pnl='legacy_reciprocal')
    assert e.net_ret(-1, 100, 50, 0, 0) == pytest.approx(1.00)       # the LEGACY bug, kept only for AS_IS


def test_costs_reduce_both_sides():
    e = _engine(fee=0.001)
    # long, no move: (1-s)/(1+s) * (1-f)^2 - 1 with s=0.002 -> about -0.6%
    assert e.net_ret(1, 100, 100, 0.002, 0.002) == pytest.approx((0.998 / 1.002) * 0.999 ** 2 - 1)
    assert e.net_ret(-1, 100, 100, 0.002, 0.002) < 0


# ---------------- look-ahead ----------------
def _fake_4h(n=6 * 400, seed=0):
    rng = np.random.default_rng(seed)
    c = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    v = rng.uniform(100, 1000, n)
    idx = pd.date_range('2021-01-01', periods=n, freq='4h')
    return pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v, 'Quote_Volume': v * c,
                         'Trades': rng.integers(50, 500, n).astype(float),'Taker_Buy_Volume': v * rng.uniform(0.3, 0.7, n),
                         'Taker_Buy_Quote_Volume': v * c * 0.5, 'Funding_Rate': rng.normal(0, 1e-4, n),
                         'has_futures': 1}, index=idx)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_features_do_not_see_the_future(seed):
    d = _fake_4h(seed=seed)
    base = coin_features(d)
    rng = np.random.default_rng(seed + 10)
    t = int(rng.integers(1500, len(d) - 50))
    d2 = d.copy()
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote_Volume', 'Trades', 'Taker_Buy_Volume', 'Funding_Rate']
    d2.iloc[t + 1:, [d2.columns.get_loc(c) for c in cols]] *= rng.uniform(0.2, 5.0, (len(d) - t - 1, len(cols)))
    d2['High'] = d2[['Open', 'High', 'Close']].max(axis=1)
    d2['Low'] = d2[['Open', 'Low', 'Close']].min(axis=1)
    pert = coin_features(d2)
    pd.testing.assert_frame_equal(base.iloc[:t + 1], pert.iloc[:t + 1])      # rows <= t unchanged
    assert not base.iloc[t + 1:].equals(pert.iloc[t + 1:])                    # rows > t do react


# ---------------- labels ----------------
def _path(closes, highs, lows):
    n = len(closes)
    return pd.DataFrame({'Close': closes, 'High': highs, 'Low': lows}, index=pd.date_range('2022-01-01', periods=n, freq='4h'))


def test_up_label_hits_target_before_mae():
    d = _path([100, 120, 150, 190], [100, 130, 160, 205], [100, 95, 110, 150])
    up, dn, *_ = label_bars(d, np.array([0]), 3, 2.0, 0.5, 0.25, delisted=False)
    assert up[0] == 1 and dn[0] == 0


def test_up_label_fails_when_mae_breached_first():
    d = _path([100, 80, 150, 190], [100, 90, 160, 205], [100, 70, 110, 150])   # low 70 < 75 first
    up, *_ = label_bars(d, np.array([0]), 3, 2.0, 0.5, 0.25, delisted=False)
    assert up[0] == 0


def test_down_label_and_same_bar_ambiguity():
    d = _path([100, 90, 60, 45], [100, 110, 95, 70], [100, 85, 55, 40])
    _, dn, *_ = label_bars(d, np.array([0]), 3, 2.0, 0.5, 0.25, delisted=False)
    assert dn[0] == 1
    d = _path([100, 60, 45], [100, 130, 50], [100, 45, 40])                    # +30% and -55% in the same bar
    _, dn, *_ = label_bars(d, np.array([0]), 2, 2.0, 0.5, 0.25, delisted=False)
    assert dn[0] == 0


def test_label_is_nan_when_window_needs_unseen_prices():
    d = _path([100, 110, 120], [100, 115, 125], [100, 105, 115])
    up, *_ = label_bars(d, np.array([0]), 5, 2.0, 0.5, 0.25, delisted=False)
    assert np.isnan(up[0])
    up, *_ = label_bars(d, np.array([0]), 5, 2.0, 0.5, 0.25, delisted=True)  # delisted coin: outcome is known
    assert up[0] == 0


# ---------------- purge / embargo ----------------
def test_new_folds_purge_and_embargo():
    days = pd.date_range('2020-06-01', '2025-06-30', freq='D')
    from common_eval.walkforward import blocks
    for ts, te in blocks(CFG):
        fit, valid = M.fold_masks(days, ts, 0.2)
        tr = fit | valid
        assert (days[tr] + pd.Timedelta(days=1) + M.H_TD < ts).all()         # label horizon ends before the test
        assert (days[tr] < ts - M.EMB_TD).all()                              # embargo
        if valid.any() and fit.any():
            assert (days[fit] + pd.Timedelta(days=1) + M.H_TD < days[valid].min()).all()   # fit purged from valid


def test_common_eval_retrain_purge():
    from common_eval.walkforward import blocks
    emb = pd.Timedelta(hours=4 * CFG['horizons']['EMBARGO_BARS'])
    d = pd.date_range('2021-01-01', '2025-06-30', freq='D')
    le = d + pd.Timedelta(days=60)
    for ts, te in blocks(CFG):
        tr = (d < ts - emb) & (le < ts)
        assert (le[tr] < ts).all() and (d[tr] < ts - emb).all()


# ---------------- holdout lock ----------------
def test_holdout_guard():
    with pytest.raises(PermissionError):
        guard_holdout(CFG, '2025-07-01')
    guard_holdout(CFG, '2025-06-30 20:00')
