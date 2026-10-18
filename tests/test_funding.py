"""Funding tests for common_eval (approved 2026-10-18). Only settlements with entry_time < t < exit_time count."""
import copy
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common_eval.config import load_config          # noqa: E402
from common_eval.backtest import Engine             # noqa: E402

CFG = load_config()
T0 = pd.Timestamp('2024-01-01 00:00')
H4 = pd.Timedelta(hours=4)


class MK:
    """3-4 bars at 100, funding events given as (timestamp, rate)."""
    def __init__(self, events, n=3, close=None):
        self.times = pd.date_range(T0, periods=n, freq='4h')
        self.O = np.full((n, 1), 100.0); self.H = self.O; self.L = self.O
        self.C = np.full((n, 1), 100.0) if close is None else np.array(close, float).reshape(-1, 1)
        self.T, self.N = n, 1
        ts = np.array([pd.Timestamp(t).value for t, _ in events], dtype=np.int64)
        self.fund_events = {0: (ts, np.array([r for _, r in events], float))}


def eng(events, cfg=None, **kw):
    return Engine(MK(events, **{k: v for k, v in kw.items() if k in ('n', 'close')}), copy.deepcopy(cfg or CFG),
                  **{k: v for k, v in kw.items() if k == 'funding'})


def test_short_receives_positive_and_pays_negative():
    e = eng([(T0 + H4 * 0.5, 0.001), (T0 + H4 * 1.5, 0.001), (T0 + H4 * 2.5, -0.0005)])
    assert e.funding_path(-1, 0, 0, 2, 'TIME') == pytest.approx([0.001, 0.002, 0.0015])


def test_long_not_charged_when_only_short_is_configured():
    e = eng([(T0 + H4 * 0.5, 0.001)])
    assert e.funding_path(1, 0, 0, 2, 'TIME') is None


def test_long_pays_positive_funding_if_enabled():
    c = copy.deepcopy(CFG); c['costs']['FUNDING_SIDES'] = ['long', 'short']
    e = eng([(T0 + H4 * 0.5, 0.001), (T0 + H4 * 1.5, 0.001)], cfg=c)
    assert e.funding_path(1, 0, 0, 2, 'TIME')[-1] == pytest.approx(-0.002)


def test_intrabar_stop_exit_excludes_settlements_of_the_exit_bar():
    e = eng([(T0 + H4 * 0.5, 0.001), (T0 + H4 * 1.5, 0.001), (T0 + H4 * 2.5, 0.001)])
    assert e.funding_path(-1, 0, 0, 2, 'TRAIL')[-1] == pytest.approx(0.002)


def test_funding_scales_with_price():
    e = eng([(T0 + H4 * 0.5, 0.001), (T0 + H4 * 1.5, 0.001)], n=2, close=[100.0, 50.0])
    assert e.funding_path(-1, 0, 0, 1, 'TIME')[-1] == pytest.approx(0.001 + 0.001 * 0.5)


def test_off_switch_for_as_is_runs():
    e = eng([(T0 + H4 * 0.5, 0.001)], funding=False)
    assert e.funding_path(-1, 0, 0, 0, 'TIME') is None


def test_only_settlements_strictly_between_entry_and_exit_count():
    """Requested 2026-10-18: a settlement before the entry (even in the same bar / exactly at the entry Open) and one
    exactly at the exit time must not count; one just after the entry does."""
    entry = T0 + H4                                   # position opened at the Open of bar 1
    exit_close = T0 + H4 * 3                          # time-stop exit at the Close of bar 2
    ev = [(entry - pd.Timedelta(minutes=1), 0.05),     # before entry
          (entry, 0.05),                               # exactly at the entry Open
          (entry + pd.Timedelta(seconds=1), 0.001),    # just after entry -> counts
          (exit_close, 0.05),                          # exactly at the exit -> excluded
          (exit_close + pd.Timedelta(minutes=1), 0.05)]
    e = eng(ev, n=4)
    assert e.funding_path(-1, 1, 0, 2, 'TIME')[-1] == pytest.approx(0.001)
