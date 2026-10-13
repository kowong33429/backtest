# 🚀 Crypto Entry/Exit Research Pipeline (pooled XGBoost + honest backtests)

A leakage-safe ML research pipeline that answers one question per bar — **"does a
big rally start here?"** — pools ~100 "ZEC-like" coins into one training panel,
trains an XGBoost entry classifier under **purged + embargoed walk-forward CV**,
and measures real P&L with an **out-of-sample** event-driven backtest. Every rule
in [`AGENTS.md`](./AGENTS.md) (strict N-1, macro publication lag, ffill-only) is
enforced end-to-end.

> **TL;DR — the baseline that works.** Entry model **m1 (+100% within 60 days,
> stop −40%)**, trade every bar whose out-of-sample probability **≥ 0.60**, exit
> on a **fixed +100% take-profit** (SL −40%, 60-day time stop). Out-of-sample:
> **2,154 trades · 43.2% win · +8.35% expectancy/trade · PF 1.53 · +$17,976**
> at $100/trade. This is the configuration the cleaned-up codebase keeps.

> ⚠️ **Read [Survivorship-free test](#-survivorship-free-test-2026-10) first.** Every result on
> the ~144-coin research basket is inflated: the basket was picked by `trend_scanner` from coins
> that *already* rallied. On a universe of every Binance USDT-M perp that ever existed (incl.
> delisted), the entry-model strategies below do **not** beat buying and holding BTC. The best
> honest result so far is the plain BTC trend filter (`--preset btc-trend`, +30%/yr, −25% DD).
> What changed and when: [Changelog](#-changelog).

> ℹ️ **Re-run after the macro leak fix (2026-10-13).** m1, short_v2 and m1_univ were rebuilt and
> retrained on the fixed macro features, and the tables marked *(re-run 2026-10-13)* were refreshed.
> Rows marked *pre-fix* come from models that were not retrained (m2, daily, original 30d, exit
> experiments) and from an older data snapshot. The paper record restarted on 2026-10-13. Details:
> [changelog](#2026-10-13--rebuild-after-the-macro-leak-fix).

---

## 🧯 Survivorship-free test (2026-10)

| Step | Script | Output |
|------|--------|--------|
| All 703 crypto USDT-M perps ever listed (active + delisted; stocks/ETFs/indices excluded), 4H perp prices + funding from data.binance.vision | `core/universe_download.py` | `data_all/<sym>/<SYM>_4h_full.csv` |
| Panel kept only while a coin is in the **point-in-time top-120 by 30d volume** (ranked with yesterday's volume) | `core/universe_panel.py` | `data_all/model/entry_panel_univ.parquet`, `data_all/volume_rank.parquet` |
| m1 retrained, same label / CV (6 purged folds, 3d embargo), `--lean` | `core/train_entry_model.py` | `data_all/model/m1_univ/` |
| Rotation on that universe | `core/rotation_backtest.py --root data_all --universe-top 120 --oof data_all/model/m1_univ/entry_oof.parquet` | |

Same window for every row: 2021-10-28 → 2026-10 (first OOS fold onward), costs 0.15%/side.
Rows marked *pre-fix* were not re-run after the macro leak fix.

| Setup | CAGR | Max DD (MTM) |
|-------|------|--------------|
| b7 on the research basket (spot prices, basket-trained m1) *(re-run 2026-10-13)* | +120.1% | −39.5% |
| b7, universe-trained m1, alts restricted to the basket coins *(pre-fix)* | +37.9% | −57.7% |
| b7, universe-trained m1, 144 **random** universe coins (5 draws, median) *(pre-fix)* | +14.9% | — |
| **b7 on the point-in-time top-120 universe** *(re-run 2026-10-13)* | **−7.9%** | −76.1% |
| b7-ens on the same universe *(re-run 2026-10-13)* | −7.2% | −76.7% |
| core only / v5 on the same universe *(pre-fix)* | +1.8% / −34.9% | −70.4% / −92.7% |
| **BTC buy & hold** | **+7.5%** | −76.7% |

Universe-trained m1 has pooled OOS PR-AUC **0.077 vs a 0.081 base rate** (pre-fix 0.073) — no
skill. Top-60 and unrestricted universes give the same picture. Most of the basket edge came from
hindsight coin selection plus a model trained on those coins.

**Catching more of the big runs (2026-10-13).** The rotation misses most mega-rallies on this
universe (31 of 166 episodes of +500% within 90 days touched, none captured ≥3x), largely because
a coin breaking out of a long base is not yet a top-volume coin (ZEC was ranked 200–300 by volume
while it traded $40–55). A breakout sleeve fixes the reach, but not the economics. Measured on the
data after the 2026-10-12 update, same window:

| Preset (`--root data_all --universe-top 120 --oof data_all/model/m1_univ/entry_oof.parquet`) | CAGR | Max DD | Episodes touched / ≥3x | ZEC 2025 entry |
|---|---|---|---|---|
| `btc-trend` — BTC while above SMA50 & SMA200, else cash | **+30.1%** | **−24.6%** | — | — |
| `greedy` — every liquid 90d breakout (vol ≥ 1.3x, ≥ $3M/day, ≥ 90d old), 1% each, + BTC trend | +27.2% | −39.9% | 75 / 17 | **$52.83** (sold $452.04, 8.6x) |
| `greedy-alts` — same breakouts, idle cash stays cash | +7.6% | −40.5% | 75 / 17 | $52.83 |
| `b7` (for reference; CAGR / DD re-run 2026-10-13, episodes / ZEC pre-fix) | −7.9% | −76.1% | 31 / 0 | $229.8 |

`greedy` is robust (10 random 70% coin subsets: median +28.2%, min +25.4%; 2x costs +23.4%;
one-day delay +23.6%; both 2021-23 and 2024-26 positive), but almost all of its return is the BTC
trend leg: the alt breakouts on their own make about +7.6%/yr, so adding them to `btc-trend`
*lowers* CAGR by ~3 points and deepens the drawdown by ~15. Use them only if you explicitly want
exposure to the big runners.

### Paper trading (forward record)

`core/rotation_paper.py` runs once a day after 00:00 UTC: it appends the newest closed bars,
scores them with the final model (history keeps the OOF probs), builds the book with the same
`build_weights` as the backtest, and saves the target for that day's Open. The paper P&L uses
only targets saved by earlier runs — it is never back-filled. The record restarted on 2026-10-13
with the retrained `m1_univ` (first target: the 2026-10-13 Open); the earlier record, made with the
pre-fix model, is archived in `data_all/model/_preleak_20261013/paper/`.

```bash
python core/rotation_paper.py --update --root data_all --universe-top 120 --preset b7-ens \
    --oof data_all/model/m1_univ/entry_oof.parquet --model-dir data_all/model/m1_univ --paper-dir data_all/paper
python core/rotation_paper.py --update --root data_all --universe-top 120 --preset btc-trend \
    --oof data_all/model/m1_univ/entry_oof.parquet --model-dir data_all/model/m1_univ --paper-dir data_all/paper/btc-trend
python core/rotation_paper.py --report --root data_all --universe-top 120 --oof data_all/model/m1_univ/entry_oof.parquet --paper-dir data_all/paper
```

---

## 📝 Changelog

All backtest numbers are out-of-sample; costs 0.10% fee + 0.05% slippage per side. "Basket" = the
144-coin research basket (survivorship-biased, see above); "universe" = `data_all`.

### 2026-10-13 — rebuild after the macro leak fix
- **Rebuilt and retrained** with the fixed `MacroSnapshot` (same label / CV / XGBoost args as before):
  m1 (`entry_models_v2.py --models m1_100pct_60d --force`), short_v2 (`dataset.py --direction short
  --tp 1.0 --sl 0.4 --horizon-days 30` → `train_entry_model.py --n-splits 5 --n-estimators 600
  --max-depth 5 --min-child-weight 50 --colsample 0.5`), m1_univ (`universe_panel.py` →
  `train_entry_model.py --n-splits 6 --lean`). Pre-fix models, OOFs and the old paper record are
  kept in `data/model/_preleak_20261013/` and `data_all/model/_preleak_20261013/`.
- **Feature set restored to the original 108** (`dataset.SCALE_FREE_PREFIXES`): `10a2ad9` had
  swapped Pos_In / Dist_VWAP / Dist_High / Dist_Low for Donchian columns, so the pre-fix m1 and
  short_v2 used 108 features but the pre-fix m1_univ used 96. All three now use the same 108; the
  Donchian columns are still computed for the short-label filters.
- **Fear & Greed can drop out silently**: `fetch_fear_greed` returns `None` on a timeout and the
  panel is built without `FearGreed` (happened once during this rebuild; the short panel was rebuilt).
  Check for `[Sentiment] Fear & Greed: N days` in the build log.
- **Old vs new on the same data and labels** (labels identical row for row). The README tables
  before this rebuild came from an older data snapshot, so the "old" column below is the pre-fix
  OOF re-scored on today's data, not the earlier table values.

  | Test | Pre-fix model | Fixed model |
  |---|---|---|
  | m1 OOS PR-AUC (base 0.141) | 0.187 | 0.224 (logistic 0.147) |
  | m1 thr 0.60, TP +100% / SL −40% / 60d | 2,148 tr · PF 1.43 · +$14,714 | 2,154 tr · PF 1.53 · +$17,976 |
  | m1 thr 0.90 | 421 tr · PF 1.53 · +$3,375 | 437 tr · PF 3.16 · +$10,989 |
  | short_v2 OOS PR-AUC (base 0.054) | — | 0.102 (logistic 0.096) |
  | short_v2 thr 0.50 (basket short backtest) | 712 tr · PF 3.30 · +$19,746 | 1,240 tr · PF 2.91 · +$27,947 |
  | short_v2 at ~460 trades (old thr 0.60 / new thr 0.80) | PF 3.64 · +$14,168 | PF 3.35 · +$13,863 |
  | Portfolio m1 thr 0.8 + short exit (`m1_short_exit`) | +40.0% CAGR · −85.8% MTM DD | +71.5% CAGR · −76.0% MTM DD |
  | m1_univ OOS PR-AUC (base 0.081) | 0.073 | 0.077 |
  | Rotation b7, universe top-120 | — | −7.9% CAGR · −76.1% DD |
  | Rotation b6 / b7 / b7-ens, basket, full period | — | +187.3% / +198.8% / +184.1% CAGR · −46.3% / −39.5% / −42.1% DD |

- **Read the m1 gain with care**: the fixed m1 scores higher than the leaky one, but its thr 0.90
  P&L is concentrated in 2020–21 (+$8.7k of +$11.0k; the first fold trains on only 11k rows), and
  the portfolio gain is almost all 2021 (+1119% vs +517%). The probabilities correlate only 0.78
  with the pre-fix ones, so this is mostly a different fit, not a proven improvement. The
  survivorship-free verdict is unchanged: m1_univ has no skill and rotation loses to BTC.
- `btc-trend` / `greedy` / `greedy-alts` (model-free or nearly so) re-run unchanged: +30.1% /
  +27.2% / +7.6%.

### 2026-10-13 — macro look-ahead fixes (review of `c542b8a` + `bee0883`)
- **Yahoo look-ahead (`entry_finder.MacroSnapshot`)**: a day-D close (SP500, VIX, DXY, GOLD, OIL,
  US10Y, US2Y) was stamped at D 00:00 and merged into every 4H bar of day D, but the US session
  closes ~21:00 UTC. Each close is now stamped D+1 00:00 (checked: VIX 2024-03-08 close appears on
  2024-03-09).
- **FRED lag per series (`MacroSnapshot.FRED_LAG_DAYS`)**: the single 35-day lag was too short for
  M2 (~18 days of leak) and CPI (~5 days). Now M2 60, CPI 45, FEDFUNDS 35, HY_SPREAD 2 days; every
  March 2024 value lands after its real release. `fred_lag_days=N` still forces one lag for all.
  HY_SPREAD moves from 35 to 2 days, so that feature changes noticeably (not a leak, just more timely).
- **`core/data_loader.py`** (used by `universal_researcher.py` / `research_agent.py` only): same
  Yahoo fix, FRED had **no** lag at all → per-series lags (also WALCL 2, RRP 1, T10YIE 1), FRED fetched
  120 days earlier so the lagged series cover the start; daily funding mean now stamped at the END
  of its day (`label='right'`) instead of the start.
- **`rotation_paper.py`**: `--preset` accepts `btc-trend`, `greedy`, `greedy-alts` (greedy loads quote
  volume for the breakout sleeve); `PAPER` folder excluded from the symbol list.
- **Still open (found in review, not fixed)**: backtests on `data_all` perps charge no funding (matters
  if traded on perps, not spot); `barrier` mode fills SL at the level even on a gap down (other modes
  use `min(Open, stop)`); `rotation_paper.paper_record` silently skips days with no saved target and
  uses non-drift turnover; breadth counts delisted coins as below SMA when `--universe-top` is off.

### 2026-10-13 — survivorship-free universe, paper trading, greedy breakouts
- **Added `core/universe_download.py`**: 4H perp klines + funding for all 703 crypto USDT-M perps ever
  listed (active and delisted) from data.binance.vision → `data_all/` (git-ignored). Stocks, ETFs,
  commodities, FX and index perps are excluded.
- **Added `core/universe_panel.py`**: entry panel kept only while a coin is in the point-in-time
  top-N by 30d median quote volume (ranked with yesterday's volume); writes `data_all/volume_rank.parquet`.
- **`train_entry_model.py --lean`**: float32 features and no Logistic baseline, for large panels.
  m1 retrained on the universe panel → `data_all/model/m1_univ/` (OOS PR-AUC 0.073 vs base 0.081).
- **`rotation_backtest.py`**: `--root`, `--universe-top N` (point-in-time liquid universe for core,
  leaders and breadth), `--breakout` sleeve (`--bo-lookback/-vol-x/-min-usd/-size/-min-age/-rank`),
  `--no-core`, presets `btc-trend`, `greedy`, `greedy-alts`. `main()` split into `get_parser`,
  `apply_preset`, `load_inputs`, `build_weights` so experiments reuse the production code.
- **`portfolio_backtest.py --root`** to run the m1 portfolio on `data_all`.
- **Added `core/rotation_paper.py`**: daily forward recorder (update bars → score new bars with the
  final model → same `build_weights` → save today's target; P&L only from earlier saved targets).
  Verified to reproduce the backtest weights exactly. First target saved for 2026-10-12.
- **Finding**: on the universe every rotation preset loses to BTC buy & hold; the edge on the basket
  was hindsight selection. `btc-trend` (+30.1%, −24.6% DD) is the best honest result; `greedy`
  buys ZEC at $52.83 (2025-09-14) and touches 75/166 mega-rallies but adds nothing over `btc-trend`.

### 2026-10-12 — rotation strategy and iteration log (basket)
- **Added `core/rotation_backtest.py`**: every 7 days hold the top-10 coins by the buy model's OOF
  prob (7d mean), keep while rank ≤ 20, breadth filter (alts only if > 50% of coins are above their
  50d SMA), idle cash → BTC. Basket: +165% CAGR / −58% DD vs m1 threshold portfolio +37% / −82%.
- **Leader sleeve** (`--leaders`): relative-strength leaders on their own trend, any regime, only from
  idle cash — catches lone runners such as ZEC 2025 when the core is switched off.
- **Presets from a 15-round iteration log** (each change kept only if full-period Calmar rose without
  either half dropping > 10%; later rounds also required the median of 10 random coin subsets to rise):
  `b6` (leaders enter/hold within 25% of the 60d high, 35% once +100%; young-coin sleeve; BTC
  fallback above SMA50 & SMA200; 80% vol target; core needs BTC > SMA100), `b7` (b6 with 3 leaders and
  70% vol target; basket +221.7% / −34.3%), `b7-ens` (b7 averaged over near-high bands 20/25/30%,
  because 25% was a sharp optimum). Rejected: breadth smoothing, exit confirmation, volume filter,
  blended core score, entry-only near-high filter.

### 2026-10-11 — short model, model-driven exits (commit `c542b8a`)
- **Short model v2**: `train_entry_model.py` gained `--n-estimators/--max-depth/--min-child-weight/
  --colsample/--seed`; 600 trees, depth 5, min_child_weight 50, colsample 0.5 → `data/model/short_v2`
  (basket short backtest at ~700 trades: PF 2.15 → 3.30, P&L +57%). New features (58 TA, BTC,
  cross-section), long-window mean-reversion features, SL-first weighting, label alignment and the
  correlation filter were all tested and **not** kept.
- **`core/exits.py`**: shared long-exit engine (barrier / trailing / short-model / exit-model) used
  by `entry_backtest.py` and `portfolio_backtest.py`; short-model exit = short prob ≥ thr AND Close ≥
  20% below the trade peak. `portfolio_backtest.py` now reports a daily mark-to-market drawdown.
- **`core/exit_model.py`**: dedicated exit model on in-trade bars — no skill (PR-AUC 0.51 vs base
  0.52), kept for reference only.
- `entry_backtest_short.py --sl-cooldown-days`, `plot_train_entries.py --sort`, plot colours for the
  new exit outcomes.

---

## 🏆 Baseline model (kept)

| Item | Value |
|------|-------|
| Entry label | `+100%` within `60d` before `−40%` (triple-barrier, path-aware) |
| Features | ~100 scale-free TA + macro + Fear&Greed + funding, strict `.shift(1)` |
| CV | Purged + embargoed walk-forward (`n_splits=6`, `embargo=3d`) |
| Signal | OOS `oof_prob_xgb ≥ 0.60` |
| Exit | Fixed **TP +100%** / **SL −40%** / **time 60d**, execute at next bar `Open` |
| Costs | 0.1% fee + 0.05% slippage **per side** (0.30% round-trip) |
| **Result (OOS)** | **2,154 trades · win 43.2% · exp +8.35%/trade · PF 1.53 · +$17,976** (re-run 2026-10-13) |

Artifacts: `data/model/v2/m1_100pct_60d/` (model `entry_xgb.json`, `entry_oof.parquet`,
`entry_model_meta.json`, SHAP `entry_shap.png`) and the baseline trade log in
`data/model/v2/m1_100pct_60d/thr06/entry_backtest_trades.csv`.

---

## 🧪 Everything we tried

All P&L figures are **out-of-sample** at **$100/trade**, costs 0.30% round-trip.
Entry models share one pooled panel and the same purged walk-forward CV; only the
label (or timeframe) differs. "Exit experiments" all reuse the **same m1 entry
signal (thr 0.60)** and change only how a trade is closed.

### Entry definitions

| Entry model | Label | TF | Thr | Trades | Win% | Exp%/tr | PF | Total P&L |
|-------------|-------|----|-----|-------:|-----:|--------:|---:|----------:|
| Original *(pre-fix)* | +100% / 30d / −40% | 4H | 0.60 | 708 | 45.5 | +8.73 | 1.69 | +$6,181 |
| **m1 (baseline)** | **+100% / 60d / −40%** | 4H | **0.60** | **2,154** | **43.2** | **+8.35** | **1.53** | **+$17,976** |
| m1 @ thr 0.70 | +100% / 60d / −40% | 4H | 0.70 | 1,575 | 45.1 | +10.34 | 1.68 | +$16,281 |
| m1 @ thr 0.90 | +100% / 60d / −40% | 4H | 0.90 | 437 | 56.3 | +25.15 | 3.16 | +$10,989 |
| m2 *(pre-fix)* | +50% / 30d / −20% | 4H | 0.80 | 1,020 | 39.9 | +3.28 | 1.31 | +$3,342 |
| m1 on daily *(pre-fix)* | +100% / 60d / −40% | 1D | 0.80 | 522 | 43.9 | +6.91 | 1.46 | +$3,607 |

**Takeaways:** the +100%/60d label on 4H is the sweet spot. Raising the threshold
lifts expectancy and win rate but cuts total P&L (fewer trades) — **0.60
maximizes total $**. The +50%/30d label and the daily rebuild are both weaker.
XGBoost's OOS PR-AUC (~0.22) only modestly beats the logistic baseline (~0.15),
so the edge is real but thin — the money comes from the asymmetric payoff, not
from a high win rate.

### Exit strategies (all on the m1 entry, thr 0.60)

*Pre-fix:* this whole table uses the pre-fix m1 and an older data snapshot; it was not re-run
after the macro leak fix. Compare rows with each other, not with the re-run baseline above.

| Exit strategy | Trades | Win% | Exp%/tr | PF | Total P&L | Notes |
|---------------|-------:|-----:|--------:|---:|----------:|-------|
| **Fixed TP +100% (baseline)** | **1,688** | **45.6** | **+9.25** | **1.62** | **+$15,606** | simple, robust |
| Daily climax (blow-off top), TP off | 1,530 | 44.8 | +11.25 | 1.78 | +$17,213 | fires rarely (68 climax exits); mostly time-stops |
| Daily climax + 30d time stop | 2,434 | 46.1 | +5.76 | 1.54 | +$14,027 | shorter holds, lower expectancy |
| Exhaustion (vol-climax + RSI divergence), hybrid | 6,170 | 79.0 | +1.63 | 1.35 | +$10,033 | exits far too early (avg hold ~10 bars) |
| Exhaustion, arm only after +30% gain | 2,112 | 52.2 | +7.06 | 1.51 | +$14,909 | arming delay fixes most of the early-exit bleed |
| ML exit model (double-off-trailing-low), pure | 4,396 | 72.0 | +2.48 | 1.41 | +$10,918 | high win rate, tiny per-trade edge |
| ML exit, arm after +50% gain | 1,802 | 47.8 | +7.25 | 1.49 | +$13,057 | best of the ML-exit variants |

**Takeaways:** no exit beat the fixed +100% TP on **total P&L** once you account
for how often it fires. The behavioural exits (exhaustion, ML-exit) raise the win
rate dramatically by closing winners early — but that caps the asymmetric upside
the strategy depends on, so total $ falls. The daily-climax exit edges ahead on
paper (+$17k) but only because it almost never triggers (it mostly collapses back
to the time stop); it adds complexity for a fragile gain. **Simple fixed TP wins.**

The `--arm-gain` sweep (`sweep_arm_gain.py`) is the clearest evidence: delaying any
behavioural exit until the trade is already up +30–60% recovers most of the P&L it
otherwise destroys, converging back toward the fixed-TP result.

### Exit *classifier* (standalone, not a trade test)

A 4H "the move has already doubled off its trailing-60d low" classifier
(`exit_models_4h.py`) scores PR-AUC 0.89 OOS — but this is **circularity, not
edge**: the label is ~a deterministic trailing-return rule and the n-1 features
encode the same thing. It is only meaningful when paired with the entry model
(the "ML exit" rows above), where it does not beat a fixed TP.

---

## 🔁 Reproduce the baseline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Universe + data (4H OHLCV for the scanned basket)
python core/trend_scanner.py
python core/batch_download.py --priority

# 2+3+4) Build panel -> train (purged walk-forward) -> backtest, for m1
python core/entry_models_v2.py --models m1_100pct_60d

# 5) The baseline backtest: thr 0.60, fixed +100% TP / -40% SL / 60d
python core/entry_backtest.py \
    --oof data/model/v2/m1_100pct_60d/entry_oof.parquet \
    --threshold 0.60 --tp 1.00 --sl 0.40 --horizon-days 60 \
    --out-dir data/model/v2/m1_100pct_60d/thr06

# 6) Visualize the actual trades
python core/plot_m1_trades.py \
    --trades data/model/v2/m1_100pct_60d/thr06/entry_backtest_trades.csv

# 7) Short model short_v2 (label: price halves = +100% short profit within 30d before -40%)
python core/dataset.py --direction short --tp 1.0 --sl 0.4 --horizon-days 30 \
    --out data/model/entry_panel_short.parquet
python core/train_entry_model.py --panel data/model/entry_panel_short.parquet \
    --n-splits 5 --embargo-days 3 --n-estimators 600 --max-depth 5 \
    --min-child-weight 50 --colsample 0.5 --no-shap --out-dir data/model/short_v2
python core/entry_backtest_short.py --oof data/model/short_v2/entry_oof.parquet \
    --save-threshold 0.5 --out-dir data/model/short_v2

# 8) Survivorship-free m1_univ (needs data_all/ from core/universe_download.py)
python core/universe_panel.py
python core/train_entry_model.py --panel data_all/model/entry_panel_univ.parquet \
    --n-splits 6 --embargo-days 3 --lean --no-shap --out-dir data_all/model/m1_univ

# 9) Portfolio: m1 entries, short-model exit
python core/portfolio_backtest.py --oof data/model/v2/m1_100pct_60d/entry_oof.parquet \
    --threshold 0.8 --exit-mode short --exit-probs data/model/short_v2/entry_oof.parquet \
    --exit-thr 0.5 --min-dd 0.20 --horizon-days 365 --out-dir data/model/v2/m1_short_exit
```

`entry_models_v2.py` is a thin orchestrator — it shells out to the audited
`dataset.py → train_entry_model.py → entry_backtest.py` so the leakage-safe path
is never re-implemented.

> **Check every panel build log** for `[Sentiment] Fear & Greed: N days` and
> `Macro table ready: 16 features`. If the Fear & Greed API times out, the panel is
> built **without** `FearGreed` and no error is raised. The trained model then has
> 107 features instead of 108. Re-run the build when that happens.

---

## 🤖 Use the pre-trained model on another machine (no retraining)

The baseline model is committed to git, so a fresh clone can score coins
**without** rebuilding the ~780 MB training panel. Only two files are needed and
both are in the repo:

```text
data/model/v2/m1_100pct_60d/
├── entry_xgb.json          # the trained XGBoost model (108 features)
└── entry_model_meta.json   # feature column order + default threshold (0.90) + OOS PR-AUC
```

> **Threshold note.** The meta's `threshold` is **0.90** — the trainer's
> precision-oriented pick (highest precision at recall ≥ 5%). The **baseline
> strategy** deliberately overrides this to **0.60**, which trades more and
> maximizes *total* P&L (see the experiment table). So for the baseline behavior,
> pass `--threshold 0.60` below.

> The parquet/CSV artifacts (`entry_panel.parquet`, `entry_oof.parquet`, trade
> logs) are **git-ignored** by design — they are regenerated, not versioned. You
> do not need them to run inference.

### 1. Clone + install

```bash
git clone <this-repo> && cd backtest
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download 4H OHLCV for the coin you want to score

The model consumes the same 4H feature pipeline it was trained on, so it needs a
`*_4h_full.csv` to score (features are rebuilt locally — only raw OHLCV travels):

```bash
python core/data_downloader.py --symbol ZECUSDT --interval 4h
# -> data/zecusdt/ZECUSDT_4h_full.csv
```

### 3. Score with the committed model

`predict_entry.py` loads `entry_xgb.json`, reads the feature order + threshold
from `entry_model_meta.json`, rebuilds the strict-N-1 features, and writes
per-bar probabilities + a chart:

```bash
python core/predict_entry.py \
    --csv data/zecusdt/ZECUSDT_4h_full.csv \
    --model-dir data/model/v2/m1_100pct_60d \
    --threshold 0.60                          # baseline cutoff (meta defaults to 0.90)
# -> entry_signals.csv + entry_signals.html next to the CSV,
#    and the latest bar's probability printed to the console.
```

Drop `--threshold` to use the meta's 0.90 (fewer, higher-precision signals). Add
`--no-macro` if the machine can't reach the macro/Fear&Greed sources — the model
still runs (those feature columns are filled with 0, a slight degradation).

### Load the model from JSON directly (minimal snippet)

If you want to wire the model into your own code instead of the CLI:

```python
import json
import pandas as pd
from xgboost import XGBClassifier

MODEL_DIR = "data/model/v2/m1_100pct_60d"
model = XGBClassifier()
model.load_model(f"{MODEL_DIR}/entry_xgb.json")          # build model from JSON
meta = json.load(open(f"{MODEL_DIR}/entry_model_meta.json"))
features  = meta["features"]     # 108 columns — order matters
threshold = 0.60                 # baseline cutoff (meta["threshold"] defaults to 0.90)

# X must be the SAME scale-free feature frame dataset.py builds (features.py +
# select_scale_free + macro/F&G/funding), columns in exactly `features` order.
# core/predict_entry.py:build_live_features() does this for you from a 4H CSV.
prob   = model.predict_proba(X[features])[:, 1]
signal = prob >= threshold
```

The only hard requirement is that `X` is built by **this repo's** feature
pipeline — the model expects those exact columns in that exact order. Reuse
`core/predict_entry.py` (`build_live_features`) rather than hand-rolling features,
or the scores will be meaningless.

---

## 📁 Project structure (baseline only)

```text
backtest/
├── AGENTS.md                  # The rulebook (READ THIS FIRST)
├── data/                      # Per-symbol OHLCV + model artifacts (git-ignored)
└── core/
    │   # ── data acquisition ──
    ├── trend_scanner.py       # Multi-exchange big-trend scanner -> universe basket
    ├── batch_download.py      # Bulk OHLCV downloader (shells out per coin)
    ├── data_downloader.py     # Download OHLCV + funding from Binance
    │   # ── feature / label / macro ──
    ├── features.py            # ~100 TA features, strict n-1 .shift(1)
    ├── labeler.py             # Forward triple-barrier entry label (+tp/-sl/horizon)
    ├── entry_finder.py        # MacroSnapshot + Fear&Greed helpers used by the panel
    ├── dataset.py             # Build pooled, leakage-safe entry panel (scale-free feats)
    │   # ── train / backtest / visualize ──
    ├── train_entry_model.py   # XGBoost + purged/embargoed walk-forward CV, OOF + SHAP
    ├── entry_models_v2.py     # Orchestrator: panel -> train -> backtest for m1
    ├── entry_backtest.py      # Honest OOS backtest (fixed TP/SL/time) — the baseline
    └── plot_m1_trades.py      # Plotly: the actual backtest trades on real price
```

### Model files (per baseline model dir, e.g. `data/model/v2/m1_100pct_60d/`)

| File | What it is |
|------|------------|
| `entry_xgb.json` | Trained XGBoost model (refit on all data, for live inference) |
| `entry_model_meta.json` | Feature list, chosen threshold, OOS PR-AUC, base rate, CV config |
| `entry_oof.parquet` | Out-of-fold probabilities per bar (what the backtest trades) |
| `entry_panel.parquet` | The pooled training panel (features + labels + weights) |
| `entry_features.txt` | The exact feature column list |
| `entry_shap.png` | SHAP summary — which features drive the model |
| `thr06/entry_backtest_trades.csv` | The baseline trade log (2,154 trades) |

### Visualization files

| File | Visualizes |
|------|------------|
| `core/plot_m1_trades.py` | The **backtest result** — every traded entry/exit on real price, colored by TP/SL/TIME, with an all-coins P&L overview (`→ m1_trades_viz.html`) |
| `data/model/v2/m1_100pct_60d/entry_shap.png` | Feature importance (SHAP) of the trained model |

---

## 📐 How the code implements the `AGENTS.md` rules

- **Rule #1 — N-1 shift:** `features.py` shifts every engineered feature by
  `.shift(1)`; funding is shifted in `dataset.py`. The only unshifted price is the
  execution `Open` of bar `n` (and the backtest executes at the *next* bar's Open).
- **Rule #2 — Macro publication lag:** `entry_finder.MacroSnapshot` applies a
  per-series FRED release lag (`FRED_LAG_DAYS`: M2 60, CPI 45, FEDFUNDS 35,
  HY_SPREAD 2 days), stamps each Yahoo close on the next day, and forward-fills
  only (`.ffill()`); macro/F&G are merged backward-asof so a bar never sees a
  future release.
- **Labeling:** `labeler.py` races a `+tp` / `−sl` / time triple-barrier forward
  from each bar's `Open`; same-bar TP&SL ambiguity resolves to SL (conservative).
- **Leakage-safe CV:** `train_entry_model.py` uses **purged + embargoed
  walk-forward** (López de Prado AFML ch. 7), not random K-Fold — training bars
  whose label resolves inside the test window are dropped.
- **Imbalance & weights:** `scale_pos_weight` from the pos/neg ratio; AFML
  average-uniqueness sample weights downweight overlapping forward labels.
- **Honest evaluation:** PR-AUC (not ROC-AUC) is the headline, reported OOS only,
  always against a logistic baseline XGBoost must beat.
