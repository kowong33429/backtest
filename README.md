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
> **1,688 trades · 45.6% win · +9.25% expectancy/trade · PF 1.62 · +$15,606**
> at $100/trade. This is the configuration the cleaned-up codebase keeps.

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
| **Result (OOS)** | **1,688 trades · win 45.6% · exp +9.25%/trade · PF 1.62 · +$15,606** |

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
| Original | +100% / 30d / −40% | 4H | 0.60 | 708 | 45.5 | +8.73 | 1.69 | +$6,181 |
| **m1 (baseline)** | **+100% / 60d / −40%** | 4H | **0.60** | **1,688** | **45.6** | **+9.25** | **1.62** | **+$15,606** |
| m1 @ thr 0.70 | +100% / 60d / −40% | 4H | 0.70 | 1,322 | 47.2 | +10.07 | 1.68 | +$13,310 |
| m1 @ thr 0.90 | +100% / 60d / −40% | 4H | 0.90 | 494 | 48.2 | +13.96 | 1.99 | +$6,896 |
| m2 | +50% / 30d / −20% | 4H | 0.80 | 1,020 | 39.9 | +3.28 | 1.31 | +$3,342 |
| m1 on daily | +100% / 60d / −40% | 1D | 0.80 | 522 | 43.9 | +6.91 | 1.46 | +$3,607 |

**Takeaways:** the +100%/60d label on 4H is the sweet spot. Raising the threshold
lifts expectancy and win rate but cuts total P&L (fewer trades) — **0.60
maximizes total $**. The +50%/30d label and the daily rebuild are both weaker.
XGBoost's OOS PR-AUC (~0.19) only modestly beats the logistic baseline (~0.15),
so the edge is real but thin — the money comes from the asymmetric payoff, not
from a high win rate.

### Exit strategies (all on the m1 entry, thr 0.60)

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

## ⚠️ AI-optimized exits (Optuna): a leakage + overfitting cautionary tale

Four "quant" exit optimizers (`optimize_exits.py`, `optimize_moneyflow.py`,
`optimize_finance.py`, `optimize_econ.py`) use Optuna to search exit parameters
(ATR-trailing multiples, money-flow thresholds, econ regime rules, horizons) to
**maximize total P&L**. On first write they looked spectacular — **$20k–29k** —
but that number was produced two illegal ways at once:

1. **Look-ahead leakage** — indicators filled with `.ffill().bfill()` (bfill pulls
   future values into the warm-up window) and the macro/regime state read from the
   *execution* candle `entry_i` instead of the signal candle `entry_i-1`.
2. **No hold-out** — Optuna optimized and reported P&L on the **entire** dataset,
   so the "best" parameters were curve-fit to the very data they were scored on.

Both were fixed (ffill-only per [`AGENTS.md`](./AGENTS.md), regime read at `n-1`,
and a chronological **70/30 train→validation split** — tune on `≤ 2024-04-20`,
report on the unseen remainder). The result is decisive:

| Optimizer (m1 entry, thr 0.70) | 🟥 BEFORE — "cheating"<br>in-sample, whole set, +leakage | 🟩 AFTER — train<br>(in-sample 70%, no leak) | 🟩 AFTER — **validation**<br>(**true OOS 30%**, no leak) |
|---|--:|--:|--:|
| `optimize_exits` (ATR trailing) | **+$20,708** | +$24,665 | **−$2,026** |
| `optimize_moneyflow` (CMF/MFI) | **+$22,381** | +$21,625 | **−$1,865** |
| `optimize_finance` (sector rotation) | **+$22,518** | +$24,538 | **−$2,509** |
| *baseline Fixed TP (reference)* | — | *+$13,740* | *−$2,164* |

> The BEFORE column reproduces the headline figures (an earlier run reported
> ~$27k–29k; Optuna is stochastic without a fixed seed, so magnitudes wander but
> the story doesn't). Train P&L still looks great **after** the fix — that is
> exactly the trap. Only the **validation** column is honest.

**Takeaways.**
- Every optimized exit **loses money out-of-sample** (−$1.9k to −$2.5k). The
  $20k–29k edge was an illusion of leakage + curve-fitting, not a real strategy.
- On the *same* validation window a plain fixed TP is also negative (−$2,164) —
  that period (2024-H2 → 2025 drawdown) is a hard regime for everyone — so the
  tuned exits add **no edge over the baseline** even on their own terms.
- This is the whole reason the pipeline mandates strict N-1 + out-of-sample
  testing: in-sample optimization will happily manufacture a five-figure "profit"
  that evaporates the moment it meets unseen data. **The simple fixed TP baseline
  remains the only configuration that survives honest evaluation.**

### Year-by-year balance — the "edge" is a single outlier year

Same entry everywhere (m1 OOF signal, enter at next `Open`); only the exit differs.
`$100/trade`, 0.30% round-trip. Optimizers use the best params their honest search
found. Reproduce with `python core/yearly_compare.py --threshold {0.70|0.60}`.

**Threshold 0.70** (matches the optimizers' built-in entry filter):

| Strategy | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | **TOTAL** | **ex-2020** |
|----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|----------:|------------:|
| **baseline fixed TP+100%** | 2,736 | 3,590 | −45 | 4,154 | 1,842 | 330 | 703 | 13,310 | **10,574** |
| + optimize_exits | 15,181 | 2,197 | −1,024 | 4,201 | 1,663 | 367 | 624 | 23,209 | 8,028 |
| + optimize_moneyflow | 12,075 | 3,179 | −1,597 | 3,982 | 2,621 | 51 | −131 | 20,181 | 8,106 |
| + optimize_finance | 14,111 | 3,885 | −1,486 | 4,018 | 1,553 | 104 | 232 | 22,418 | 8,307 |

**Threshold 0.60** (the kept baseline's actual cutoff — maximizes total $):

| Strategy | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | **TOTAL** | **ex-2020** |
|----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|----------:|------------:|
| **baseline fixed TP+100%** | 2,925 | 4,788 | −710 | 4,151 | 2,563 | −1,109 | 2,998 | 15,606 | **12,681** |
| + optimize_exits | 15,158 | 3,387 | −1,349 | 4,274 | 1,833 | −135 | 537 | 23,705 | 8,547 |
| + optimize_moneyflow | 12,200 | 3,686 | −2,118 | 4,185 | 2,634 | −346 | 905 | 21,145 | 8,945 |
| + optimize_finance | 14,176 | 4,181 | −2,212 | 4,303 | 1,492 | −498 | 1,326 | 22,768 | 8,592 |

**The optimizers' entire lead comes from 2020 alone** — one tail year (sparse early
listings, a few monster pumps) that long-horizon loose-stop trailing happened to
curve-fit. Strip 2020 and the plain **fixed TP wins outright at both thresholds**
(ex-2020: 10.6k/12.7k vs ~8–9k), while trading 2–3× fewer times and staying positive
in almost every year — the optimized exits lose **−$1k to −$2.2k in 2022** and sag
through the 2024-H2→2025 out-of-sample window. A larger total $ built on one
irreproducible year is not an edge.

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
```

`entry_models_v2.py` is a thin orchestrator — it shells out to the audited
`dataset.py → train_entry_model.py → entry_backtest.py` so the leakage-safe path
is never re-implemented.

---

## 🤖 Use the pre-trained model on another machine (no retraining)

The baseline model is committed to git, so a fresh clone can score coins
**without** rebuilding the 634 MB training panel. Only two files are needed and
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
| `thr06/entry_backtest_trades.csv` | The baseline trade log (1,688 trades) |

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
- **Rule #2 — Macro publication lag:** `entry_finder.MacroSnapshot` applies the
  FRED release lag and forward-fills only (`.ffill()`); macro/F&G are merged
  backward-asof so a bar never sees a future release.
- **Labeling:** `labeler.py` races a `+tp` / `−sl` / time triple-barrier forward
  from each bar's `Open`; same-bar TP&SL ambiguity resolves to SL (conservative).
- **Leakage-safe CV:** `train_entry_model.py` uses **purged + embargoed
  walk-forward** (López de Prado AFML ch. 7), not random K-Fold — training bars
  whose label resolves inside the test window are dropped.
- **Imbalance & weights:** `scale_pos_weight` from the pos/neg ratio; AFML
  average-uniqueness sample weights downweight overlapping forward labels.
- **Honest evaluation:** PR-AUC (not ROC-AUC) is the headline, reported OOS only,
  always against a logistic baseline XGBoost must beat.
