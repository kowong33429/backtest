# 🚀 Agentic Quant System (XGBoost Long/Short Pipeline)

A production-oriented ML trading research pipeline for Crypto, built around the
5-step architecture and critical rules defined in [`AGENTS.md`](./AGENTS.md).
An LLM-driven agent (LangGraph + Gemini) sweeps the Triple-Barrier targets while
XGBoost trains separate Long/Short entry models, and an event-driven backtester
simulates realistic execution.

---

## 🧭 Pipeline Overview (see `AGENTS.md` for the full ruleset)

| Step | Module | Timeframe | Purpose |
|------|--------|-----------|---------|
| 1. Universe / Scan | `trend_scanner.py`, `batch_download.py` | 1D | Find & download candidate coins |
| 2. Feature Engineering | `features.py` | 4H | Stationary, Fibonacci/log-spaced features + macro |
| 3. Dynamic Triple-Barrier Labeling | `labels.py` | 4H | Long/Short targets |
| 4. Entry Models | `optimizer.py` | 4H | 2 XGBoost binary classifiers (Long & Short) |
| 5. Exit Engine & Backtest | `backtester.py`, `visualizer.py` | 4H | ATR trailing exits + realistic simulation |

The whole loop for a single symbol is orchestrated by `research_agent.py`.

---

## 📁 Project Structure

```text
backtest/
├── AGENTS.md                 # The rulebook (READ THIS FIRST)
├── data/                     # Per-symbol OHLCV + outputs (git-ignored)
│   └── zecusdt/
│       └── ZECUSDT_4h_full.csv
└── core/
    ├── data_downloader.py    # Download OHLCV + funding from Binance
    ├── data_loader.py        # Merge OHLCV with Macro (Yahoo/FRED) + Fear&Greed
    ├── features.py           # Feature engineering (Fibonacci/log windows, N-1 shift)
    ├── labels.py             # Dynamic Triple-Barrier labeling (Long/Short)
    ├── optimizer.py          # XGBoost + PurgedKFold-style TimeSeriesSplit, corr filter
    ├── backtester.py         # ATR SL + trailing-stop event-driven backtester
    ├── visualizer.py         # Plotly charts (signals, SHAP, correlation, equity)
    ├── research_agent.py     # LangGraph agent — orchestrates the full loop
    ├── trend_scanner.py      # Multi-exchange big-trend scanner (universe candidates)
    ├── batch_download.py     # Bulk OHLCV downloader
    ├── batch_research.py     # Run research_agent across many coins
    └── universal_researcher.py # Experimental cross-sectional (single global) model
```

---

## ⚙️ Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create a .env with your keys (Gemini used by the research agent)
echo "GEMINI_API_KEY=your_key_here" > .env
```

---

## ▶️ Usage

**1. Download data** (creates `data/<symbol>/` automatically):

```bash
python core/data_downloader.py --symbol ZECUSDT --interval 4h
```

**2. Run the research agent** on a symbol:

```bash
python core/research_agent.py --symbol ZECUSDT
```

This runs the full loop: features → Dynamic Triple-Barrier labels → XGBoost
Long/Short training (TimeSeriesSplit) → parameter sweep → realistic backtest →
report + Plotly visualizations under `data/<symbol>/`.

**Optional — batch across many coins:**

```bash
python core/trend_scanner.py            # scan for universe candidates
python core/batch_download.py --priority
python core/batch_research.py --priority
```

---

## 📐 How the code implements the `AGENTS.md` rules

- **Rule #1 — N-1 shift:** `features.py` shifts every engineered feature by
  `.shift(1)`; only the current bar's `Open` (the execution price) stays
  unshifted, and it is where each labeled trade is entered.
- **Rule #3 — Macro handling:** `data_loader.py` pulls Yahoo + FRED, forward-fills
  only (`.ffill()`), and derives % changes / rolling stats rather than raw levels.
- **Rule #3 — Dynamic Triple-Barrier:** `labels.py` scans **every candle**,
  entering at that bar's `Open` and racing an upper (TP), lower (SL) and vertical
  (time-limit, default **700 bars**) barrier. Long and Short are labeled
  independently with inverted TP/SL. Barriers can be **ATR-scaled** (`use_atr=True`)
  for volatility-adaptive targets.
- **Rule #4 — Feature selection:** windows use **Fibonacci/log spacing**
  (`8, 13, 21, 34, 55, 89, 144` and `20, 50, 100, 200`), and the optimizer drops
  features with correlation **> 0.75**. SHAP plots rank the final drivers.
- **Rule #5 — Class imbalance:** `optimizer.py` sets `scale_pos_weight` from the
  positive/negative ratio for both models.
- **Rule #6/#7 — Exit hierarchy & sizing:** `backtester.py` applies ATR-based
  stop, break-even, and trailing-stop exits with directional Long/Short logic.
- **Rule #8 — CV:** `optimizer.py` uses `TimeSeriesSplit` (never random K-Fold).
- **Rule #10 — Visualization:** `visualizer.py` produces interactive Plotly charts
  and SHAP summaries at each step for human sanity-checking.

> **Outputs** for each symbol (report `summary_report.md`, `shap_long.png`,
> `shap_short.png`, `trade_log.csv`, and interactive charts) are written to
> `data/<symbol>/`.
