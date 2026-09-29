"""
entry_finder.py — ZEC "Best Entry Point" finder (replaces Triple-Barrier labeling)

WHAT THIS DOES
--------------
Instead of labeling every candle with a Triple-Barrier target, this module hunts
for the *entry points* of big rallies on ZEC and, for each one, produces a
human-verifiable dossier:

  1. ALL entry points that launched a rally >= min-gain within 14-120 days.
  2. A before/after interactive candlestick chart per entry (context window
     BEFORE the entry, and the rally that unfolds AFTER it).
  3. The market STATE as known just BEFORE the entry (strict n-1, AGENTS.md
     Rule #1):
        - Technical indicators (RSI/MACD/ADX/ATR/returns/… from features.py)
        - Macroeconomic data (Yahoo globals + FRED, with FRED publication lag
          and stationary transforms — AGENTS.md Rule #3)
        - Sentiment: Crypto Fear & Greed Index + funding rate
        - News headlines around the entry date (CryptoPanic API)

RALLY LOGIC (from the user's spec)
----------------------------------
Capture every rally that runs >= min-gain within 14-120 days, whether it is a
V-shape pump or a slow accumulation. The "--slow-only" flag keeps only the
ZEC-like slow-accumulation pattern:

    7d momentum < 50%   (price did NOT explode in the first week)
    gain       >= 300%
    duration   >= 30 days

DETECTION TIMEFRAME
-------------------
Rally detection runs directly on the 4H bars (per the user's choice). Day-based
thresholds are converted to 4H bars using the median bar spacing of the file.

Usage:
    python core/entry_finder.py                 # all rallies >=200% in 14-120d
    python core/entry_finder.py --slow-only      # ZEC-like slow accumulation only
    python core/entry_finder.py --min-gain 300 --before-days 120
    python core/entry_finder.py --no-macro --no-news   # fast, price-only snapshot
"""
import os
import sys
import time
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# Force UTF-8 output (matches the rest of the repo)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FeatureEngineer  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ============================================================================
# 1. RALLY / ENTRY-POINT DETECTION (on 4H bars)
# ============================================================================

def find_entry_points(df, bars_per_day, min_gain=200.0, min_days=14, max_days=200,
                      local_window_days=14):
    """
    Scan the 4H price series for local bottoms that launched a big rally.

    A candle `i` is an entry point if:
      * its Low is the lowest within +/- `local_window_days` (a real local bottom)
      * the max High over the next [min_days, max_days] gains >= `min_gain`%

    Returns a list of entry dicts (before de-duplication).
    """
    if df is None or len(df) < min_days * bars_per_day:
        return []

    highs = df['High'].values
    lows = df['Low'].values
    dates = df['Date'].values
    n = len(df)

    min_bars = int(min_days * bars_per_day)
    max_bars = int(max_days * bars_per_day)
    lw = int(local_window_days * bars_per_day)
    mom_bars = int(7 * bars_per_day)  # 7-day momentum horizon

    entries = []
    for i in range(n - min_bars):
        entry_low = lows[i]
        if entry_low <= 0 or not np.isfinite(entry_low):
            continue

        # Must be a local bottom within +/- local_window_days
        lb = max(0, i - lw)
        lf = min(n, i + lw + 1)
        if entry_low > np.min(lows[lb:lf]):
            continue

        # Look forward for the peak within the [min_bars, max_bars] window
        end = min(i + max_bars + 1, n)
        future_highs = highs[i + 1:end]
        if len(future_highs) < min_bars:
            continue

        peak_offset = int(np.argmax(future_highs))
        peak_price = future_highs[peak_offset]
        peak_idx = i + 1 + peak_offset

        gain = (peak_price - entry_low) / entry_low * 100.0
        days = (pd.Timestamp(dates[peak_idx]) - pd.Timestamp(dates[i])).days

        if gain < min_gain or days < min_days:
            continue

        # 7-day momentum: how far did price run in the FIRST week after entry?
        mom_end = min(i + 1 + mom_bars, n)
        mom_highs = highs[i + 1:mom_end]
        momentum_7d = ((np.max(mom_highs) - entry_low) / entry_low * 100.0
                       if len(mom_highs) > 0 else 0.0)

        entry_ts = pd.Timestamp(dates[i])
        entries.append({
            'entry_idx': i,
            'peak_idx': peak_idx,
            'entry_date': entry_ts,
            'entry_price': float(entry_low),
            'peak_date': pd.Timestamp(dates[peak_idx]),
            'peak_price': float(peak_price),
            'gain_pct': round(gain, 1),
            'days': days,
            'momentum_7d_pct': round(momentum_7d, 1),
            'pattern': 'slow' if momentum_7d < 50 else 'v-shape',
            'year': int(entry_ts.year),
        })

    return _dedupe_entries(entries, bars_per_day)


def _dedupe_entries(entries, bars_per_day, window_days=30):
    """
    Collapse redundant detections of the SAME rally.

    Several local bottoms can climb to the same peak, so their entry→peak
    windows overlap — they are one rally seen from different feet, not distinct
    trades. We sort by gain (strongest first) and keep an entry only if its
    [entry_idx, peak_idx] interval does NOT overlap an already-kept one (and its
    entry is >= `window_days` from any kept entry). The biggest-gain member of
    each overlapping cluster wins.
    """
    if not entries:
        return []
    entries = sorted(entries, key=lambda x: x['gain_pct'], reverse=True)
    kept = []
    for e in entries:
        es, ep = e['entry_idx'], e['peak_idx']
        redundant = False
        for k in kept:
            ks, kp = k['entry_idx'], k['peak_idx']
            # Interval overlap (share any bar between entry and peak)?
            if es <= kp and ks <= ep:
                redundant = True
                break
            # Or entries too close together in calendar time.
            if abs((e['entry_date'] - k['entry_date']).days) < window_days:
                redundant = True
                break
        if not redundant:
            kept.append(e)
    kept.sort(key=lambda x: x['entry_date'])
    return kept


# ============================================================================
# 2. MACRO SNAPSHOT (AGENTS.md Rule #3 — publication lag + stationary transforms)
# ============================================================================

class MacroSnapshot:
    """
    Builds a daily macro table that is safe to sample at any entry date:
      * Yahoo globals (SP500/DXY/GOLD/OIL/…) -> rolling returns / z-scores
        (NEVER raw indices — tree models can't extrapolate).
      * FRED (CPI/M2/…) -> shifted forward by `fred_lag_days` (publication lag),
        forward-filled, then converted to % change.
      * VIX and HY_SPREAD kept raw (naturally mean-reverting — the Rule #3
        exception).
    """

    YF_TICKERS = {
        "SP500": "^GSPC", "DXY": "DX-Y.NYB", "GOLD": "GC=F",
        "OIL": "CL=F", "US10Y": "^TNX", "US2Y": "^IRX", "VIX": "^VIX",
    }
    FRED_TICKERS = {
        "M2": "M2SL", "CPI": "CPIAUCSL", "FEDFUNDS": "FEDFUNDS",
        "HY_SPREAD": "BAMLH0A0HYM2",
    }

    def __init__(self, start, end, fred_lag_days=35):
        self.start = pd.Timestamp(start) - pd.Timedelta(days=400)  # warmup for returns/YoY
        self.end = pd.Timestamp(end) + pd.Timedelta(days=2)
        self.fred_lag_days = fred_lag_days
        self.table = None

    def build(self):
        print("[Macro] Fetching Yahoo globals + FRED (with publication lag)...")
        # Normalize to midnight so daily stamps match Yahoo/FRED (which are
        # midnight-indexed); otherwise a 04:00 range never aligns on reindex.
        idx = pd.date_range(self.start.normalize(), self.end.normalize(), freq='D')
        out = pd.DataFrame(index=idx)
        out.index.name = 'Date'

        # ---- Yahoo globals ----
        try:
            import yfinance as yf
            for name, tk in self.YF_TICKERS.items():
                try:
                    d = yf.download(tk, start=self.start, end=self.end, progress=False)
                    if d is not None and not d.empty:
                        s = d['Close'].squeeze()
                        # yfinance may return a tz-aware index; strip tz so it
                        # aligns with the tz-naive daily range (else all-NaN).
                        si = pd.to_datetime(s.index)
                        s.index = si.tz_localize(None) if si.tz is not None else si
                        out[name] = s.reindex(idx).ffill()
                except Exception as e:
                    print(f"  -> {name}: FAILED ({e})")
        except Exception as e:
            print(f"  -> yfinance unavailable ({e})")

        # ---- FRED (publication lag FIRST, then ffill) ----
        try:
            from pandas_datareader import data as web
            fred = web.DataReader(list(self.FRED_TICKERS.values()), "fred",
                                  self.start, self.end)
            fred.rename(columns={v: k for k, v in self.FRED_TICKERS.items()}, inplace=True)
            # CRITICAL: shift timestamps forward to the real release date
            fred.index = pd.to_datetime(fred.index) + pd.Timedelta(days=self.fred_lag_days)
            fred = fred.reindex(idx.union(fred.index)).sort_index().ffill().reindex(idx)
            for c in fred.columns:
                out[c] = fred[c]
        except Exception as e:
            print(f"  -> FRED: FAILED ({e})")

        out = out.ffill()

        # ---- Stationary transforms (Rule #3) ----
        derived = pd.DataFrame(index=out.index)
        for name in ["SP500", "DXY", "GOLD", "OIL"]:
            if name in out:
                derived[f"{name}_ret20d"] = out[name].pct_change(20)
                derived[f"{name}_ret50d"] = out[name].pct_change(50)
        # Yield-curve spread (engineered, not raw levels)
        if "US10Y" in out and "US2Y" in out:
            derived["YieldCurve_10Y_2Y"] = out["US10Y"] - out["US2Y"]
        # FRED -> % change YoY / MoM
        for name in ["M2", "CPI"]:
            if name in out:
                derived[f"{name}_YoY"] = out[name].pct_change(365) * 100
                derived[f"{name}_MoM"] = out[name].pct_change(30) * 100
        if "FEDFUNDS" in out:
            derived["FEDFUNDS"] = out["FEDFUNDS"]  # already a rate (%), fine as-is
        # Raw mean-reverting exceptions
        for name in ["VIX", "HY_SPREAD"]:
            if name in out:
                derived[name] = out[name]

        self.table = derived.replace([np.inf, -np.inf], np.nan).ffill()
        print(f"  -> Macro table ready: {self.table.shape[1]} features")
        return self

    def at(self, when):
        """Return the macro row as known on `when` (strict backward asof)."""
        if self.table is None or self.table.empty:
            return {}
        when = pd.Timestamp(when).normalize()
        sub = self.table[self.table.index <= when]
        if sub.empty:
            return {}
        row = sub.iloc[-1]
        return {k: (round(float(v), 4) if pd.notna(v) else None) for k, v in row.items()}


# ============================================================================
# 3. SENTIMENT: Crypto Fear & Greed Index
# ============================================================================

def fetch_fear_greed():
    """Full Fear & Greed history as a date-indexed Series (or None)."""
    try:
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        data = requests.get(url, timeout=20).json().get("data", [])
        if not data:
            return None
        rows = [(pd.to_datetime(int(d["timestamp"]), unit="s").normalize(),
                 int(d["value"]), d.get("value_classification", ""))
                for d in data]
        fg = pd.DataFrame(rows, columns=["Date", "FearGreed", "FG_Label"])
        fg = fg.drop_duplicates("Date").set_index("Date").sort_index()
        print(f"[Sentiment] Fear & Greed: {len(fg)} days")
        return fg
    except Exception as e:
        print(f"[Sentiment] Fear & Greed FAILED ({e})")
        return None


def fg_at(fg, when):
    if fg is None or fg.empty:
        return {}
    when = pd.Timestamp(when).normalize()
    sub = fg[fg.index <= when]
    if sub.empty:
        return {}
    r = sub.iloc[-1]
    return {"FearGreed": int(r["FearGreed"]), "FG_Label": r["FG_Label"]}


# ============================================================================
# 4. NEWS: CryptoPanic API
# ============================================================================

class NewsFetcher:
    """
    Pull ZEC-tagged posts from CryptoPanic and match them to a date window.

    NOTE: CryptoPanic's free tier serves mostly RECENT posts; historical
    coverage (especially pre-2021) is sparse. When no headline is found for an
    entry, the dossier falls back to the real Fear & Greed / funding sentiment.
    """

    BASE = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, currency="ZEC", max_pages=15):
        self.currency = currency
        self.max_pages = max_pages
        self.token = (os.getenv("CRYPTOPANIC_TOKEN")
                      or os.getenv("CRYPTOPANIC_API_KEY"))
        self.posts = []  # list of (timestamp, title, url, source)
        self.enabled = bool(self.token)

    def load(self):
        if not self.enabled:
            print("[News] CRYPTOPANIC_TOKEN not set — skipping headlines "
                  "(using Fear & Greed / funding as sentiment fallback).")
            return self
        print(f"[News] Fetching CryptoPanic posts for {self.currency} ...")
        url = f"{self.BASE}?auth_token={self.token}&currencies={self.currency}&public=true"
        pages = 0
        while url and pages < self.max_pages:
            try:
                resp = requests.get(url, timeout=20)
                if resp.status_code != 200:
                    print(f"  -> HTTP {resp.status_code}; stopping.")
                    break
                js = resp.json()
                for p in js.get("results", []):
                    ts = pd.to_datetime(p.get("published_at"))
                    src = (p.get("source") or {}).get("title", "")
                    self.posts.append((ts, p.get("title", ""), p.get("url", ""), src))
                url = js.get("next")
                pages += 1
                time.sleep(0.3)
            except Exception as e:
                print(f"  -> FAILED ({e})")
                break
        # Normalize tz so comparisons work
        self.posts = [(pd.Timestamp(ts).tz_localize(None) if ts is not None and ts.tzinfo
                       else ts, t, u, s) for ts, t, u, s in self.posts if ts is not None]
        print(f"  -> {len(self.posts)} posts cached "
              f"(range {min([p[0] for p in self.posts]) if self.posts else 'n/a'} ~ "
              f"{max([p[0] for p in self.posts]) if self.posts else 'n/a'})")
        return self

    def around(self, when, days_before=7, days_after=1, limit=5):
        """Headlines in [when - days_before, when + days_after]."""
        if not self.posts:
            return []
        when = pd.Timestamp(when)
        lo, hi = when - pd.Timedelta(days=days_before), when + pd.Timedelta(days=days_after)
        hits = [p for p in self.posts if lo <= p[0] <= hi]
        hits.sort(key=lambda x: x[0], reverse=True)
        return [{"date": p[0].strftime("%Y-%m-%d"), "title": p[1],
                 "url": p[2], "source": p[3]} for p in hits[:limit]]


# ============================================================================
# 5. TECHNICAL SNAPSHOT (strict n-1 via features.py shift(1))
# ============================================================================

# Curated, human-readable subset of the ~100 engineered features.
SNAPSHOT_FEATURES = [
    'RSI_20', 'RSI_50', 'MACD', 'MACD_Hist',
    'ADX_20', 'ADX_50', 'SMA_Cross',
    'ATR_Ratio', 'BB_Width_20',
    'Return_42b',    # ~7-day return
    'Return_180b',   # ~1-month return
    'Dist_High_180b', 'Dist_Low_180b',
    'Volume_Surge_20', 'Pos_In_180b',
    'Dist_VWAP_180b', 'OFI_Proxy',
]


def build_feature_table(df):
    """Run the repo's FeatureEngineer (which applies the strict .shift(1))."""
    print("[TA] Engineering technical features (with strict n-1 shift)...")
    fe = FeatureEngineer(df.copy())
    feat = fe.generate_all_features()
    return feat.set_index('Date')


def ta_at(feat_table, when):
    """Technical snapshot as known BEFORE entry (features are already shifted)."""
    if feat_table is None or feat_table.empty:
        return {}
    when = pd.Timestamp(when)
    sub = feat_table[feat_table.index <= when]
    if sub.empty:
        return {}
    row = sub.iloc[-1]
    out = {}
    for f in SNAPSHOT_FEATURES:
        if f in row.index and pd.notna(row[f]):
            out[f] = round(float(row[f]), 4)
    if 'Funding_Rate' in row.index and pd.notna(row['Funding_Rate']):
        out['Funding_Rate'] = round(float(row['Funding_Rate']), 6)
    return out


# ============================================================================
# 6. PER-ENTRY CHART (before + after)
# ============================================================================

def plot_entry(df, entry, bars_per_day, out_path, before_days=90, after_buffer=0.25):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    i, peak_idx = entry['entry_idx'], entry['peak_idx']
    before_bars = int(before_days * bars_per_day)
    rally_bars = peak_idx - i
    after_bars = int(rally_bars * (1 + after_buffer))
    lo = max(0, i - before_bars)
    hi = min(len(df), i + after_bars + 1)
    win = df.iloc[lo:hi].copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(
            f"{entry['entry_date'].date()} → {entry['peak_date'].date()}  "
            f"(+{entry['gain_pct']}% in {entry['days']}d, {entry['pattern']}, "
            f"7d-mom {entry['momentum_7d_pct']}%)",
            "Volume"),
    )

    fig.add_trace(go.Candlestick(
        x=win['Date'], open=win['Open'], high=win['High'],
        low=win['Low'], close=win['Close'], name='Price'), row=1, col=1)

    # SMA context overlays
    for w, color in [(20, 'orange'), (50, 'deepskyblue')]:
        fig.add_trace(go.Scatter(
            x=win['Date'], y=win['Close'].rolling(w).mean(),
            mode='lines', line=dict(color=color, width=1),
            name=f'SMA{w}'), row=1, col=1)

    # Shade the "after entry" rally region
    fig.add_vrect(x0=entry['entry_date'], x1=win['Date'].iloc[-1],
                  fillcolor='lime', opacity=0.06, line_width=0, row=1, col=1)

    # Entry & peak markers
    fig.add_trace(go.Scatter(
        x=[entry['entry_date']], y=[entry['entry_price']], mode='markers+text',
        marker=dict(symbol='triangle-up', size=16, color='lime'),
        text=['ENTRY'], textposition='bottom center',
        name='Entry'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[entry['peak_date']], y=[entry['peak_price']], mode='markers+text',
        marker=dict(symbol='star', size=16, color='gold'),
        text=['PEAK'], textposition='top center',
        name='Peak'), row=1, col=1)
    fig.add_vline(x=entry['entry_date'], line=dict(color='white', width=1, dash='dash'),
                  row=1, col=1)

    fig.add_trace(go.Bar(x=win['Date'], y=win['Volume'], name='Volume',
                         marker_color='gray'), row=2, col=1)

    fig.update_layout(
        template='plotly_dark', height=760, xaxis_rangeslider_visible=False,
        title=f"ZEC entry — {entry['entry_date'].date()}  (shaded = post-entry rally)",
        legend=dict(orientation='h', y=1.06, x=1, xanchor='right'))
    fig.write_html(out_path)
    return out_path


# ============================================================================
# 7. COMBINED REPORT
# ============================================================================

def _fmt_kv(d):
    if not d:
        return "<span class='muted'>—</span>"
    return "".join(
        f"<div class='kv'><span class='k'>{k}</span>"
        f"<span class='v'>{v}</span></div>" for k, v in d.items())


def _fmt_news(news):
    if not news:
        return ("<span class='muted'>No headlines found for this window "
                "(sparse historical coverage / no token).</span>")
    items = "".join(
        f"<li><span class='ndate'>{n['date']}</span> "
        f"<a href='{n['url']}' target='_blank'>{n['title']}</a>"
        f" <span class='muted'>· {n['source']}</span></li>" for n in news)
    return f"<ul class='news'>{items}</ul>"


def build_report(entries, chart_files, out_path, symbol, slow_only):
    cards = []
    for e, chart in zip(entries, chart_files):
        rel = os.path.basename(chart)
        cards.append(f"""
        <section class="card">
          <h2>{e['entry_date'].date()} → {e['peak_date'].date()}
            <span class="badge {e['pattern']}">{e['pattern']}</span></h2>
          <div class="stats">
            <div class="stat"><span class="big">+{e['gain_pct']}%</span><span>gain</span></div>
            <div class="stat"><span class="big">{e['days']}d</span><span>duration</span></div>
            <div class="stat"><span class="big">{e['momentum_7d_pct']}%</span><span>7d momentum</span></div>
            <div class="stat"><span class="big">${e['entry_price']:.4g}</span><span>entry</span></div>
            <div class="stat"><span class="big">${e['peak_price']:.4g}</span><span>peak</span></div>
          </div>
          <iframe src="{rel}" loading="lazy"></iframe>
          <div class="cols">
            <div class="col"><h3>📉 Technical (n-1)</h3>{_fmt_kv(e['ta'])}</div>
            <div class="col"><h3>🌍 Macro (n-1)</h3>{_fmt_kv(e['macro'])}</div>
            <div class="col"><h3>😱 Sentiment</h3>{_fmt_kv(e['sentiment'])}</div>
          </div>
          <div class="newswrap"><h3>📰 News around entry</h3>{_fmt_news(e['news'])}</div>
        </section>""")

    mode = "Slow-accumulation (ZEC-like) only" if slow_only else "All rally patterns"
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{symbol} Entry Points</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#0e1117; color:#e6e6e6; font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:24px; }}
  h1 {{ margin:0 0 4px; }}
  .sub {{ color:#8b949e; margin-bottom:24px; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:20px; margin-bottom:28px; }}
  .card h2 {{ margin:0 0 12px; font-size:1.25rem; }}
  .badge {{ font-size:.7rem; padding:2px 8px; border-radius:10px; vertical-align:middle; }}
  .badge.slow {{ background:#1f6feb33; color:#79c0ff; }}
  .badge.v-shape {{ background:#f8514933; color:#ff7b72; }}
  .stats {{ display:flex; gap:24px; flex-wrap:wrap; margin-bottom:14px; }}
  .stat {{ display:flex; flex-direction:column; }}
  .stat .big {{ font-size:1.3rem; font-weight:700; color:#58a6ff; }}
  .stat span:last-child {{ font-size:.72rem; color:#8b949e; text-transform:uppercase; }}
  iframe {{ width:100%; height:780px; border:0; border-radius:8px; background:#0e1117; }}
  .cols {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin-top:16px; }}
  .col h3, .newswrap h3 {{ font-size:.85rem; color:#8b949e; text-transform:uppercase; margin:0 0 8px; }}
  .kv {{ display:flex; justify-content:space-between; font-size:.82rem; padding:2px 0; border-bottom:1px solid #21262d; }}
  .kv .k {{ color:#8b949e; }} .kv .v {{ color:#e6e6e6; font-variant-numeric:tabular-nums; }}
  .muted {{ color:#6e7681; font-style:italic; }}
  .news {{ list-style:none; padding:0; margin:0; }}
  .news li {{ padding:4px 0; font-size:.85rem; border-bottom:1px solid #21262d; }}
  .news a {{ color:#79c0ff; text-decoration:none; }}
  .ndate {{ color:#8b949e; font-variant-numeric:tabular-nums; margin-right:6px; }}
  .newswrap {{ margin-top:16px; }}
</style></head><body>
  <h1>{symbol} — Best Entry Points</h1>
  <div class="sub">{mode} · {len(entries)} entries · generated {datetime.now():%Y-%m-%d %H:%M}
    · features shown are strict n-1 (known before the entry open)</div>
  {''.join(cards)}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="ZEC best-entry-point finder")
    ap.add_argument('--csv', default='data/zecusdt/ZECUSDT_4h_full.csv')
    ap.add_argument('--symbol', default='ZECUSDT')
    ap.add_argument('--min-gain', type=float, default=200.0)
    ap.add_argument('--min-days', type=int, default=14)
    ap.add_argument('--max-days', type=int, default=200)
    ap.add_argument('--before-days', type=int, default=90,
                    help='Context window (days) shown before each entry')
    ap.add_argument('--slow-only', action='store_true',
                    help='Keep only ZEC-like slow accumulation (7d-mom<50%%, gain>=300%%, days>=30)')
    ap.add_argument('--no-macro', action='store_true')
    ap.add_argument('--no-news', action='store_true')
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, args.csv) if not os.path.isabs(args.csv) else args.csv
    out_dir = os.path.join(os.path.dirname(csv_path), 'entry_points')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print(f"  ENTRY FINDER  |  {args.symbol}  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 78)

    # ---- Load 4H data ----
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    med = df['Date'].diff().median()
    bars_per_day = max(1, round(pd.Timedelta(days=1) / med))
    print(f"[Data] {len(df)} bars | {df['Date'].min()} ~ {df['Date'].max()} "
          f"| ~{bars_per_day} bars/day")

    # ---- Detect entries ----
    entries = find_entry_points(df, bars_per_day, args.min_gain,
                                args.min_days, args.max_days)
    if args.slow_only:
        entries = [e for e in entries if e['momentum_7d_pct'] < 50
                   and e['gain_pct'] >= 300 and e['days'] >= 30]
    print(f"[Scan] {len(entries)} entry point(s) "
          f"({'slow-only' if args.slow_only else 'all patterns'})")
    if not entries:
        print("No entries matched. Try lowering --min-gain or dropping --slow-only.")
        return

    # ---- Snapshot sources ----
    feat_table = build_feature_table(df)
    macro = None
    if not args.no_macro:
        macro = MacroSnapshot(df['Date'].min(), df['Date'].max()).build()
    fg = fetch_fear_greed()
    news_fetcher = None
    if not args.no_news:
        base_ccy = args.symbol.replace('USDT', '').replace('USD', '')
        news_fetcher = NewsFetcher(currency=base_ccy).load()

    # ---- Per-entry dossiers ----
    chart_files = []
    for k, e in enumerate(entries, 1):
        e['ta'] = ta_at(feat_table, e['entry_date'])
        e['macro'] = macro.at(e['entry_date']) if macro else {}
        e['sentiment'] = fg_at(fg, e['entry_date'])
        if 'Funding_Rate' in e['ta']:
            e['sentiment']['Funding_Rate'] = e['ta'].pop('Funding_Rate')
        e['news'] = (news_fetcher.around(e['entry_date'], args.before_days // 3)
                     if news_fetcher else [])
        cf = os.path.join(out_dir, f"entry_{e['entry_date']:%Y%m%d}.html")
        plot_entry(df, e, bars_per_day, cf, before_days=args.before_days)
        chart_files.append(cf)
        print(f"  [{k}/{len(entries)}] {e['entry_date'].date()} "
              f"+{e['gain_pct']}% / {e['days']}d ({e['pattern']}) -> {os.path.basename(cf)}")

    # ---- CSV ----
    csv_cols = ['entry_date', 'entry_price', 'peak_date', 'peak_price',
                'gain_pct', 'days', 'momentum_7d_pct', 'pattern', 'year']
    dfe = pd.DataFrame([{c: e[c] for c in csv_cols} for e in entries])
    csv_out = os.path.join(out_dir, 'entries.csv')
    dfe.to_csv(csv_out, index=False, encoding='utf-8-sig')

    # ---- Report ----
    report = build_report(entries, chart_files,
                          os.path.join(out_dir, 'report.html'),
                          args.symbol, args.slow_only)

    print("\n" + "=" * 78)
    print(f"  DONE — {len(entries)} entries")
    print(f"  Table:  {csv_out}")
    print(f"  Report: {report}")
    print(f"  Charts: {out_dir}/entry_*.html")
    print("=" * 78)
    print(dfe.to_string(index=False))


if __name__ == '__main__':
    main()
