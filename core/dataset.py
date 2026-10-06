"""
dataset.py — Pooled, leakage-safe training panel for the Entry model

Builds ONE stacked table across many "ZEC-like" coins so the entry model learns
the setup shape cross-sectionally (one coin ~= a dozen rallies — far too few; a
basket of 100+ gives tens of thousands of positives).

For each coin:
  1. FeatureEngineer  -> ~100 technical features, already strict n-1 (.shift(1)).
  2. EntryLabeler     -> Entry_Label (+100%/30d/-40% path-aware) + uniqueness weights.
  3. Shared macro (publication-lagged) + Fear&Greed + per-coin shifted funding.
  4. Keep ONLY scale-free features (ratios / bounded oscillators / returns) so
     coins at wildly different price & volume scales are directly comparable.
     Absolute-unit columns (SMA_*, ATR_*, MACD in price units, raw OFI/Volume)
     are dropped — a tree can't generalize a $-level across coins.

Output: data/model/entry_panel.parquet  (Date, symbol, <features>, Entry_Label,
        Label_EndDate, Sample_Weight), plus a feature-list .txt.

LEAKAGE (AGENTS.md Rule #1 & #3)
  - Features: shifted in features.py. Funding: shifted here. Macro: publication
    lag + ffill (MacroSnapshot). F&G / macro merged backward-asof (never future).
  - Labels look forward (that is Y, not a feature) — see labeler.py.
"""
import os
import sys
import glob
import argparse
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FeatureEngineer           # noqa: E402
from labeler import EntryLabeler               # noqa: E402
from entry_finder import MacroSnapshot, fetch_fear_greed, fg_at  # noqa: E402


# ----------------------------------------------------------------------------
# Scale-free feature selection (cross-coin comparability)
# ----------------------------------------------------------------------------
# A column is kept only if it is a ratio, a bounded oscillator, or a return —
# i.e. its magnitude does not depend on the coin's absolute price or volume.
SCALE_FREE_PREFIXES = (
    'RSI_',            # 0-100 oscillator (also covers RSI_*_Lag1)
    'ADX_',            # 0-100 trend strength
    'BB_Width_',       # (upper-lower)/sma ratio (also BB_Width_*_Lag1)
    'Dist_SMA_',       # (close-sma)/sma ratio
    'Return_',         # pct_change (also Return_1/2 and _Lag1)
    'Dist_Low_',       # (close-low)/low ratio
    'Dist_High_',      # (close-high)/high ratio
    'Volume_Surge_',   # volume / rolling-avg volume ratio
    'Pos_In_',         # 0-1 position within range
    'Dist_VWAP_',      # (close-vwap)/vwap ratio
)
SCALE_FREE_EXACT = {'SMA_Cross', 'ATR_Ratio'}
# Explicitly DROPPED (absolute units -> not comparable across coins):
#   SMA_*, ATR_* (price), MACD / MACD_Hist / *_Lag1 (price-scale EMA diff),
#   OFI_Proxy (volume-scaled), raw OHLCV/Volume.


def select_scale_free(cols):
    keep = []
    for c in cols:
        if c in SCALE_FREE_EXACT or c.startswith(SCALE_FREE_PREFIXES):
            # MACD_Hist_Lag1 etc. must not sneak in via a prefix — none of the
            # prefixes match 'MACD', so this is safe. Guard anyway:
            if c.startswith('MACD'):
                continue
            keep.append(c)
    return keep


# ----------------------------------------------------------------------------
# Coin discovery
# ----------------------------------------------------------------------------
def discover_coins(base_dir, basket_json=None, symbols=None, interval='4h'):
    """Return list of (symbol, csv_path) for coins that have a `interval` CSV on disk."""
    data_dir = os.path.join(base_dir, 'data')
    suffix = f'_{interval}_full.csv'
    found = {}
    for csv in glob.glob(os.path.join(data_dir, '*', f'*{suffix}')):
        sym = os.path.basename(csv).replace(suffix, '').upper()
        found[sym] = csv

    if symbols:
        want = {s.upper() for s in symbols}
    elif basket_json and os.path.exists(basket_json):
        with open(basket_json) as f:
            want = {s.upper() for s in json.load(f)}
    else:
        want = set(found.keys())

    pairs = [(s, found[s]) for s in sorted(want) if s in found]
    missing = sorted(want - set(found.keys()))
    if missing:
        print(f"[Dataset] {len(missing)} basket coins have no CSV yet (skipped): "
              f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    return pairs


# ----------------------------------------------------------------------------
# Per-coin build
# ----------------------------------------------------------------------------
def build_coin(symbol, csv_path, macro, fg, tp_pct, sl_pct, horizon_days,
               min_rows=1500):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    if len(df) < min_rows:
        print(f"  -> {symbol}: only {len(df)} rows (< {min_rows}) — skipped")
        return None

    med = df['Date'].diff().median()
    bpd = max(1, round(pd.Timedelta(days=1) / med))

    # 1) Labels on the FULL series: forward triple-barrier (needs future highs/
    #    lows; labeler.py), emitting Entry_Label/Label_EndDate/Sample_Weight.
    lab = EntryLabeler(df, tp_pct=tp_pct, sl_pct=sl_pct,
                       horizon_days=horizon_days, bars_per_day=bpd).generate()
    lab = lab[['Date', 'Entry_Label', 'Label_EndDate', 'Sample_Weight']]

    # 2) Features (strict n-1 shift applied inside FeatureEngineer). Time-based
    #    windows scale with the series' bars/day so they span the same real time
    #    on 4H (bpd=6) and 1D (bpd=1).
    feat = FeatureEngineer(df.copy(), bars_per_day=bpd).generate_all_features()
    # Funding is a base column (unshifted) in features.py — shift it here so the
    # decision at Open[T] only sees funding realized up to T-1.
    if 'Funding_Rate' in feat.columns:
        feat['Funding_Rate'] = feat['Funding_Rate'].shift(1)

    feat_cols = select_scale_free(feat.columns)
    base = feat[['Date'] + feat_cols].copy()
    if 'Funding_Rate' in feat.columns:
        base['Funding_Rate'] = feat['Funding_Rate']

    # 3) Merge labels (inner — only bars that have both features and a label).
    m = base.merge(lab, on='Date', how='inner')

    # 4) Shared sentiment / macro (backward-asof — never future).
    if fg is not None and not fg.empty:
        fg_vals = [fg_at(fg, d).get('FearGreed') for d in m['Date']]
        m['FearGreed'] = fg_vals
    if macro is not None and macro.table is not None and not macro.table.empty:
        mt = macro.table.sort_index()
        mt_reset = mt.reset_index().rename(columns={'index': 'Date'})
        mt_reset['Date'] = pd.to_datetime(mt_reset['Date'])
        m = pd.merge_asof(m.sort_values('Date'), mt_reset.sort_values('Date'),
                          on='Date', direction='backward')

    m['symbol'] = symbol
    m = m.dropna(subset=feat_cols)  # require the core TA features present
    pos = int(m['Entry_Label'].sum())
    print(f"  -> {symbol}: {len(m)} rows, {pos} pos ({pos/max(1,len(m))*100:.1f}%), "
          f"{len(feat_cols)} TA feats")
    return m


def main():
    ap = argparse.ArgumentParser(description="Build pooled entry-model panel")
    ap.add_argument('--basket', default='data/scan_results/scanned_priority_coins.json',
                    help='JSON list of symbols to pool (default: trend_scanner output)')
    ap.add_argument('--symbols', nargs='*', default=None,
                    help='Explicit symbol list (overrides --basket)')
    ap.add_argument('--tp', type=float, default=1.00, help='TP fraction (default 1.00=+100%%)')
    ap.add_argument('--sl', type=float, default=0.40, help='SL fraction (default 0.40=-40%%)')
    ap.add_argument('--horizon-days', type=int, default=30,
                    help='forward horizon (days) to reach the TP before the SL')
    ap.add_argument('--interval', default='4h',
                    help="Timeframe suffix of the source CSVs (e.g. 4h, 1d). Default 4h.")
    ap.add_argument('--min-rows', type=int, default=1500,
                    help="Skip coins with fewer than this many bars. Default 1500 "
                         "(~250 days on 4H); lower it for daily data (e.g. 400).")
    ap.add_argument('--no-macro', action='store_true')
    ap.add_argument('--out', default='data/model/entry_panel.parquet')
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    basket_json = os.path.join(base_dir, args.basket) if not os.path.isabs(args.basket) else args.basket
    out_path = os.path.join(base_dir, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pairs = discover_coins(base_dir, basket_json, args.symbols, interval=args.interval)
    label_desc = f"ENTRY +{args.tp*100:.0f}%/{args.horizon_days}d/-{args.sl*100:.0f}%"
    print(f"[Dataset] Building panel from {len(pairs)} coins "
          f"(interval={args.interval}, label {label_desc})")
    if not pairs:
        print("No coins with CSVs found. Run trend_scanner + batch_download first.")
        return

    # Shared macro + F&G over the FULL date range across all coins.
    gmin, gmax = None, None
    for _, csv in pairs:
        d = pd.read_csv(csv, usecols=['Date'])
        d = pd.to_datetime(d['Date'])
        gmin = d.min() if gmin is None else min(gmin, d.min())
        gmax = d.max() if gmax is None else max(gmax, d.max())
    macro = None
    if not args.no_macro:
        try:
            macro = MacroSnapshot(gmin, gmax).build()
        except Exception as e:
            print(f"[Dataset] Macro build failed ({e}) — continuing without macro.")
    fg = fetch_fear_greed()

    frames = []
    for k, (sym, csv) in enumerate(pairs, 1):
        print(f"[{k}/{len(pairs)}] {sym}")
        try:
            m = build_coin(sym, csv, macro, fg, args.tp, args.sl, args.horizon_days,
                           min_rows=args.min_rows)
            if m is not None:
                frames.append(m)
        except Exception as e:
            print(f"  -> {sym}: FAILED ({e})")

    if not frames:
        print("No usable coins. Aborting.")
        return

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(['Date', 'symbol']).reset_index(drop=True)

    # Persist the feature list (everything except bookkeeping/label columns).
    non_feat = {'Date', 'symbol', 'Entry_Label', 'Label_EndDate', 'Sample_Weight'}
    feature_list = [c for c in panel.columns if c not in non_feat]

    try:
        panel.to_parquet(out_path, index=False)
    except Exception as e:
        out_path = out_path.replace('.parquet', '.csv')
        panel.to_csv(out_path, index=False)
        print(f"[Dataset] parquet unavailable ({e}); wrote CSV instead.")
    feat_txt = os.path.join(os.path.dirname(out_path), 'entry_features.txt')
    with open(feat_txt, 'w') as f:
        f.write('\n'.join(feature_list))

    pos = int(panel['Entry_Label'].sum())
    print("\n" + "=" * 70)
    print(f"  PANEL: {len(panel)} rows | {panel['symbol'].nunique()} coins | "
          f"{len(feature_list)} features")
    print(f"  Positives: {pos} ({pos/len(panel)*100:.2f}%)  "
          f"scale_pos_weight ~= {(len(panel)-pos)/max(1,pos):.1f}")
    print(f"  Date range: {panel['Date'].min()} ~ {panel['Date'].max()}")
    print(f"  Saved: {out_path}")
    print(f"  Features: {feat_txt}")
    print("=" * 70)
    print("\nPer-coin positive counts:")
    print(panel.groupby('symbol')['Entry_Label'].agg(['size', 'sum']).to_string())


if __name__ == '__main__':
    main()
