"""
rotation_paper.py — Forward (paper) trading of the rotation strategy. Run ONCE A DAY after 00:00 UTC.

Each run:
  1. (--update) appends the newest 4H bars for every coin in the universe from Binance
     (spot for root=data, USDT-M perps for root=data_all).
  2. Scores the buy model on the new bars with the FINAL model (data/model/.../entry_xgb.json). Days
     covered by the OOF file keep their OOF prob, so the strategy's history/state matches the backtest.
  3. Builds the book with rotation_backtest.build_weights (same code as the backtest) and takes the
     weights decided at the close of the last COMPLETE UTC day = the target for today's Open.
  4. Saves that target to <paper-dir>/targets/<trade_day>.csv and appends it to the ledger.
     The paper P&L is computed ONLY from targets saved by earlier runs (never back-filled), using
     the realised daily Opens — this is the genuinely out-of-sample record.

Nothing is sent to an exchange; this is a recorder.

Usage:
    python core/rotation_paper.py --update                       # preset b7-ens on the research basket
    python core/rotation_paper.py --update --preset b7
    python core/rotation_paper.py --update --root data_all --preset btc-trend   # best honest result (README)
    python core/rotation_paper.py --report                       # only print the paper record
"""
import os
import sys
import json
import time
import glob
import argparse

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rotation_backtest as rb                                   # noqa: E402
from features import FeatureEngineer                             # noqa: E402
from dataset import select_scale_free                            # noqa: E402
from entry_finder import MacroSnapshot, fetch_fear_greed, fg_at  # noqa: E402

COST = 0.0015


def update_csv(path, symbol, futures):
    """Append the newest closed 4H bars (and funding for perps) to an existing CSV."""
    df = pd.read_csv(path, parse_dates=['Date'])
    start = int(df['Date'].max().timestamp() * 1000) + 1
    url = ('https://fapi.binance.com/fapi/v1/klines' if futures else 'https://api.binance.com/api/v3/klines')
    rows = []
    while True:
        r = requests.get(url, params={'symbol': symbol, 'interval': '4h', 'startTime': start, 'limit': 1000}, timeout=30)
        if r.status_code != 200:
            break
        k = r.json()
        if not k:
            break
        rows += k
        start = k[-1][6] + 1
        if len(k) < 1000:
            break
        time.sleep(0.1)
    now_ms = int(time.time() * 1000)
    rows = [k for k in rows if k[6] < now_ms]                    # closed bars only
    if not rows:
        return 0
    new = pd.DataFrame({'Date': pd.to_datetime([k[0] for k in rows], unit='ms'),
                        'Open': [float(k[1]) for k in rows], 'High': [float(k[2]) for k in rows],
                        'Low': [float(k[3]) for k in rows], 'Close': [float(k[4]) for k in rows],
                        'Volume': [float(k[5]) for k in rows]})
    if 'Quote_Volume' in df.columns:
        new['Quote_Volume'] = [float(k[7]) for k in rows]
    if 'Funding_Rate' in df.columns:
        fr = 0.0
        if futures:
            f = requests.get('https://fapi.binance.com/fapi/v1/fundingRate',
                             params={'symbol': symbol, 'startTime': int(df['Date'].max().timestamp() * 1000), 'limit': 1000},
                             timeout=30).json()
            if isinstance(f, list) and f:
                fd = pd.DataFrame({'Date': pd.to_datetime([x['fundingTime'] for x in f], unit='ms'),
                                   'Funding_Rate': [float(x['fundingRate']) for x in f]}).sort_values('Date')
                new = pd.merge_asof(new.sort_values('Date'), fd, on='Date', direction='backward')
        if 'Funding_Rate' not in new.columns:
            new['Funding_Rate'] = df['Funding_Rate'].iloc[-1] if futures else fr
        new['Funding_Rate'] = new['Funding_Rate'].fillna(df['Funding_Rate'].iloc[-1])
    out = pd.concat([df, new], ignore_index=True).drop_duplicates('Date', keep='last').sort_values('Date')
    out.to_csv(path, index=False)
    return len(new)


def live_probs(base, root, syms, model_dir, since):
    """Final-model prob for every 4H row on/after `since` (features strictly N-1 as in dataset.py)."""
    from xgboost import XGBClassifier
    meta = json.load(open(os.path.join(model_dir, 'entry_model_meta.json')))
    feats = meta['features']
    model = XGBClassifier()
    model.load_model(os.path.join(model_dir, 'entry_xgb.json'))
    macro = MacroSnapshot(pd.Timestamp('2018-01-01'), pd.Timestamp.now()).build()
    mt = macro.table.sort_index().reset_index().rename(columns={'index': 'Date'}) if macro.table is not None else None
    fg = fetch_fear_greed()
    out = []
    for s in syms:
        p = os.path.join(base, root, s.lower(), f'{s}_4h_full.csv')
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
        if df['Date'].max() < since:
            continue
        feat = FeatureEngineer(df.copy(), bars_per_day=6).generate_all_features()
        if 'Funding_Rate' in feat.columns:
            feat['Funding_Rate'] = feat['Funding_Rate'].shift(1)
        m = feat[['Date'] + select_scale_free(feat.columns) + (['Funding_Rate'] if 'Funding_Rate' in feat else [])].copy()
        m = m[m['Date'] >= since - pd.Timedelta(days=1)]
        if m.empty:
            continue
        if fg is not None and not fg.empty:
            m['FearGreed'] = [fg_at(fg, d).get('FearGreed') for d in m['Date']]
        if mt is not None:
            m = pd.merge_asof(m.sort_values('Date'), mt.assign(Date=pd.to_datetime(mt['Date'])).sort_values('Date'),
                              on='Date', direction='backward')
        for c in feats:
            if c not in m.columns:
                m[c] = 0.0
        m['oof_prob_xgb'] = model.predict_proba(m[feats].astype(float).fillna(0.0))[:, 1]
        m['symbol'] = s
        out.append(m[['Date', 'symbol', 'oof_prob_xgb']])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['Date', 'symbol', 'oof_prob_xgb'])


def paper_record(paper_dir, O):
    """Realised paper equity from saved targets only (target for day t is held open t -> open t+1)."""
    files = sorted(glob.glob(os.path.join(paper_dir, 'targets', '*.csv')))
    if not files:
        return None
    T = {}
    for f in files:
        t = pd.read_csv(f)
        T[pd.Timestamp(os.path.basename(f)[:-4])] = t.set_index('symbol')['weight']
    W = pd.DataFrame(T).T.fillna(0.0).sort_index()
    days = [d for d in W.index if d + pd.Timedelta(days=1) in O.index and d in O.index]
    if not days:
        return W, None
    W2 = W.loc[days]
    Oc = O.reindex(columns=W2.columns)
    r = (Oc.shift(-1) / Oc - 1).reindex(days).fillna(0.0)
    prev = W2.shift(1).fillna(0.0)
    turnover = (W2 - prev).abs().sum(axis=1)
    pr = (W2 * r).sum(axis=1) - COST * turnover
    return W, (1 + pr).cumprod()


def main():
    ap = argparse.ArgumentParser(description='Daily paper trading of the rotation strategy')
    ap.add_argument('--preset', default='b7-ens',
                    choices=['b6', 'b7', 'b7-ens', 'btc-trend', 'greedy', 'greedy-alts'])
    ap.add_argument('--root', default='data')
    ap.add_argument('--oof', default='data/model/v2/m1_100pct_60d/entry_oof.parquet')
    ap.add_argument('--model-dir', default='data/model/v2/m1_100pct_60d')
    ap.add_argument('--paper-dir', default='data/paper')
    ap.add_argument('--universe-top', type=int, default=0,
                    help="point-in-time top-N by volume (root=data_all); 0 = fixed universe from the OOF file")
    ap.add_argument('--update', action='store_true', help='download the newest bars first')
    ap.add_argument('--report', action='store_true', help='only print the paper record')
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    P = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    paper_dir = P(args.paper_dir)
    os.makedirs(os.path.join(paper_dir, 'targets'), exist_ok=True)
    rargs = rb.apply_preset(rb.get_parser().parse_args(['--preset', args.preset, '--oof', args.oof, '--root', args.root,
                                                         '--universe-top', str(args.universe_top)]))

    oof = pd.read_parquet(P(args.oof), columns=['Date', 'symbol', 'oof_prob_xgb']).dropna()
    oof['Date'] = pd.to_datetime(oof['Date'])
    syms = sorted(oof['symbol'].unique())
    if args.universe_top:            # dynamic universe: every coin with data can enter the top-N later
        syms = sorted(set(syms) | {os.path.basename(d).upper() for d in glob.glob(os.path.join(base, args.root, '*'))
                                   if os.path.isdir(d)} - {'BTCUSDT', 'MODEL', 'PAPER'})

    if args.update and not args.report:
        futures = args.root != 'data'
        n = 0
        for s in syms + ['BTCUSDT']:
            p = os.path.join(base, args.root, s.lower(), f'{s}_4h_full.csv')
            if os.path.exists(p):
                try:
                    n += update_csv(p, s, futures)
                except Exception as e:
                    print(f'  update {s}: {e}')
        print(f'[Update] +{n} new 4H bars across the universe')
    if args.universe_top and not args.report:
        from universe_panel import volume_rank
        rank = volume_rank(base, syms, args.root)
        rank.to_parquet(os.path.join(base, args.root, 'volume_rank.parquet'))
        near = rank.iloc[-1]
        scored = sorted(near[near <= 2 * args.universe_top].index)      # only coins that can matter soon
    else:
        scored = syms

    O, C = rb.load_daily(base, syms + ['BTCUSDT'], root=args.root)
    utc_today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    O, C = O.loc[:utc_today - pd.Timedelta(days=1)], C.loc[:utc_today - pd.Timedelta(days=1)]   # complete days only
    last_full = C['BTCUSDT'].last_valid_index()
    O, C = O.loc[:last_full], C.loc[:last_full]
    today = last_full + pd.Timedelta(days=1)                         # the Open this target is for
    if today != utc_today:
        print(f'[Warn] price data ends {last_full.date()} — target is for {today.date()}, not today '
              f'({utc_today.date()}). Run with --update for a live target.')

    if not args.report:
        oof_end = oof['Date'].max()
        live = live_probs(base, args.root, scored, P(args.model_dir), oof_end + pd.Timedelta(hours=4))
        probs = pd.concat([oof, live[live['Date'] > oof_end]], ignore_index=True)
        probs['day'] = probs['Date'].dt.normalize()
        # one extra (empty) row so the decision taken at the close of last_full lands on `today`
        O2 = pd.concat([O, pd.DataFrame(index=[today], columns=O.columns, dtype=float)])
        C2 = pd.concat([C, pd.DataFrame(index=[today], columns=C.columns, dtype=float)])
        data = {'oof': probs, 'syms': syms, 'O': O2, 'C': C2}
        if rargs.breakout:            # greedy presets: breakout sleeve needs daily quote volume
            V = rb.load_quote_volume(base, syms, args.root).loc[:last_full]
            data['V'] = pd.concat([V, pd.DataFrame(index=[today], columns=V.columns, dtype=float)])
        if rargs.ensemble_near_high:
            import copy
            Ws = []
            for x in (float(v) for v in rargs.ensemble_near_high.split(',')):
                a = copy.copy(rargs)
                a.lead_near_high = x
                Ws.append(rb.build_weights(a, data))
            cols = sorted(set().union(*[w.columns for w in Ws]))
            W = sum(w.reindex(columns=cols, fill_value=0.0) for w in Ws) / len(Ws)
        else:
            W = rb.build_weights(rargs, data)
        tgt = W.loc[today]
        tgt = tgt[tgt > 1e-6].sort_values(ascending=False)
        path = os.path.join(paper_dir, 'targets', f'{today.date()}.csv')
        if os.path.exists(path):
            print(f'[Paper] target for {today.date()} already recorded — not overwritten')
        else:
            pd.DataFrame({'symbol': tgt.index, 'weight': tgt.values}).to_csv(path, index=False)
            with open(os.path.join(paper_dir, 'ledger.csv'), 'a') as f:
                for s_, w_ in tgt.items():
                    f.write(f'{pd.Timestamp.utcnow():%Y-%m-%d %H:%M},{today.date()},{s_},{w_:.4f},{args.preset}\n')
        print(f"\n[Target for open {today.date()} UTC | preset {args.preset} | data through {last_full.date()}]")
        print(f"  cash {1 - tgt.sum():.1%}")
        for s_, w_ in tgt.items():
            print(f'  {s_:16s} {w_:6.1%}')

    W_all, eq = paper_record(paper_dir, O)
    if eq is None or len(eq) == 0:
        print('\n[Paper record] no realised days yet (needs targets saved on earlier days).')
    else:
        dd = (eq / eq.cummax() - 1).min()
        print(f"\n[Paper record] {eq.index[0].date()} -> {eq.index[-1].date()} ({len(eq)} days): "
              f"{(eq.iloc[-1] - 1):+.1%} | max DD {dd:.1%}")
        eq.rename('equity').to_csv(os.path.join(paper_dir, 'paper_equity.csv'), index_label='Date')


if __name__ == '__main__':
    main()
