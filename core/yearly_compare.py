"""yearly_compare.py — Per-year PnL: baseline fixed-TP vs the 3 exit optimizers.

Same ENTRY for every row (m1 OOF signal >= threshold, execute at the next bar's
Open) — only the EXIT differs. Each optimizer uses the best parameters its
(leakage-free, train/validation-split) search found; see optimize_*.py. $100 per
trade, 0.30% round-trip cost. Indicators are ffill-only and every regime/beta read
is taken at the signal candle n-1, per AGENTS.md.

The optimizers hardcode a 0.70 entry filter, so --threshold lets you reproduce the
apples-to-apples 0.70 table or the 0.60 baseline table (optimizer exits reused as-is).

Usage:
    python core/yearly_compare.py --threshold 0.70
    python core/yearly_compare.py --threshold 0.60
"""
import os, glob, argparse, importlib.util
import numpy as np, pandas as pd

FEE, SLIP, NOTIONAL = 0.001, 0.0005, 100.0
COST = 2 * (FEE + SLIP)
BARS_PER_DAY = 6  # 4H bars

# Best params found by each optimizer's honest (no-leak, train-split) search.
BEST = {
    'optimize_exits':     dict(sl=0.1, atr_mult=11.5, horizon_days=90),                 # mode=always_trail
    'optimize_moneyflow': dict(sl=0.6, riskon=15.0, riskoff=4.0, horizon_days=110,
                               cmf_thresh=-0.5, mfi_thresh=10.0),
    'optimize_finance':   dict(sl=0.6, atr_mult=12.0, horizon_days=105,
                               risk_off_only_bluechip=True, high_beta_threshold=8.0),
}


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.70)
    ap.add_argument('--oof', default='data/model/v2/m1_100pct_60d/entry_oof.parquet')
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core = os.path.join(base_dir, 'core')
    oe   = imp('oe',   os.path.join(core, 'optimize_exits.py'))
    omf  = imp('omf',  os.path.join(core, 'optimize_moneyflow.py'))
    ofin = imp('ofin', os.path.join(core, 'optimize_finance.py'))

    oof = pd.read_parquet(os.path.join(base_dir, args.oof))
    oof['Date'] = pd.to_datetime(oof['Date'])
    sig = oof[oof['oof_prob_xgb'] >= args.threshold]
    csvs = {os.path.basename(c).replace('_4h_full.csv', '').upper(): c
            for c in glob.glob(os.path.join(base_dir, 'data', '*', '*_4h_full.csv'))}

    print(f"threshold {args.threshold} | {len(sig):,} signal bars | loading macro (DXY/SP500)...")
    macro = ofin.load_macro_data(oof['Date'].min() - pd.Timedelta(days=365),
                                 oof['Date'].max() + pd.Timedelta(days=30))

    rec = {k: [] for k in ['baseline_fixed_tp100', 'optimize_exits',
                           'optimize_moneyflow', 'optimize_finance']}

    def add(key, dt_entry, net):
        rec[key].append((pd.Timestamp(dt_entry), net * NOTIONAL))

    for sym, grp in sig.groupby('symbol'):
        csv = csvs.get(sym)
        if csv is None:
            continue
        b = pd.read_csv(csv); b['Date'] = pd.to_datetime(b['Date'])
        b = b.sort_values('Date').reset_index(drop=True)

        bm = b.copy()
        bm['Macro_Date'] = bm['Date'].dt.normalize()
        bm = pd.merge(bm, macro[['Macro_Date', 'Econ_Risk_On']], on='Macro_Date', how='left')
        bm['Econ_Risk_On'] = bm['Econ_Risk_On'].ffill().fillna(0)

        d_ex  = oe.calculate_indicators(b.copy(), sma_fast=50, sma_slow=200)
        d_mf  = omf.calculate_indicators(bm.copy())
        d_fin = ofin.calculate_indicators(bm.copy())

        o, hi, lo, cl = b['Open'].values, b['High'].values, b['Low'].values, b['Close'].values
        dt = b['Date'].values
        pos = {d: i for i, d in enumerate(dt)}
        sig_pos = sorted(pos[d] for d in set(grp['Date']) if d in pos)

        atr_ex = d_ex['ATR'].values
        atr_mf, mfi, cmf = d_mf['ATR'].values, d_mf['MFI'].values, d_mf['CMF'].values
        atr_fin, atr_pct = d_fin['ATR'].values, d_fin['ATR_Pct'].values
        econ = bm['Econ_Risk_On'].values

        cd = {k: -1 for k in rec}
        for s in sig_pos:
            entry_i = s + 1
            if entry_i >= len(b):
                continue
            entry_px = o[entry_i]
            if entry_px <= 0 or not np.isfinite(entry_px):
                continue
            de = dt[entry_i]

            # 1) BASELINE fixed TP +100% / SL -40% / 60d
            if s > cd['baseline_fixed_tp100']:
                up, dn = entry_px * 2.0, entry_px * 0.6
                end = min(entry_i + 60 * BARS_PER_DAY, len(b) - 1)
                ei, ex = end, cl[end]
                for j in range(entry_i, end + 1):
                    if lo[j] <= dn: ei, ex = j, dn; break
                    if hi[j] >= up: ei, ex = j, up; break
                add('baseline_fixed_tp100', de, ex / entry_px - 1.0 - COST)
                cd['baseline_fixed_tp100'] = ei

            # 2) optimize_exits: always_trail
            p = BEST['optimize_exits']
            if s > cd['optimize_exits'] and np.isfinite(atr_ex[entry_i]):
                dn = entry_px * (1 - p['sl']); end = min(entry_i + p['horizon_days'] * BARS_PER_DAY, len(b) - 1)
                hs = hi[entry_i]; ei, ex = end, cl[end]
                for j in range(entry_i, end + 1):
                    hs = max(hs, hi[j]); csl = max(dn, hs - atr_ex[j] * p['atr_mult'])
                    if lo[j] <= csl: ei, ex = j, csl; break
                add('optimize_exits', de, ex / entry_px - 1.0 - COST)
                cd['optimize_exits'] = ei

            # 3) optimize_moneyflow: econ-regime trail + CMF/MFI early exit
            p = BEST['optimize_moneyflow']
            if s > cd['optimize_moneyflow'] and np.isfinite(atr_mf[entry_i]):
                dn = entry_px * (1 - p['sl']); end = min(entry_i + p['horizon_days'] * BARS_PER_DAY, len(b) - 1)
                hs = hi[entry_i]; ei, ex = end, cl[end]
                mult = p['riskon'] if econ[entry_i - 1] >= 0 else p['riskoff']   # n-1 regime
                for j in range(entry_i, end + 1):
                    hs = max(hs, hi[j])
                    if cmf[j] < p['cmf_thresh'] or mfi[j] < p['mfi_thresh']: ei, ex = j, cl[j]; break
                    csl = max(dn, hs - atr_mf[j] * mult)
                    if lo[j] <= csl: ei, ex = j, csl; break
                add('optimize_moneyflow', de, ex / entry_px - 1.0 - COST)
                cd['optimize_moneyflow'] = ei

            # 4) optimize_finance: sector rotation filter + trail
            p = BEST['optimize_finance']
            if s > cd['optimize_finance'] and np.isfinite(atr_fin[entry_i]) and np.isfinite(atr_pct[s]):
                is_hb = atr_pct[s] > p['high_beta_threshold']                    # beta at n-1 (=s)
                if not (econ[s] == -1 and p['risk_off_only_bluechip'] and is_hb):  # else filtered out
                    dn = entry_px * (1 - p['sl']); end = min(entry_i + p['horizon_days'] * BARS_PER_DAY, len(b) - 1)
                    hs = hi[entry_i]; ei, ex = end, cl[end]
                    for j in range(entry_i, end + 1):
                        hs = max(hs, hi[j]); csl = max(dn, hs - atr_fin[j] * p['atr_mult'])
                        if lo[j] <= csl: ei, ex = j, csl; break
                    add('optimize_finance', de, ex / entry_px - 1.0 - COST)
                    cd['optimize_finance'] = ei

    rows = {}
    for k, lst in rec.items():
        if not lst:
            continue
        df = pd.DataFrame(lst, columns=['entry', 'pnl'])
        rows[k] = df.groupby(df['entry'].dt.year)['pnl'].sum()
    tbl = pd.DataFrame(rows).T
    tbl['TOTAL'] = tbl.sum(axis=1)

    pd.set_option('display.width', 200, 'display.max_columns', 50)
    print(f"\n=== PER-YEAR PnL ($100/trade, thr {args.threshold}, same entry) ===")
    print(tbl.round(0).to_string())
    print("\ntrades:", {k: len(v) for k, v in rec.items()})


if __name__ == '__main__':
    main()
