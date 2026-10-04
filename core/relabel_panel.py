"""
relabel_panel.py — Re-label an existing feature panel with a new TP/SL/horizon.

Features do NOT depend on the stop level, so to change the label we only need to
recompute EntryLabeler per coin (from the raw CSVs) and swap the label columns in
the existing panel — far faster than rebuilding all features.
"""
import os
import sys
import glob
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labeler import EntryLabeler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-panel', default='data/model/entry_panel.parquet')
    ap.add_argument('--out-panel', default='data/model/entry_panel_dd20.parquet')
    ap.add_argument('--tp', type=float, default=1.00)
    ap.add_argument('--sl', type=float, default=0.20)
    ap.add_argument('--horizon-days', type=int, default=30)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = os.path.join(base, args.in_panel)
    out_path = os.path.join(base, args.out_panel)

    panel = pd.read_parquet(in_path)
    panel['Date'] = pd.to_datetime(panel['Date'])
    print(f"[Relabel] loaded panel {panel.shape} | {panel['symbol'].nunique()} coins")

    csvs = {os.path.basename(c).replace('_4h_full.csv', '').upper(): c
            for c in glob.glob(os.path.join(base, 'data', '*', '*_4h_full.csv'))}

    new_frames = []
    for k, sym in enumerate(sorted(panel['symbol'].unique()), 1):
        csv = csvs.get(sym)
        if csv is None:
            print(f"  {sym}: CSV missing — kept old labels")
            new_frames.append(panel[panel['symbol'] == sym])
            continue
        df = pd.read_csv(csv)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        lab = EntryLabeler(df, tp_pct=args.tp, sl_pct=args.sl,
                           horizon_days=args.horizon_days).generate()
        lab = lab[['Date', 'Entry_Label', 'Label_EndDate', 'Sample_Weight']]
        sub = panel[panel['symbol'] == sym].drop(
            columns=['Entry_Label', 'Label_EndDate', 'Sample_Weight'])
        merged = sub.merge(lab, on='Date', how='inner')
        new_frames.append(merged)

    out = pd.concat(new_frames, ignore_index=True).sort_values(['Date', 'symbol']).reset_index(drop=True)
    out.to_parquet(out_path, index=False)
    pos = int(out['Entry_Label'].sum())
    print(f"\n[Relabel] +{args.tp*100:.0f}%/{args.horizon_days}d/-{args.sl*100:.0f}%: "
          f"{out.shape[0]} rows, {pos} positives ({pos/len(out)*100:.2f}%), "
          f"scale_pos_weight ~= {(len(out)-pos)/max(1,pos):.1f}")
    print(f"[Relabel] saved -> {out_path}")


if __name__ == '__main__':
    main()
