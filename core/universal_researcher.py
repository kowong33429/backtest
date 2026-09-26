"""
universal_researcher.py — Train a single Universal XGBoost Model across multiple coins.
(Cross-Sectional / Global Model approach)
"""
import os
import json
import pandas as pd
import numpy as np
import argparse

from data_loader import DataLoader
from features import FeatureEngineer
from labels import LabelGenerator
from optimizer import QuantOptimizer


class PreLabeledGenerator:
    """Mock LabelGenerator to bypass LabelGenerator inside QuantOptimizer,
    since we pre-labeled and concatenated the data ourselves."""
    def __init__(self, df, **kwargs):
        self.df = df
    def generate_labels(self):
        return self.df


def get_priority_symbols(base_dir):
    priority_file = os.path.join(base_dir, 'data', 'scan_results', 'scanned_priority_coins.json')
    if os.path.exists(priority_file):
        with open(priority_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def prepare_coin_data(symbol, base_dir):
    csv_path = os.path.join(base_dir, 'data', symbol.replace('USDT', '').lower() + 'usdt', f"{symbol}_4h_full.csv")
    if not os.path.exists(csv_path):
        return None
    
    loader = DataLoader(csv_path, symbol)
    df = loader.get_full_data()
    
    engineer = FeatureEngineer(df)
    df_features = engineer.generate_all_features()
    
    # CRITICAL: Drop coin-specific absolute values (Absolute Prices, MACD, ATR, etc.)
    # A universal model must only see percentages, ratios, and normalized indicators.
    absolute_cols = ['MACD', 'MACD_Hist', 'MACD_Lag1', 'MACD_Hist_Lag1', 'ATR_14', 'SMA_50', 'SMA_200', 'OFI_Proxy']
    df_features.drop(columns=[c for c in absolute_cols if c in df_features.columns], inplace=True)
    
    return df_features


def main():
    parser = argparse.ArgumentParser(description='Train Universal Model')
    parser.add_argument('--limit', type=int, default=10, 
                        help='Limit number of coins to train on (default 10 to save time. Use 150 for all)')
    args = parser.parse_args()

    core_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(core_dir)

    symbols = get_priority_symbols(base_dir)
    if not symbols:
        print("No priority coins found in data/scan_results/scanned_priority_coins.json.")
        return

    symbols = symbols[:args.limit]
    
    print("=" * 60)
    print(f"🚀 INITIALIZING UNIVERSAL MODEL TRAINING ({len(symbols)} Coins)")
    print("=" * 60)
    
    all_coin_dfs = {}
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Processing features for {sym}...")
        df = prepare_coin_data(sym, base_dir)
        if df is not None and not df.empty:
            all_coin_dfs[sym] = df

    if not all_coin_dfs:
        print("No data loaded. Exiting.")
        return

    # Sweep Parameters for Universal Model
    tp_range = [0.10, 0.15, 0.20]
    sl_range = [0.10, 0.15]
    
    best_prec = 0
    best_config = {}
    
    print("\n" + "=" * 60)
    print("🔍 Starting Universal Hyperparameter Sweep...")
    print("=" * 60)
    
    for tp in tp_range:
        for sl in sl_range:
            print(f"\n  -> Testing Configuration: TP = {tp*100:.0f}%, SL = {sl*100:.0f}%")
            
            # 1. Generate labels for each coin separately, then concatenate
            concat_list = []
            for sym, df in all_coin_dfs.items():
                labeler = LabelGenerator(df, window=20, tp_pct=tp, sl_pct=sl)
                df_labeled = labeler.generate_labels()
                concat_list.append(df_labeled)
                
            df_universal = pd.concat(concat_list, ignore_index=True)
            
            # 2. Sort by date so TimeSeriesSplit still works chronologically across all coins
            df_universal = df_universal.sort_values('Date').reset_index(drop=True)
            
            print(f"     Combined Dataset: {len(df_universal)} rows")
            
            # 3. Train the model using QuantOptimizer
            opt = QuantOptimizer(df_universal, PreLabeledGenerator)
            res = opt.evaluate_params(tp, sl) # Passed to mock labeler
            
            if res:
                prec = res['avg_precision']
                print(f"     [Result] Combined Precision: {prec*100:.2f}% (Long: {res['prec_long']*100:.2f}%, Short: {res['prec_short']*100:.2f}%)")
                if prec > best_prec:
                    best_prec = prec
                    best_config = res
                    best_config['tp'] = tp
                    best_config['sl'] = sl
            else:
                print("     [Result] Not enough trade signals triggered.")

    print("\n" + "=" * 60)
    print("🏆 UNIVERSAL MODEL TRAINING COMPLETE")
    print("=" * 60)
    if best_config:
        print(f"Best Config: TP = {best_config['tp']*100:.0f}%, SL = {best_config['sl']*100:.0f}%")
        print(f"Top Combined Precision: {best_prec*100:.2f}%")
        print("\nTop 10 Universal Features (Factors driving the entire crypto market):")
        for feat, score in best_config['importances'].head(10).items():
            print(f"  - {feat:<20} {score:.4f}")
    else:
        print("Failed to find any profitable configuration.")
    print("=" * 60)

if __name__ == '__main__':
    main()
