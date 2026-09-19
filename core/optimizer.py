"""
optimizer.py — XGBoost Hyperparameter Sweeping with TimeSeriesSplit

กฎเหล็ก:
- ใช้ TimeSeriesSplit เท่านั้น (ห้ามใช้ random split)
- Feature importance ดึงจาก fold สุดท้าย ไม่ใช่ full-data fit (ป้องกัน data leakage)
- ใช้ scale_pos_weight เพื่อรับมือกับ class imbalance (Signal=1 มีน้อยมาก)
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score


class QuantOptimizer:
    def __init__(self, df_features, label_generator_class):
        self.df = df_features
        self.label_generator_class = label_generator_class

    def evaluate_params(self, tp_pct, sl_pct, max_bars=40):
        """Evaluate a single TP/SL config using TimeSeriesSplit CV."""
        # 1. Generate Labels
        labeler = self.label_generator_class(
            self.df, window=20, tp_pct=tp_pct, sl_pct=sl_pct, max_bars=max_bars
        )
        df_labeled = labeler.generate_labels()

        # 2. Prepare X and y
        drop_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Signal']
        df_numeric = df_labeled.select_dtypes(include=[np.number])
        X = df_numeric.drop(columns=[c for c in drop_cols if c in df_numeric.columns])
        y = df_labeled['Signal']

        if y.sum() < 5:
            return 0.0, None

        # 3. TimeSeriesSplit (no shuffling, no randomness)
        tscv = TimeSeriesSplit(n_splits=3)
        precisions = []
        last_model = None

        scale_weight = max(1, (len(y) - y.sum()) / max(1, y.sum()))

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if y_train.sum() == 0:
                continue

            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                scale_pos_weight=scale_weight,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            prec = precision_score(y_test, preds, zero_division=0) if sum(preds) > 0 else 0.0
            precisions.append(prec)
            last_model = model  # Keep only the last fold model

        avg_precision = float(np.mean(precisions)) if precisions else 0.0

        # Feature importance from the LAST fold only (no full-data leakage)
        importances = None
        if last_model is not None:
            importances = pd.Series(
                last_model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

        return avg_precision, importances

    def run_sweep(self, tp_range, sl_range):
        """Grid search over TP/SL combinations. Returns top 3 by precision."""
        print(f"Starting parameter sweep. TP={tp_range}, SL={sl_range}")
        results = []

        for tp in tp_range:
            for sl in sl_range:
                prec, importances = self.evaluate_params(tp, sl)
                results.append({
                    'tp_pct': tp,
                    'sl_pct': sl,
                    'precision': prec,
                    'top_features': importances.head(3).to_dict() if importances is not None else {},
                })

        results.sort(key=lambda x: x['precision'], reverse=True)
        return results[:3]
