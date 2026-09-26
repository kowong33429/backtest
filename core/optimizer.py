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

    def _drop_highly_correlated_features(self, X, threshold=0.9):
        """Drops one of each pair of features with correlation > threshold."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        if to_drop:
            print(f"    [Filter] Dropping {len(to_drop)} highly correlated features.")
        return X.drop(columns=to_drop)

    def _train_model(self, X, y, tscv):
        """Helper to train a single model (Long or Short) using TimeSeriesSplit."""
        precisions = []
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        last_model = None
        last_X_test = None

        if y.sum() < 5:
            return None

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
                n_jobs=1,
                verbosity=0,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            prec = precision_score(y_test, preds, zero_division=0) if sum(preds) > 0 else 0.0
            precisions.append(prec)
            
            all_y_true.extend(y_test.values)
            all_y_pred.extend(preds)
            all_y_prob.extend(probs)
            
            last_model = model
            last_X_test = X_test

        avg_precision = float(np.mean(precisions)) if precisions else 0.0

        importances = None
        if last_model is not None:
            importances = pd.Series(
                last_model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

        return {
            'avg_precision': avg_precision,
            'importances': importances,
            'fold_precisions': precisions,
            'all_y_true': all_y_true,
            'all_y_pred': all_y_pred,
            'all_y_prob': all_y_prob,
            'last_model': last_model,
            'last_X_test': last_X_test
        }

    def evaluate_params(self, tp_pct, sl_pct, max_bars=300,
                        momentum_bars=42, momentum_min_pct=0.03):
        """Evaluate a single TP/SL config using TimeSeriesSplit CV for both Long and Short."""
        # 1. Generate Labels (with Momentum Filter)
        labeler = self.label_generator_class(
            self.df, window=20, tp_pct=tp_pct, sl_pct=sl_pct, max_bars=max_bars,
            momentum_bars=momentum_bars, momentum_min_pct=momentum_min_pct
        )
        df_labeled = labeler.generate_labels()

        # 2. Prepare X and y
        drop_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Signal']
        df_numeric = df_labeled.select_dtypes(include=[np.number])
        X = df_numeric.drop(columns=[c for c in drop_cols if c in df_numeric.columns])
        
        # Correlation Filter
        X = self._drop_highly_correlated_features(X, threshold=0.9)

        # 3. Create independent labels
        y_long = (df_labeled['Signal'] == 1).astype(int)
        y_short = (df_labeled['Signal'] == 2).astype(int)
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        print("    Training Long Model...")
        res_long = self._train_model(X, y_long, tscv)
        print("    Training Short Model...")
        res_short = self._train_model(X, y_short, tscv)
        
        if res_long is None and res_short is None:
            return None
            
        # Combine metrics
        prec_long = res_long['avg_precision'] if res_long else 0.0
        prec_short = res_short['avg_precision'] if res_short else 0.0
        combined_prec = (prec_long + prec_short) / 2.0
        
        # Merge importances (average them)
        imp_long = res_long['importances'] if res_long else pd.Series(dtype=float)
        imp_short = res_short['importances'] if res_short else pd.Series(dtype=float)
        combined_imp = pd.concat([imp_long, imp_short], axis=1).mean(axis=1).sort_values(ascending=False)

        return {
            'avg_precision': combined_prec,
            'prec_long': prec_long,
            'prec_short': prec_short,
            'importances': combined_imp,
            'res_long': res_long,
            'res_short': res_short
        }

    def run_sweep(self, tp_range, sl_range):
        """Grid search over TP/SL combinations. Returns top 3 by combined precision."""
        print(f"Starting parameter sweep. TP={tp_range}, SL={sl_range}")
        results = []

        for tp in tp_range:
            for sl in sl_range:
                res = self.evaluate_params(tp, sl)
                if res is None:
                    continue
                    
                prec = res['avg_precision']
                importances = res['importances']
                
                results.append({
                    'tp_pct': tp,
                    'sl_pct': sl,
                    'precision': prec,
                    'prec_long': res['prec_long'],
                    'prec_short': res['prec_short'],
                    'top_features': importances.head(10).to_dict() if not importances.empty else {},
                    'res_long': res['res_long'],
                    'res_short': res['res_short']
                })

        results.sort(key=lambda x: x['precision'], reverse=True)
        return results[:3]
