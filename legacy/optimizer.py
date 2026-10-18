"""
optimizer.py — XGBoost Hyperparameter Sweeping with TimeSeriesSplit

กฎเหล็ก:
- ใช้ TimeSeriesSplit เท่านั้น (ห้ามใช้ random split)
- Feature importance ดึงจาก fold สุดท้าย ไม่ใช่ full-data fit (ป้องกัน data leakage)
- ใช้ scale_pos_weight เพื่อรับมือกับ class imbalance (Label_Long/Short=1 มีน้อยมาก)
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score


class QuantOptimizer:
    def __init__(self, df_features, label_generator_class, n_splits=5, top_k_features=25):
        self.df = df_features
        self.label_generator_class = label_generator_class
        self.n_splits = n_splits          # walk-forward folds (expanding window)
        self.top_k_features = top_k_features  # per-side feature cap after importance ranking

    def _drop_highly_correlated_features(self, X, threshold=0.75):
        """Drops one of each pair of features with correlation > threshold.
        AGENTS.md Rule #4: drop redundant features with correlation > 0.75."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        if to_drop:
            print(f"    [Filter] Dropping {len(to_drop)} highly correlated features.")
        return X.drop(columns=to_drop)

    def _train_model(self, X, y, tscv, tp_pct, sl_pct):
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

        # Out-of-fold probabilities aligned to X's row positions. Each bar is
        # predicted ONLY by a model that never trained on it (walk-forward).
        # The backtester uses this array so it never trades on in-sample fit.
        # The first fold's training block is never in any test set -> stays NaN
        # (no signal there); that is the expected warm-up region.
        oof_prob = np.full(len(X), np.nan, dtype=float)

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

            oof_prob[test_idx] = probs  # positional fill (RangeIndex on X)

            prec = precision_score(y_test, preds, zero_division=0) if sum(preds) > 0 else 0.0
            precisions.append(prec)

            all_y_true.extend(y_test.values)
            all_y_pred.extend(preds)
            all_y_prob.extend(probs)

            last_model = model
            last_X_test = X_test

        avg_precision = float(np.mean(precisions)) if precisions else 0.0

        # Derive the probability threshold from OUT-OF-FOLD predictions only
        # (never the training fit). This is the threshold the backtester will
        # trade on, so it must be chosen the same way it will be used.
        prob_threshold, precision_at_threshold = self._best_threshold(
            all_y_true, all_y_prob, tp_pct, sl_pct
        )

        importances = None
        if last_model is not None:
            importances = pd.Series(
                last_model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

        return {
            'avg_precision': avg_precision,
            'prob_threshold': prob_threshold,
            'precision_at_threshold': precision_at_threshold,
            'importances': importances,
            'fold_precisions': precisions,
            'all_y_true': all_y_true,
            'all_y_pred': all_y_pred,
            'all_y_prob': all_y_prob,
            'oof_prob': oof_prob,
            'last_model': last_model,
            'last_X_test': last_X_test
        }

    def _train_and_select(self, X, y, tscv, tp_pct, sl_pct):
        """Fit once to rank feature importances for THIS side, keep the top
        drivers (importance > 0, capped at top_k_features), then refit on that
        reduced set so Long and Short models use their own features."""
        pre = self._train_model(X, y, tscv, tp_pct, sl_pct)
        if pre is None or pre['importances'] is None:
            return pre

        ranked = pre['importances']
        selected = [c for c in ranked.index if ranked[c] > 0][:self.top_k_features]
        if not selected or len(selected) == len(X.columns):
            return pre  # nothing to prune

        print(f"      [Select] {len(selected)}/{len(X.columns)} features kept for this side.")
        return self._train_model(X[selected], y, tscv, tp_pct, sl_pct)

    def _best_threshold(self, y_true, y_prob, tp_pct, sl_pct,
                        min_signals=10, cap=0.75):
        """Pick the probability cutoff that MAXIMIZES total expected value on the
        pooled out-of-fold predictions, capped at `cap`.

        Rationale: maximizing precision alone drives the cutoff toward ~0.95,
        which fires almost nothing and misses the biggest trends entirely (the
        model traded only 61 times in 7.5y and skipped 2021/2024/2025). Total EV
        rewards catching MORE true positives, not just being right on a handful:

            payoff(signal) = +tp_pct if it hit TP (y_true=1) else -sl_pct
            total_EV(t)    = n_selected(t) * [ prec*tp_pct - (1-prec)*sl_pct ]

        A lower cutoff that captures many winners can beat a high one that takes
        two lucky signals. The hard `cap` (default 0.75) guarantees the strategy
        keeps trading rather than sneaking back up to a near-1.0 threshold.

        Returns (threshold, precision_at_threshold). Falls back to (0.5, 0.0)
        when there are too few positives to choose meaningfully.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        if y_true.size == 0:
            return 0.5, 0.0

        best_t, best_ev, best_prec = 0.5, -np.inf, 0.0
        found = False
        for t in np.arange(0.30, cap + 1e-9, 0.01):
            selected = y_prob >= t
            n_sel = int(selected.sum())
            if n_sel < min_signals:
                continue
            prec = float(y_true[selected].mean())  # TP / predicted-positives
            ev_per_signal = prec * tp_pct - (1.0 - prec) * sl_pct
            total_ev = n_sel * ev_per_signal
            if total_ev > best_ev:
                best_ev, best_t, best_prec = total_ev, float(t), prec
                found = True
        return (best_t, best_prec) if found else (0.5, 0.0)

    def evaluate_params(self, tp_pct, sl_pct, max_bars=700, use_atr=False):
        """Evaluate a single TP/SL config using TimeSeriesSplit CV for both Long and Short."""
        # 1. Generate Labels (Dynamic Triple-Barrier)
        labeler = self.label_generator_class(
            self.df, tp_pct=tp_pct, sl_pct=sl_pct, max_bars=max_bars, use_atr=use_atr
        )
        df_labeled = labeler.generate_labels()

        # 2. Prepare X and y
        drop_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                     'Label_Long', 'Label_Short']
        df_numeric = df_labeled.select_dtypes(include=[np.number])
        X = df_numeric.drop(columns=[c for c in drop_cols if c in df_numeric.columns])

        # Correlation Filter (AGENTS.md Rule #4: drop > 0.75)
        X = self._drop_highly_correlated_features(X, threshold=0.75)

        # 3. Independent binary targets (two separate models — AGENTS.md Step 4)
        y_long = df_labeled['Label_Long'].astype(int)
        y_short = df_labeled['Label_Short'].astype(int)
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # Long and Short each SELECT THEIR OWN features (AGENTS.md Step 4 trains
        # two independent models): fit once to rank importances, keep the top
        # drivers for that side, then refit on the reduced set. A feature that
        # matters for shorts may be pure noise for longs.
        print("    Training Long Model...")
        res_long = self._train_and_select(X, y_long, tscv, tp_pct, sl_pct)
        print("    Training Short Model...")
        res_short = self._train_and_select(X, y_short, tscv, tp_pct, sl_pct)

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
            'thr_long': res_long['prob_threshold'] if res_long else 0.5,
            'thr_short': res_short['prob_threshold'] if res_short else 0.5,
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
                    'thr_long': res['thr_long'],
                    'thr_short': res['thr_short'],
                    'top_features': importances.head(10).to_dict() if not importances.empty else {},
                    'res_long': res['res_long'],
                    'res_short': res['res_short']
                })

        results.sort(key=lambda x: x['precision'], reverse=True)
        return results[:3]
