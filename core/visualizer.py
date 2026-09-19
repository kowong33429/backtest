import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    def __init__(self, df):
        self.df = df
        
    def plot_results(self, feature_importances=None, tail_bars=1000):
        if tail_bars is not None:
            plot_df = self.df.tail(tail_bars).copy()
        else:
            plot_df = self.df.copy()

        if feature_importances is not None and not feature_importances.empty:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=False,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=('Price Action & Signals', 'Top Feature Importances')
            )
        else:
            fig = make_subplots(rows=1, cols=1, subplot_titles=('Price Action & Signals',))

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=plot_df['Date'],
            open=plot_df['Open'],
            high=plot_df['High'],
            low=plot_df['Low'],
            close=plot_df['Close'],
            name='Candlestick'
        ), row=1, col=1)

        # Labels
        if 'Signal' in plot_df.columns:
            buy_signals = plot_df[plot_df['Signal'] == 1]
            sell_signals = plot_df[plot_df['Signal'] == 2]
            
            fig.add_trace(go.Scatter(
                x=buy_signals['Date'], y=buy_signals['Low'] * 0.98,
                mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'),
                name='True Label: Long'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=sell_signals['Date'], y=sell_signals['High'] * 1.02,
                mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'),
                name='True Label: Short'
            ), row=1, col=1)
                            
        # ML Probabilities
        if 'ML_Prob_Long' in plot_df.columns:
            high_prob_long = plot_df[plot_df['ML_Prob_Long'] > 0.8]
            fig.add_trace(go.Scatter(
                x=high_prob_long['Date'], y=high_prob_long['Low'] * 0.95,
                mode='markers', marker=dict(symbol='star', size=10, color='gold'),
                name='ML Long > 80%'
            ), row=1, col=1)
            
        if 'ML_Prob_Short' in plot_df.columns:
            high_prob_short = plot_df[plot_df['ML_Prob_Short'] > 0.8]
            fig.add_trace(go.Scatter(
                x=high_prob_short['Date'], y=high_prob_short['High'] * 1.05,
                mode='markers', marker=dict(symbol='diamond', size=10, color='mediumpurple'),
                name='ML Short > 80%'
            ), row=1, col=1)

        # Feature Importance
        if feature_importances is not None and not feature_importances.empty:
            sorted_feats = feature_importances.sort_values(ascending=True)
            fig.add_trace(go.Bar(
                x=sorted_feats.values,
                y=sorted_feats.index,
                orientation='h',
                name='Importance',
                marker_color='royalblue'
            ), row=2, col=1)

        title = 'ZECUSDT Agentic Quant System (Long/Short)'
        if tail_bars:
            title += f' (Last {tail_bars} bars)'
            
        fig.update_layout(
            title=title, xaxis_rangeslider_visible=False,
            template='plotly_dark', height=800 if (feature_importances is not None and not feature_importances.empty) else 600
        )
        fig.show()

    def plot_feature_distributions(self, feature_names):
        """Plot Box plots for the top features, grouped by Signal"""
        if 'Signal' not in self.df.columns:
            return
            
        features = [f for f in feature_names if f in self.df.columns][:5]
        if not features:
            return
            
        fig = make_subplots(rows=len(features), cols=1, subplot_titles=features)
        
        for i, feat in enumerate(features, 1):
            fig.add_trace(go.Box(y=self.df[self.df['Signal'] == 0][feat], name='Signal=0 (Noise)', marker_color='gray'), row=i, col=1)
            fig.add_trace(go.Box(y=self.df[self.df['Signal'] == 1][feat], name='Signal=1 (Long)', marker_color='lime'), row=i, col=1)
            fig.add_trace(go.Box(y=self.df[self.df['Signal'] == 2][feat], name='Signal=2 (Short)', marker_color='red'), row=i, col=1)
            
        fig.update_layout(title='Feature Distributions (Signal vs Noise)', template='plotly_dark', height=250 * len(features))
        fig.show()

    def plot_evaluation_metrics(self, res_long, res_short, tp_pct, sl_pct):
        """Plot Equity Curve (Fixed vs Dynamic Kelly) and Fold Stability"""
        fig = make_subplots(
            rows=2, cols=1, 
            subplot_titles=('Simulated Equity Curve (Fixed Risk vs Dynamic Kelly)', 'Cross-Validation Fold Stability')
        )
        
        # Calculate Equity
        equity_fixed = [10000]
        equity_kelly = [10000]
        win_count, loss_count, conflict_count = 0, 0, 0
        
        b = tp_pct / sl_pct # Reward to Risk Ratio
        
        yt_l, yp_l, prob_l = res_long['all_y_true'], res_long['all_y_pred'], res_long['all_y_prob']
        yt_s, yp_s, prob_s = res_short['all_y_true'], res_short['all_y_pred'], res_short['all_y_prob']
        
        for i in range(len(yt_l)):
            p_long = prob_l[i] if yp_l[i] == 1 else 0
            p_short = prob_s[i] if yp_s[i] == 1 else 0
            
            # Conflict Resolution
            if p_long > 0.8 and p_short > 0.8:
                conflict_count += 1
                equity_fixed.append(equity_fixed[-1])
                equity_kelly.append(equity_kelly[-1])
                continue
                
            # Long Trade
            if p_long > 0.8:
                f_kelly = max(0, p_long - (1 - p_long) / b)
                half_kelly = f_kelly * 0.5
                
                if yt_l[i] == 1:
                    equity_fixed.append(equity_fixed[-1] * (1 + tp_pct))
                    equity_kelly.append(equity_kelly[-1] * (1 + (tp_pct * half_kelly * 10))) # Assume max 10x effective Kelly leverage scaling
                    win_count += 1
                else:
                    equity_fixed.append(equity_fixed[-1] * (1 - sl_pct))
                    equity_kelly.append(equity_kelly[-1] * (1 - (sl_pct * half_kelly * 10)))
                    loss_count += 1
                    
            # Short Trade
            elif p_short > 0.8:
                f_kelly = max(0, p_short - (1 - p_short) / b)
                half_kelly = f_kelly * 0.5
                
                if yt_s[i] == 1: # True label is 2 (Short), wait, the res_short is trained on binary target where 1 is the positive class
                    equity_fixed.append(equity_fixed[-1] * (1 + tp_pct))
                    equity_kelly.append(equity_kelly[-1] * (1 + (tp_pct * half_kelly * 10)))
                    win_count += 1
                else:
                    equity_fixed.append(equity_fixed[-1] * (1 - sl_pct))
                    equity_kelly.append(equity_kelly[-1] * (1 - (sl_pct * half_kelly * 10)))
                    loss_count += 1
            else:
                equity_fixed.append(equity_fixed[-1])
                equity_kelly.append(equity_kelly[-1])
        
        fig.add_trace(go.Scatter(y=equity_fixed, mode='lines', name='Fixed Risk (1%)', line=dict(color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(y=equity_kelly, mode='lines', name='Dynamic Kelly (Half-Kelly)', line=dict(color='cyan', width=2)), row=1, col=1)
        
        # 2. Fold Stability
        fig.add_trace(go.Scatter(
            x=[f"Fold {j+1}" for j in range(len(res_long['fold_precisions']))], y=res_long['fold_precisions'],
            mode='lines+markers', name='Long Precision per Fold', marker=dict(size=10, color='lime')
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[f"Fold {j+1}" for j in range(len(res_short['fold_precisions']))], y=res_short['fold_precisions'],
            mode='lines+markers', name='Short Precision per Fold', marker=dict(size=10, color='red')
        ), row=2, col=1)
        
        fig.update_layout(title=f'Trade Analytics (Wins: {win_count}, Losses: {loss_count}, Conflicts Avoided: {conflict_count})', template='plotly_dark', height=700)
        fig.show()
