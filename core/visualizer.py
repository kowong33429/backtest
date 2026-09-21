import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    def __init__(self, df, trade_log=None, symbol="ZECUSDT"):
        self.df = df
        self.trade_log = trade_log
        self.symbol = symbol
        
    def plot_results(self, feature_importances=None, tail_bars=1000):
        if tail_bars is not None:
            plot_df = self.df.tail(tail_bars).copy()
        else:
            plot_df = self.df.copy()

        if feature_importances:
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

        # Realistic Trades
        if self.trade_log is not None and not self.trade_log.empty:
            start_date = plot_df['Date'].min()
            end_date = plot_df['Date'].max()
            
            # Filter trades that occurred within the plot_df timeframe
            visible_trades = self.trade_log[
                (self.trade_log['Entry_Date'] >= start_date) & 
                (self.trade_log['Entry_Date'] <= end_date)
            ]
            
            for _, trade in visible_trades.iterrows():
                is_long = trade['Type'] == 'Long'
                if is_long:
                    color = 'lime' if trade['PnL_Pct'] > 0 else 'red'
                    symbol = 'triangle-up'
                else:
                    color = 'cyan' if trade['PnL_Pct'] > 0 else 'orange'
                    symbol = 'triangle-down'
                
                # Draw line from Entry to Exit
                fig.add_trace(go.Scatter(
                    x=[trade['Entry_Date'], trade['Exit_Date']],
                    y=[trade['Entry_Price'], trade['Exit_Price']],
                    mode='lines+markers',
                    line=dict(color=color, width=2, dash='dot'),
                    marker=dict(symbol=symbol, size=10),
                    name=f"{trade['Type']} ({trade['PnL_Pct']*100:.1f}%)",
                    showlegend=False
                ), row=1, col=1)

        # Feature Importance
        if feature_importances:
            import pandas as pd
            feat_series = pd.Series(feature_importances)
            sorted_feats = feat_series.sort_values(ascending=True)
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
            template='plotly_dark', height=800 if feature_importances else 600
        )
        out_path = f"data/{self.symbol.lower()}/price_action.html"
        fig.write_html(out_path)
        print(f"  📊 บันทึกกราฟ Price Action ไว้ที่: {out_path}")

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
        out_path = f"data/{self.symbol.lower()}/feature_dist.html"
        fig.write_html(out_path)
        print(f"  📊 บันทึกกราฟ Feature Distribution ไว้ที่: {out_path}")

    def plot_evaluation_metrics(self, res_long, res_short, tp_pct, sl_pct):
        """Plot Realistic Equity Curve and Fold Stability"""
        fig = make_subplots(
            rows=2, cols=1, 
            subplot_titles=('Realistic Simulated Equity Curve', 'Cross-Validation Fold Stability')
        )
        
        # Calculate Equity from Trade Log
        win_count, loss_count = 0, 0
        if self.trade_log is not None and not self.trade_log.empty:
            equity_curve = [10000]
            dates = [self.df['Date'].iloc[0]]
            
            for _, trade in self.trade_log.iterrows():
                # Apply 1x leverage, simple compounding
                new_equity = equity_curve[-1] * (1 + trade['PnL_Pct'])
                equity_curve.append(new_equity)
                dates.append(trade['Exit_Date'])
                
                if trade['PnL_Pct'] > 0:
                    win_count += 1
                else:
                    loss_count += 1
            
            fig.add_trace(go.Scatter(x=dates, y=equity_curve, mode='lines+markers', name='Equity (1x)', line=dict(color='cyan', width=2)), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=[0], y=[10000], mode='lines', name='No Trades', line=dict(color='gray')), row=1, col=1)
        
        # 2. Fold Stability
        if res_long:
            fig.add_trace(go.Scatter(
                x=[f"Fold {j+1}" for j in range(len(res_long.get('fold_precisions', [])))], y=res_long.get('fold_precisions', []),
                mode='lines+markers', name='Long Precision per Fold', marker=dict(size=10, color='lime')
            ), row=2, col=1)
        if res_short:
            fig.add_trace(go.Scatter(
                x=[f"Fold {j+1}" for j in range(len(res_short.get('fold_precisions', [])))], y=res_short.get('fold_precisions', []),
                mode='lines+markers', name='Short Precision per Fold', marker=dict(size=10, color='red')
            ), row=2, col=1)
        
        fig.update_layout(title=f'Trade Analytics (Wins: {win_count}, Losses: {loss_count})', template='plotly_dark', height=700)
        out_path = f"data/{self.symbol.lower()}/equity_curve.html"
        fig.write_html(out_path)
        print(f"  📊 บันทึกกราฟ Equity Curve ไว้ที่: {out_path}")

    def plot_correlation_heatmap(self, feature_names):
        """Plot Correlation Heatmap for the top features"""
        features = [f for f in feature_names if f in self.df.columns]
        if not features:
            return
            
        corr = self.df[features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title='Top Features Correlation Heatmap',
            template='plotly_dark',
            width=800, height=800
        )
        out_path = f"data/{self.symbol.lower()}/correlation.html"
        fig.write_html(out_path)
        print(f"  📊 บันทึกกราฟ Correlation ไว้ที่: {out_path}")
