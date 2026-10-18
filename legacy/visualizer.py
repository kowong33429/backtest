import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    def __init__(self, df, trade_log=None, symbol="ZECUSDT"):
        self.df = df
        self.trade_log = trade_log
        self.symbol = symbol
        
    def plot_results(self, feature_importances=None, tail_bars=1000):
        """Emit ONE chart per direction (Long / Short) so each is easy to read.
        Each chart overlays only that direction's trades and carries an explicit
        legend explaining every marker/colour."""
        for direction in ('Long', 'Short'):
            self._plot_direction(direction, feature_importances, tail_bars)

    def _plot_direction(self, direction, feature_importances, tail_bars):
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
                subplot_titles=(f'{direction} Trades', 'Top Feature Importances')
            )
        else:
            fig = make_subplots(rows=1, cols=1, subplot_titles=(f'{direction} Trades',))

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=plot_df['Date'],
            open=plot_df['Open'],
            high=plot_df['High'],
            low=plot_df['Low'],
            close=plot_df['Close'],
            name='Price (OHLC)'
        ), row=1, col=1)

        # Marker convention (Long: green=win / red=loss, triangle-up;
        #                    Short: cyan=win / orange=loss, triangle-down)
        win_color = 'lime' if direction == 'Long' else 'cyan'
        loss_color = 'red' if direction == 'Long' else 'orange'
        marker_symbol = 'triangle-up' if direction == 'Long' else 'triangle-down'

        # Realistic Trades — only this direction
        legend_shown = {'win': False, 'loss': False}
        if self.trade_log is not None and not self.trade_log.empty:
            start_date = plot_df['Date'].min()
            end_date = plot_df['Date'].max()

            visible_trades = self.trade_log[
                (self.trade_log['Type'] == direction) &
                (self.trade_log['Entry_Date'] >= start_date) &
                (self.trade_log['Entry_Date'] <= end_date)
            ]

            for _, trade in visible_trades.iterrows():
                is_win = trade['PnL_Pct'] > 0
                color = win_color if is_win else loss_color
                key = 'win' if is_win else 'loss'
                # Show each legend entry once so the legend stays a clean key.
                show = not legend_shown[key]
                legend_shown[key] = True
                fig.add_trace(go.Scatter(
                    x=[trade['Entry_Date'], trade['Exit_Date']],
                    y=[trade['Entry_Price'], trade['Exit_Price']],
                    mode='lines+markers',
                    line=dict(color=color, width=2, dash='dot'),
                    marker=dict(symbol=marker_symbol, size=10),
                    name=f"{direction} {'Win' if is_win else 'Loss'}",
                    legendgroup=key,
                    showlegend=show,
                    hovertemplate=(f"{direction} {'WIN' if is_win else 'LOSS'}<br>"
                                   f"PnL: {trade['PnL_Pct']*100:.1f}%<br>"
                                   f"Exit: {trade['Reason']}<extra></extra>")
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
                marker_color='royalblue',
                showlegend=False
            ), row=2, col=1)

        # Marker legend as a caption so the human knows what each symbol means.
        caption = (f"Markers: ▲/▼ = {direction} entry→exit line (dotted). "
                   f"{win_color} = winning trade, {loss_color} = losing trade. "
                   f"Hover a marker for PnL % and exit reason.")
        title = f'{self.symbol} — {direction} Trades'
        if tail_bars:
            title += f' (Last {tail_bars} bars)'

        fig.update_layout(
            title=title, xaxis_rangeslider_visible=False,
            template='plotly_dark', height=800 if feature_importances else 600,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            annotations=list(fig.layout.annotations) + [dict(
                text=caption, showarrow=False, xref='paper', yref='paper',
                x=0, y=-0.08, align='left', font=dict(size=11, color='lightgray')
            )]
        )
        out_path = f"data/{self.symbol.lower()}/price_action_{direction.lower()}.html"
        fig.write_html(out_path)
        print(f"  📊 บันทึกกราฟ {direction} Trades ไว้ที่: {out_path}")

    def plot_feature_distributions(self, feature_names):
        """Plot Box plots for the top features, grouped by the two independent
        binary labels (Label_Long / Label_Short). A candle can be positive for
        both, so the Long and Short positive groups may overlap — that is
        expected under the two-model scheme."""
        if 'Label_Long' not in self.df.columns or 'Label_Short' not in self.df.columns:
            return

        features = [f for f in feature_names if f in self.df.columns][:5]
        if not features:
            return

        fig = make_subplots(rows=len(features), cols=1, subplot_titles=features)

        for i, feat in enumerate(features, 1):
            noise = self.df[(self.df['Label_Long'] == 0) & (self.df['Label_Short'] == 0)][feat]
            fig.add_trace(go.Box(y=noise, name='Noise (neither)', marker_color='gray'), row=i, col=1)
            fig.add_trace(go.Box(y=self.df[self.df['Label_Long'] == 1][feat], name='Label_Long=1', marker_color='lime'), row=i, col=1)
            fig.add_trace(go.Box(y=self.df[self.df['Label_Short'] == 1][feat], name='Label_Short=1', marker_color='red'), row=i, col=1)

        fig.update_layout(title='Feature Distributions (Long/Short positives vs Noise)', template='plotly_dark', height=250 * len(features))
        out_path = f"data/{self.symbol.lower()}/feature_dist.html"
        fig.write_html(out_path)
        print(f"  📊 บันทึกกราฟ Feature Distribution ไว้ที่: {out_path}")

    def plot_evaluation_metrics(self, res_long, res_short, tp_pct, sl_pct):
        """Plot the NET (sized, after-fee) equity curve, its drawdown, and
        cross-validation fold stability (AGENTS.md Rule #10 Step 5).

        Equity compounds each trade's Net_Return (Kelly-sized position minus
        fees/slippage — Rule #7 & #9). Falls back to raw PnL_Pct at 1x only if
        the backtester didn't provide Net_Return."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Net Equity Curve (Kelly-sized, after fees)',
                            'Drawdown %',
                            'Cross-Validation Fold Stability')
        )

        win_count, loss_count = 0, 0
        if self.trade_log is not None and not self.trade_log.empty:
            use_net = 'Net_Return' in self.trade_log.columns
            ret_col = 'Net_Return' if use_net else 'PnL_Pct'

            equity_curve = [10000.0]
            dates = [self.df['Date'].iloc[0]]
            for _, trade in self.trade_log.iterrows():
                equity_curve.append(equity_curve[-1] * (1 + trade[ret_col]))
                dates.append(trade['Exit_Date'])
                if trade[ret_col] > 0:
                    win_count += 1
                else:
                    loss_count += 1

            name = 'Net Equity' if use_net else 'Equity (1x, gross)'
            fig.add_trace(go.Scatter(x=dates, y=equity_curve, mode='lines+markers',
                                     name=name, line=dict(color='cyan', width=2)), row=1, col=1)

            # Drawdown from running peak
            peak, dd = equity_curve[0], []
            for v in equity_curve:
                peak = max(peak, v)
                dd.append((v - peak) / peak * 100)
            fig.add_trace(go.Scatter(x=dates, y=dd, mode='lines', name='Drawdown %',
                                     fill='tozeroy', line=dict(color='orange')), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=[0], y=[10000], mode='lines', name='No Trades', line=dict(color='gray')), row=1, col=1)

        # Fold Stability
        if res_long:
            fig.add_trace(go.Scatter(
                x=[f"Fold {j+1}" for j in range(len(res_long.get('fold_precisions', [])))], y=res_long.get('fold_precisions', []),
                mode='lines+markers', name='Long Precision per Fold', marker=dict(size=10, color='lime')
            ), row=3, col=1)
        if res_short:
            fig.add_trace(go.Scatter(
                x=[f"Fold {j+1}" for j in range(len(res_short.get('fold_precisions', [])))], y=res_short.get('fold_precisions', []),
                mode='lines+markers', name='Short Precision per Fold', marker=dict(size=10, color='red')
            ), row=3, col=1)

        fig.update_layout(title=f'Trade Analytics (Wins: {win_count}, Losses: {loss_count})', template='plotly_dark', height=950)
        out_path = f"data/{self.symbol.lower()}/equity_curve.html"
        fig.write_html(out_path)
        print(f"  📊 บันทึกกราฟ Equity Curve ไว้ที่: {out_path}")

    def plot_param_stability(self, sweep_df, x_col='atr_sl_mult', y_col='trail_pct', z_col='sharpe'):
        """2D Parameter-Stability Heatmap (AGENTS.md Rule #10 Step 5).

        Shows a performance metric (default net Sharpe) across a grid of exit
        parameters (ATR SL multiplier × trailing-stop %). A robust strategy shows
        a smooth plateau of good values, not a single lucky spiky cell — this is
        the human sanity check against over-optimization."""
        if sweep_df is None or sweep_df.empty:
            print("  [Heatmap] No sweep results to plot.")
            return
        pivot = sweep_df.pivot(index=y_col, columns=x_col, values=z_col)
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale='RdYlGn', zmid=0,
            colorbar=dict(title=z_col),
            text=[[f"{v:.2f}" for v in row] for row in pivot.values],
            texttemplate="%{text}", hovertemplate=f"{x_col}=%{{x}}<br>{y_col}=%{{y}}<br>{z_col}=%{{z:.3f}}<extra></extra>"
        ))
        fig.update_layout(
            title=f'Exit-Parameter Stability — {z_col} across {x_col} × {y_col}',
            xaxis_title=x_col, yaxis_title=y_col,
            template='plotly_dark', width=800, height=600
        )
        out_path = f"data/{self.symbol.lower()}/param_stability.html"
        fig.write_html(out_path)
        print(f"  📊 บันทึกกราฟ Parameter Stability ไว้ที่: {out_path}")

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
