import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    def __init__(self, df):
        self.df = df
        
    def plot_results(self, tail_bars=1000):
        if tail_bars is not None:
            plot_df = self.df.tail(tail_bars).copy()
        else:
            plot_df = self.df.copy()

        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(x=plot_df['Date'],
                    open=plot_df['Open'],
                    high=plot_df['High'],
                    low=plot_df['Low'],
                    close=plot_df['Close'],
                    name='Candlestick'))

        # Buy Signals (Valid from Triple Barrier)
        if 'Signal' in plot_df.columns:
            buy_signals = plot_df[plot_df['Signal'] == 1]
            fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Low'] * 0.98,
                            mode='markers',
                            marker=dict(symbol='triangle-up', size=15, color='lime'),
                            name='Triple Barrier Buy'))
                            
        # ML Probabilities (if available)
        if 'ML_Prob_Buy' in plot_df.columns:
            # We can't easily plot this on the same y-axis without messing up the scale.
            # Instead, we just add hover text or highlight bars where probability is > 80%
            high_prob = plot_df[plot_df['ML_Prob_Buy'] > 0.8]
            fig.add_trace(go.Scatter(x=high_prob['Date'], y=high_prob['Low'] * 0.95,
                            mode='markers',
                            marker=dict(symbol='star', size=10, color='gold'),
                            name='ML Confidence > 80%'))

        title = 'ZECUSDT Agentic Quant System Results'
        if tail_bars:
            title += f' (Last {tail_bars} bars)'
            
        fig.update_layout(
            title=title,
            yaxis_title='Price (USDT)',
            xaxis_title='Date',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        fig.show()
