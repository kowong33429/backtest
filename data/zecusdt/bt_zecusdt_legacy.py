import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# --- 1. Load Data ---
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    # ตรวจสอบว่าข้อมูลเรียงจากอดีตมาปัจจุบัน
    df = df.sort_values('Date').reset_index(drop=True)
    return df

# --- 2. Mark Buy and Sell Points (จุดที่กำไรสูงสุดแต่ละช่วง) ---
def mark_extrema(df, window=15):
    """
    มาร์คจุดต่ำสุด (Buy) และจุดสูงสุด (Sell) แบบล่วงรู้อนาคต (Hindsight)
    """
    df = df.copy()
    
    # หาราคาต่ำสุดในกรอบเวลา
    df['Swing_Low'] = df['Low'] == df['Low'].rolling(window=window*2+1, center=True).min()
    
    # หาราคาสูงสุดในกรอบเวลา
    df['Swing_High'] = df['High'] == df['High'].rolling(window=window*2+1, center=True).max()
    
    # สร้างคอลัมน์ Signal: 1 (Buy), -1 (Sell), 0 (Hold)
    df['Signal'] = 0
    df.loc[df['Swing_Low'], 'Signal'] = 1
    df.loc[df['Swing_High'], 'Signal'] = -1
    
    # คอลัมน์ที่ระบุว่าเข้าและออกที่ราคาไหน
    df['Mark_Price'] = np.nan
    df.loc[df['Signal'] == 1, 'Mark_Price'] = df['Low']
    df.loc[df['Signal'] == -1, 'Mark_Price'] = df['High']
    
    return df

# --- 3. Visualize ---
def plot_interactive_chart(df, tail_bars=None):
    """
    วาดกราฟแท่งเทียนและจุดที่ทำการมาร์ค
    tail_bars: จำนวนแท่งล่าสุดที่จะวาด (ถ้ายาวเกินไปกราฟจะอืด ใช้จำกัดจำนวนแท่งได้ เช่น 1000)
    """
    if tail_bars is not None:
        plot_df = df.tail(tail_bars).copy()
    else:
        plot_df = df.copy()

    fig = go.Figure()

    # กราฟแท่งเทียน
    fig.add_trace(go.Candlestick(x=plot_df['Date'],
                open=plot_df['Open'],
                high=plot_df['High'],
                low=plot_df['Low'],
                close=plot_df['Close'],
                name='Candlestick'))

    # จุด Buy (เครื่องหมายสามเหลี่ยมชี้ขึ้น สีเขียว)
    buy_signals = plot_df[plot_df['Signal'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Low'] * 0.98,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='Buy (Swing Low)'))

    # จุด Sell (เครื่องหมายสามเหลี่ยมชี้ลง สีแดง)
    sell_signals = plot_df[plot_df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['High'] * 1.02,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='Sell (Swing High)'))

    # ปรับแต่ง Layout
    title = 'ZECUSDT Ideal Trade Marks'
    if tail_bars:
        title += f' (Last {tail_bars} bars)'
        
    fig.update_layout(
        title=title,
        yaxis_title='Price (USDT)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    # เปิดเบราว์เซอร์อัตโนมัติ
    fig.show()

# --- Main Execution ---
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "ZECUSDT_4h_full.csv"
    file_path = os.path.join(current_dir, file_name)
    
    if os.path.exists(file_path):
        # 1. โหลดข้อมูล
        df = load_data(file_path)
        
        # 2. หาจุดเข้าออกที่ดีที่สุด
        df_marked = mark_extrema(df, window=20) 
        
        print("คำนวณและมาร์คจุดต่ำสุด/สูงสุดเรียบร้อยแล้ว")
        print(f"พบจุด Buy (Swing Low): {len(df_marked[df_marked['Signal'] == 1])} จุด")
        print(f"พบจุด Sell (Swing High): {len(df_marked[df_marked['Signal'] == -1])} จุด")
        
        # 3. วาดกราฟ (เปิดใน Browser)
        print("\nกำลังเปิดกราฟบน Web Browser...")
        plot_interactive_chart(df_marked, tail_bars=1000)
        
    else:
        print(f"File not found: {file_path}")
