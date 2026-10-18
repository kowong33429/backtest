import sys, pandas as pd, yfinance as yf
sys.path.insert(0,'core')
from entry_finder import MacroSnapshot
m=MacroSnapshot('2024-03-01','2024-03-20').build(); t=m.table
v=yf.download('^VIX',start='2024-03-05',end='2024-03-12',progress=False)['Close'].squeeze()
print(v); print(t.loc['2024-03-07':'2024-03-12',['VIX']])
print(t.loc['2024-03-01':'2024-03-20',['CPI_YoY','M2_YoY','HY_SPREAD','FEDFUNDS']].iloc[::4])
from pandas_datareader import data as web
h=web.DataReader('BAMLH0A0HYM2','fred','2019-01-01','2026-10-10'); print('HY_SPREAD FRED first obs',h.dropna().index.min())
