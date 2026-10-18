import pandas as pd, numpy as np, json, xgboost as xgb
t=pd.read_csv('data/model/short_v2/entry_backtest_trades.csv',parse_dates=['entry_date','exit_date'])
print(t.columns.tolist(),len(t)); print(open('data/model/short_v2/threshold_sweep.csv').read())
c=0.003
rec=t.entry_price/t.exit_price-1-2*0.0015
real=1-t.exit_price/t.entry_price-2*0.0015
def st(r,n):
    print(n,'n',len(r),'win %.3f exp %.4f pf %.2f sum$ %.0f'%((r>0).mean(),r.mean(),r[r>0].sum()/-r[r<=0].sum(),(r*100).sum()))
st(t.net_ret,'logged(reciprocal)'); st(rec,'recomputed reciprocal'); st(real,'REAL short (1-exit/entry)')
print(t.outcome.value_counts().to_dict())
print(real.groupby(t.outcome).agg(['mean','count']))
t['yr']=t.entry_date.dt.year; print(pd.DataFrame({'rec':rec,'real':real,'yr':t.yr}).groupby('yr').agg(['mean','count']))
# worst case: SL real
sl=t[t.outcome=='SL']; print('SL real loss avg',(1-sl.exit_price/sl.entry_price).mean(), 'logged',(sl.entry_price/sl.exit_price-1).mean())
# top3 removal
s=real.sort_values(ascending=False)
print('real exp drop top3',s.iloc[3:].mean())
# model gains
for name,p in [('m1','data/model/v2/m1_100pct_60d/entry_xgb.json'),('m1_univ','data_all/model/m1_univ/entry_xgb.json'),('short_v2','data/model/short_v2/entry_xgb.json')]:
    b=xgb.Booster(); b.load_model(p); g=b.get_score(importance_type='total_gain'); s=pd.Series(g); s=s/s.sum()
    macro=[f for f in s.index if not f.startswith(('RSI_','ADX_','BB_Width_','Dist_SMA_','Return_','Dist_Low_','Dist_High_','Volume_Surge_','Pos_In_','Dist_VWAP_','SMA_Cross','ATR_Ratio'))]
    print('\n',name,'nfeat',len(s),'macro/other share of gain %.3f'%s[macro].sum(), 'n macro/other',len(macro))
    print(s.sort_values(ascending=False).head(12).round(4).to_dict())
    print('non-TA:',s[macro].sort_values(ascending=False).round(4).to_dict())
