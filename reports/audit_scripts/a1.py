import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as AP, roc_auc_score
t=pd.read_csv('data/model/v2/m1_100pct_60d/thr06/entry_backtest_trades.csv',parse_dates=['entry_date','exit_date'])
print(t.columns.tolist()); n=len(t)
net=t.net_ret
print('n',n,'win',(net>0).mean(),'exp',net.mean(),'pf',net[net>0].sum()/-net[net<=0].sum(),'sum$',t.pnl_usd.sum())
print(t.outcome.value_counts().to_dict())
s=net.sort_values(ascending=False)
for k in (3,10,30): print('drop top',k,'exp',s.iloc[k:].mean(),'pnl$',(s.iloc[k:]*100).sum())
for m in (2,3):  # cost x
    extra=(m-1)*0.003; print('cost x',m,'exp',(net-extra).mean(),'pf',(net-extra)[net-extra>0].sum()/-(net-extra)[net-extra<=0].sum())
t['yr']=t.entry_date.dt.year
print(t.groupby('yr').agg(n=('net_ret','size'),exp=('net_ret','mean'),win=('net_ret',lambda x:(x>0).mean()),pnl=('pnl_usd','sum')))
print('symbols',t.symbol.nunique()); c=t.groupby('symbol').pnl_usd.sum().sort_values(ascending=False); print(c.head(8)); print('top5 sym share', c.head(5).sum()/c.sum())
# concurrency: overlapping trades same day
d=pd.concat([pd.Series(1,index=t.entry_date),pd.Series(-1,index=t.exit_date)]).sort_index().cumsum()
print('max concurrent',d.max(),'mean',d.mean())
# cluster: trades per entry day top
print('entries on top 10 days:',t.entry_date.dt.normalize().value_counts().head(5).to_dict())
# eff independent: dedupe per symbol within 60d
t=t.sort_values('entry_date'); 
# OOF
o=pd.read_parquet('data/model/v2/m1_100pct_60d/entry_oof.parquet').dropna(subset=['oof_prob_xgb'])
y=o.Entry_Label.values; p=o.oof_prob_xgb.values
print('OOF n',len(o),'base',y.mean(),'PR-AUC',AP(y,p),'ROC',roc_auc_score(y,p), 'LR PR',AP(y,o.oof_prob_lr.fillna(0)))
# time-only: mean prob of each date (all coins) as score
dm=o.groupby('Date').oof_prob_xgb.transform('mean'); print('date-mean-only PR-AUC',AP(y,dm), 'ROC',roc_auc_score(y,dm))
# within-date demeaned
dem=p-dm.values; print('within-date (cross-sectional) PR-AUC',AP(y,dem),'ROC',roc_auc_score(y,dem))
# label-demeaned: within date ROC avg
# per-year PR-AUC/base
o['yr']=o.Date.dt.year
for yr,g in o.groupby('yr'):
    if g.Entry_Label.sum()>0: print(yr,'base %.3f PR %.3f n %d'%(g.Entry_Label.mean(),AP(g.Entry_Label,g.oof_prob_xgb),len(g)))
# threshold precision
for th in (.5,.6,.7,.8,.9):
    m=p>=th; print(th,m.sum(),'prec',y[m].mean())
# episodes: positive runs per symbol
o=o.sort_values(['symbol','Date']); o['new']=(o.Entry_Label==1)&(o.groupby('symbol').Entry_Label.shift(1)!=1)
print('positive labelled bars',int(o.Entry_Label.sum()),'distinct positive episodes',int(o.new.sum()))
