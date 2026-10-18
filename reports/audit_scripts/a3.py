import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'core')
import entry_backtest as eb
def run(oof_path, root, thr, label, tp=1.0, sl=0.4, hd=60, fee=0.001, slip=0.0005):
    o=pd.read_parquet(oof_path); o['Date']=pd.to_datetime(o['Date']); o=o.dropna(subset=['oof_prob_xgb'])
    sig=o[o.oof_prob_xgb>=thr]; tr=[]
    for sym,g in sig.groupby('symbol'):
        p=f'{root}/{sym.lower()}/{sym}_4h_full.csv'
        if not os.path.exists(p): continue
        df=pd.read_csv(p); df['Date']=pd.to_datetime(df['Date'])
        for t in eb.simulate_coin(df,set(g.Date),tp,sl,hd*6,fee,slip,100.0): t['symbol']=sym; tr.append(t)
    td=pd.DataFrame(tr); n=td.net_ret
    print(f'{label} thr {thr}: trades {len(td)} win {(n>0).mean():.3f} exp {n.mean()*100:+.2f}% pf {n[n>0].sum()/-n[n<=0].sum():.2f} sum$ {td.pnl_usd.sum():,.0f} outcomes {td.outcome.value_counts().to_dict()}')
    td['yr']=pd.to_datetime(td.entry_date).dt.year
    print('   by year exp%:',(td.groupby('yr').net_ret.mean()*100).round(1).to_dict())
    return td
b=run('data/model/v2/m1_100pct_60d/entry_oof.parquet','data',0.6,'m1 basket')
u=run('data_all/model/m1_univ/entry_oof.parquet','data_all',0.6,'m1 universe(top120 PIT)')
u5=run('data_all/model/m1_univ/entry_oof.parquet','data_all',0.5,'m1 universe')
# basket model restricted to same window as universe OOF (2020-12-31+)
# overlap of windows: basket entries since 2021-01
bb=b[pd.to_datetime(b.entry_date)>='2021-01-01']; n=bb.net_ret; print('basket since 2021:',len(bb),'exp %.2f%%'%(n.mean()*100))
# basket trades restricted to coins that also in top120 PIT? skip
u.to_pickle(os.environ['TEMP']+'/u_trades.pkl'); b.to_pickle(os.environ['TEMP']+'/b_trades.pkl')
# 2x cost
u2=run('data_all/model/m1_univ/entry_oof.parquet','data_all',0.6,'m1 universe 2x cost',fee=0.002,slip=0.001)
