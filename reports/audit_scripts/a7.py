import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'core'); import entry_backtest as eb
S=os.environ['SCR']
def sim(path, root, frac, label):
    o=pd.read_parquet(path).dropna(subset=['oof_prob_xgb']); o['Date']=pd.to_datetime(o.Date)
    thr=o.oof_prob_xgb.quantile(1-frac); sig=o[o.oof_prob_xgb>=thr]; tr=[]
    for sym,g in sig.groupby('symbol'):
        p=f'{root}/{sym.lower()}/{sym}_4h_full.csv'
        if not os.path.exists(p): continue
        df=pd.read_csv(p); df['Date']=pd.to_datetime(df.Date)
        tr+=[dict(t,symbol=sym) for t in eb.simulate_coin(df,set(g.Date),1.0,0.4,360,0.001,0.0005,100.0)]
    td=pd.DataFrame(tr); n=td.net_ret
    print(f'{label}: top{frac:.0%} thr {thr:.3f} trades {len(td)} win {(n>0).mean():.3f} exp {n.mean()*100:+.2f}% pf {n[n>0].sum()/-n[n<=0].sum():.2f}',flush=True)
for tag,root,frac in [('basket','data',0.24),('univ','data_all',0.24)]:
    for v in ['A_full','B_TA_only']:
        sim(f'{S}/oof_{tag}_{v}.parquet',root,frac,f'{tag} {v}')
