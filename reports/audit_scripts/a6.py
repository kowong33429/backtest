import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'core'); sys.path.insert(0,os.environ['SCR'])
import entry_backtest as eb
def sim(oof_path, root, mask_fn, label, seed=0, frac=None):
    o=pd.read_parquet(oof_path); o['Date']=pd.to_datetime(o['Date']); o=o.dropna(subset=['oof_prob_xgb'])
    sig=o[mask_fn(o)]; tr=[]
    for sym,g in sig.groupby('symbol'):
        p=f'{root}/{sym.lower()}/{sym}_4h_full.csv'
        if not os.path.exists(p): continue
        df=pd.read_csv(p); df['Date']=pd.to_datetime(df['Date'])
        for t in eb.simulate_coin(df,set(g.Date),1.0,0.4,360,0.001,0.0005,100.0): t['symbol']=sym; tr.append(t)
    td=pd.DataFrame(tr); n=td.net_ret
    print(f'{label}: trades {len(td)} win {(n>0).mean():.3f} exp {n.mean()*100:+.2f}% pf {n[n>0].sum()/-n[n<=0].sum():.2f}',flush=True); return td
rng=np.random.default_rng(1)
for nm,oof,root in [('basket','data/model/v2/m1_100pct_60d/entry_oof.parquet','data'),('universe','data_all/model/m1_univ/entry_oof.parquet','data_all')]:
    # buy EVERY bar (cooldown per coin) = unconditional benchmark
    sim(oof,root,lambda o:o.oof_prob_xgb>=0,f'{nm} ALL bars (no model)')
    # random bars matched to thr0.6 fraction
    o=pd.read_parquet(oof).dropna(subset=['oof_prob_xgb']); fr=(o.oof_prob_xgb>=0.6).mean(); print(nm,'thr0.6 frac',round(fr,3))
    for s in range(3):
        r=np.random.default_rng(s); sim(oof,root,lambda o:pd.Series(r.random(len(o))<fr,index=o.index),f'{nm} RANDOM matched-frequency seed{s}')
