import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0,'core')
import rotation_backtest as rb
S=r'C:/Users/User/AppData/Local/Temp/claude/c--Project-backtest/c8133345-49e8-4a78-8722-bdcdd17b9945/scratchpad'
def args_for(preset,root,oof,top):
    a=rb.apply_preset(rb.get_parser().parse_args(['--preset',preset,'--root',root,'--oof',oof]+(['--universe-top',str(top)] if top else [])))
    return a
def funding_daily(root,cols):
    F={}
    for s in cols:
        p=f'{root}/{s.lower()}/{s}_4h_full.csv'
        if os.path.exists(p):
            d=pd.read_csv(p,usecols=['Date','Funding_Rate'],parse_dates=['Date']).set_index('Date')
            F[s]=(d.Funding_Rate/2).resample('D').sum(min_count=1)   # rate repeats over 2 bars per 8h settlement
    return pd.DataFrame(F)
base='.'
for name,preset,root,oof,top in [('b7 universe','b7','data_all','data_all/model/m1_univ/entry_oof.parquet',120),('btc-trend universe','btc-trend','data_all','data_all/model/m1_univ/entry_oof.parquet',120)]:
    a=args_for(preset,root,oof,top); data=rb.load_inputs(a,'.'); W=rb.build_weights(a,data)
    W=W.loc[data['oof']['day'].min():]
    O,C=data['O'],data['C']
    pr,to=rb.run_book(W,O.loc[W.index],C.loc[W.index],a.cost)
    F=funding_daily(root,W.columns).reindex(index=W.index,columns=W.columns).fillna(0)
    drag=(W*F).sum(axis=1)   # long pays funding when rate>0
    yrs=(W.index[-1]-W.index[0]).days/365.25
    eq0=(1+pr).cumprod(); eq1=(1+pr-drag).cumprod()
    cg=lambda e:e.iloc[-1]**(1/yrs)-1
    print(f'{name}: CAGR {cg(eq0):+.1%} -> with funding {cg(eq1):+.1%}; avg annual funding drag {drag.sum()/yrs:+.2%} of NAV; mean gross exposure {W.sum(axis=1).mean():.2f}')
    # regime split by BTC trend
    btc=C['BTCUSDT'].reindex(W.index); up=(btc>btc.rolling(200).mean()).shift(1)
    r=pd.DataFrame({'r':pr,'up':up}).dropna()
    for k,g in r.groupby('up'): print('   BTC>SMA200' if k else '   BTC<SMA200',f'days {len(g)} total {np.prod(1+g.r)-1:+.1%} ann.mean {g.r.mean()*365:+.1%} vol {g.r.std()*np.sqrt(365):.1%}')
    bh=btc.pct_change().shift(-1)  # approx
# basket b7 regime
a=args_for('b7','data','data/model/v2/m1_100pct_60d/entry_oof.parquet',0); data=rb.load_inputs(a,'.'); W=rb.build_weights(a,data); W=W.loc[data['oof']['day'].min():]
pr,to=rb.run_book(W,data['O'].loc[W.index],data['C'].loc[W.index],a.cost)
btc=data['C']['BTCUSDT'].reindex(W.index); up=(btc>btc.rolling(200).mean()).shift(1); r=pd.DataFrame({'r':pr,'up':up}).dropna()
for k,g in r.groupby('up'): print('basket b7', 'BTC>SMA200' if k else 'BTC<SMA200',f'days {len(g)} total {np.prod(1+g.r)-1:+.1%}')
# BTC buy and hold same window (universe)
a=args_for('b7','data_all','data_all/model/m1_univ/entry_oof.parquet',120); data=rb.load_inputs(a,'.')
O=data['O']['BTCUSDT']; d0=data['oof']['day'].min(); o=O.loc[d0:]; yrs=(o.index[-1]-o.index[0]).days/365.25
print('BTC hold CAGR since',d0.date(),(o.iloc[-1]/o.iloc[0])**(1/yrs)-1)
