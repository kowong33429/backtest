import sys, os, numpy as np, pandas as pd, json, time
sys.path.insert(0,'core')
from train_entry_model import purged_walk_forward
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score as AP, roc_auc_score
panel_path, meta_path, tag = sys.argv[1], sys.argv[2], sys.argv[3]
S=os.environ['SCR']
p=pd.read_parquet(panel_path).sort_values(['Date','symbol']).reset_index(drop=True)
nonf={'Date','symbol','Entry_Label','Label_EndDate','Sample_Weight'}
feats=[c for c in p.columns if c not in nonf]
TA=('RSI_','ADX_','BB_Width_','Dist_SMA_','Return_','Dist_Low_','Dist_High_','Volume_Surge_','Pos_In_','Dist_VWAP_')
ta=[c for c in feats if c.startswith(TA) or c in ('SMA_Cross','ATR_Ratio')]
non=[c for c in feats if c not in ta]
meta=json.load(open(meta_path)); prm=meta.get('xgb_params') or {}
y=p.Entry_Label.astype(int).values; w=p.Sample_Weight.values; d=p.Date.values; le=p.Label_EndDate.values
spw=(len(y)-y.sum())/y.sum()
prm=dict(n_estimators=300,learning_rate=0.03,max_depth=4,subsample=0.8,colsample_bytree=0.8,scale_pos_weight=spw,random_state=42,n_jobs=6,verbosity=0,eval_metric='aucpr')
nf=int(meta.get('n_splits',6))
folds=list(purged_walk_forward(d,le,nf,3))
variants={'A_full':feats,'B_TA_only':ta,'C_full_minus_HY_and_Funding':[c for c in feats if c not in('HY_SPREAD','Funding_Rate')],'D_TA_plus_FG_Funding_VIX_only':ta+[c for c in ['FearGreed','VIX','Funding_Rate'] if c in feats]}
res=[]
for name,cols in variants.items():
    t0=time.time(); X=p[cols].astype('float32').fillna(0.0); oof=np.full(len(y),np.nan)
    for tr,te in folds:
        m=XGBClassifier(**prm).fit(X.iloc[tr],y[tr],sample_weight=w[tr]); oof[te]=m.predict_proba(X.iloc[te])[:,1]
    k=~np.isnan(oof); yy=y[k]; pp=oof[k]; dd=pd.Series(pp).groupby(d[k]).transform('mean').values
    r=dict(variant=name,n_feat=len(cols),base=yy.mean(),pr_auc=AP(yy,pp),roc=roc_auc_score(yy,pp),date_mean_only_pr=AP(yy,dd),xsec_pr=AP(yy,pp-dd),xsec_roc=roc_auc_score(yy,pp-dd))
    print(tag,r,'%.0fs'%(time.time()-t0),flush=True); res.append(r)
    o=p[['Date','symbol','Entry_Label']].copy(); o['oof_prob_xgb']=oof; o.to_parquet(f'{S}/oof_{tag}_{name}.parquet',index=False)
pd.DataFrame(res).to_csv(f'{S}/abl_{tag}.csv',index=False)
