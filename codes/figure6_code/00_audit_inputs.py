import pandas as pd, numpy as np
OUT=['Dementia','Alzheimers_disease','Parkinsons_disease','Stroke','Anxiety_disorder','Bipolar_disorder','Depression','Schizophrenia','Sleep_disorder']
cov=['Eid','Age','Sex','Ethnicity','Smoking_status','Alcohol_intake_frequency','Qualifications','townsend_index','SES_PCA','APOE4','BMI','SBP','HTN_meds','HbA1c','Diabetes.x','LDLc','HDLc','TG','statin','eGFR','UACR','stage0','stage1','stage2','stage3','stage4']
d=pd.read_csv('/home/ubuntu/upload/CKM_data_0605.csv',usecols=cov+OUT+[o+'_time' for o in OUT])
p=pd.read_parquet('/home/ubuntu/CKM_Protein_Function_Brain/private/raw_protein.parquet')
p['eid']=pd.to_numeric(p.eid).astype('int64')
d=d[d.Eid.isin(set(p.eid))&(d.stage4==0)].copy()
print('protein & stage0-3',len(d))
for c in cov[1:21]: 
    v=d[c]; print(f'{c:26s} miss={v.isna().mean():.4f}', 'values='+str(sorted(v.dropna().unique())[:9]) if v.nunique()<12 else f'range={v.min():.2f}-{v.max():.2f}')
prot=p.set_index('eid').loc[d.Eid]
m=prot.isna().mean()
print('protein missing per protein: median',round(m.median(),4),'p95',round(m.quantile(.95),4),'max',round(m.max(),4),'n>20%',int((m>.2).sum()))
print('participant missing share: median',round(prot.isna().mean(1).median(),4),'p99',round(prot.isna().mean(1).quantile(.99),4),'n>50%',int((prot.isna().mean(1)>.5).sum()))
print('top missing proteins',m.sort_values().tail(6).round(3).to_dict())
for g in ['GFAP','NEFL']: print(g,'missing',round(m[g],4))
