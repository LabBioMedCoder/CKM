"""Figure 6 step 1: build the private analysis dataset (never exported).

Inputs : CKM_data_0605.csv (clinical/outcomes), raw_protein.parquet (Olink, instance 0)
Outputs: private/analysis.parquet (participant-level covariates/outcomes),
         private/proteins.npy (float32, aligned rows), private/protein_names.txt,
         results/01_cohort_flow.csv (aggregate counts only)
"""
import numpy as np, pandas as pd, json
from pathlib import Path
R = Path('/home/ubuntu/Figure6_Prediction')
OUT = ['Dementia', 'Alzheimers_disease', 'Parkinsons_disease', 'Stroke', 'Anxiety_disorder',
       'Bipolar_disorder', 'Depression', 'Schizophrenia', 'Sleep_disorder']
cov = ['Eid', 'Age', 'Sex', 'Ethnicity', 'Smoking_status', 'Alcohol_intake_frequency', 'Qualifications',
       'townsend_index', 'APOE4', 'BMI', 'SBP', 'HTN_meds', 'HbA1c', 'Diabetes.x', 'Diabetes.y', 'LDLc', 'HDLc',
       'TG', 'statin', 'eGFR', 'stage0', 'stage1', 'stage2', 'stage3', 'stage4']
d = pd.read_csv('/home/ubuntu/upload/CKM_data_0605.csv', usecols=cov + OUT + [o + '_time' for o in OUT])
assert d.Eid.is_unique
assert (d[['stage0', 'stage1', 'stage2', 'stage3', 'stage4']].sum(1) == 1).all()
assert d['Diabetes.x'].equals(d['Diabetes.y'])
d['stage'] = np.select([d.stage0 == 1, d.stage1 == 1, d.stage2 == 1, d.stage3 == 1, d.stage4 == 1], [0, 1, 2, 3, 4])
p = pd.read_parquet('/home/ubuntu/CKM_Protein_Function_Brain/private/raw_protein.parquet')
p['eid'] = pd.to_numeric(p.eid).astype('int64')
assert p.eid.is_unique
flow = [('clinical file', len(d))]
d = d[d.Eid.isin(set(p.eid))]; flow.append(('with instance-0 proteomics', len(d)))
d = d[d.stage <= 3]; flow.append(('CKM stage 0-3 (stage 4 excluded)', len(d)))
T = d[[o + '_time' for o in OUT]]
bad_time = T.isna().any(axis=1) | (T <= 0).any(axis=1) | (T > 17).any(axis=1)
flow.append(('excluded: implausible follow-up time (missing, <=0 or >17 y)', int(bad_time.sum())))
d = d[~bad_time]; flow.append(('plausible follow-up', len(d)))

# ---- covariate coding (prespecified) ----
d['Ethnicity4'] = d.Ethnicity.map({'White': 'White', 'South Asian': 'SouthAsian', 'Black': 'Black',
                                   'Chinese': 'Other', 'Mixed': 'Other', 'Other': 'Other', '-1': np.nan})
d['Smoking'] = d.Smoking_status.where(d.Smoking_status >= 0)
d['Alcohol'] = d.Alcohol_intake_frequency.where(d.Alcohol_intake_frequency > 0)
edu_map = {1: 'Degree', 2: 'Alevel', 3: 'Olevel', 4: 'Olevel', 5: 'Vocational', 6: 'Vocational', -7: 'None'}
d['Education'] = d.Qualifications.map(edu_map)
d['logTG'] = np.log(d.TG)
need = ['Age', 'Sex', 'Ethnicity4', 'Smoking', 'Alcohol', 'Education', 'townsend_index', 'APOE4', 'BMI', 'SBP',
        'HTN_meds', 'HbA1c', 'Diabetes.x', 'LDLc', 'HDLc', 'logTG', 'statin', 'eGFR', 'stage']
miss = {c: int(d[c].isna().sum()) for c in need}
cc = d[need].notna().all(axis=1)
flow.append(('excluded: incomplete covariates (common-sample complete case)', int((~cc).sum())))
d = d[cc].copy(); flow.append(('analytic common sample', len(d)))
# winsorise continuous clinical predictors at 0.5/99.5 percentiles (no outcome information used)
for c in ['BMI', 'SBP', 'HbA1c', 'LDLc', 'HDLc', 'logTG', 'eGFR', 'townsend_index']:
    lo, hi = d[c].quantile([0.005, 0.995]); d[c] = d[c].clip(lo, hi)
d = d.rename(columns={'Diabetes.x': 'Diabetes'})
keep = ['Eid', 'stage', 'Age', 'Sex', 'Ethnicity4', 'Smoking', 'Alcohol', 'Education', 'townsend_index', 'APOE4',
        'BMI', 'SBP', 'HTN_meds', 'HbA1c', 'Diabetes', 'LDLc', 'HDLc', 'logTG', 'statin', 'eGFR'] + OUT + [o + '_time' for o in OUT]
d = d[keep].reset_index(drop=True)
P = p.set_index('eid').loc[d.Eid]
names = list(P.columns)
X = P.to_numpy(np.float32)
assert np.isfinite(X).all() and X.shape == (len(d), 2923)
(R / 'private').mkdir(exist_ok=True)
d.to_parquet(R / 'private/analysis.parquet', index=False)
np.save(R / 'private/proteins.npy', X)
(R / 'private/protein_names.txt').write_text('\n'.join(names))
fl = pd.DataFrame(flow, columns=['step', 'n'])
fl.to_csv(R / 'results/01_cohort_flow.csv', index=False)
ev = pd.DataFrame({'outcome': OUT, 'events': [int(d[o].sum()) for o in OUT],
                   'events_10y': [int(((d[o] == 1) & (d[o + '_time'] <= 10)).sum()) for o in OUT],
                   'median_followup': [round(float(d[o + '_time'].median()), 2) for o in OUT]})
for s in range(4): ev[f'events_S{s}'] = [int(d.loc[d.stage == s, o].sum()) for o in OUT]
ev.to_csv(R / 'results/01_events_analytic_sample.csv', index=False)
print(fl.to_string(index=False)); print('missing counts before complete case:', json.dumps(miss))
print('stage counts', d.stage.value_counts().sort_index().to_dict()); print(ev.to_string(index=False))
