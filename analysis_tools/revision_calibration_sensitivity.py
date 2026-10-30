"""Audit scalar-temperature convergence using calibration labels only.

Retain all archived primary transformations. This is a solver sensitivity
analysis, not external-data-selected recalibration and not model redesign.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_reanalysis import METRICS,metric_samples

ROOT=Path(__file__).resolve().parents[1]/'implementation/outputs/v2/final_validation/frozen_baselines/B3_CDG_Cysto_v2'
OUT=Path(__file__).resolve().parents[1]/'revision_analysis'
rows=[];temps=[]
for seed in (42,2025,3407):
    sd=ROOT/f'seed_{seed}';f=pd.read_csv(sd/'cystods_calib/logits_raw.csv')
    difference=(f.logit_1-f.logit_0).to_numpy();y=f.y_true.to_numpy();old=float((sd/'temperature.txt').read_text())
    def probability(s,d):return 1/(1+np.exp(-np.clip(s*d,-700,700)))
    def gradient(s):return float(np.mean((probability(s,difference)-y)*difference))
    lo=0.;hi=1.
    while gradient(hi)<0 and hi<1e6:hi*=2
    if gradient(0)>=0:inverse=0.
    else:
        for _ in range(100):
            mid=(lo+hi)/2
            if gradient(mid)<0:lo=mid
            else:hi=mid
        inverse=(lo+hi)/2
    new=1/inverse if inverse>0 else float('inf')
    temps.append(dict(seed=seed,archived_temperature=old,converged_temperature=new,inverse_temperature_gradient=gradient(inverse),archived_log_temperature_gradient=float(-gradient(1/old)/old)))
    for dataset in ['cystods_calib','cystods_internal','ebtc_external']:
        f=pd.read_csv(sd/dataset/'logits_raw.csv');d=(f.logit_1-f.logit_0).to_numpy();yy=f.y_true.to_numpy()
        for state,t in [('archived',old),('converged_calibration_only',new)]:
            pp=probability(1/t,d);metrics=metric_samples(yy,pp,np.ones((1,len(yy))))[0]
            rows.append(dict(seed=seed,dataset=dataset,transformation=state,temperature=t,**dict(zip(METRICS,metrics))))
        assert np.array_equal(probability(1/old,d)>=.5,probability(1/new,d)>=.5)
pd.DataFrame(temps).to_csv(OUT/'temperature_convergence_sensitivity.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(rows).to_csv(OUT/'calibration_solver_sensitivity_metrics.csv',index=False,encoding='utf-8-sig')
(OUT/'calibration_solver_protocol.json').write_text(json.dumps({'objective':'Unweighted CystoDS calibration mean NLL, convex in inverse positive temperature.','solver':'Monotone derivative bracketing and 100 bisection iterations; seed-specific; no internal-test or EBTC label used in fitting.','primary_policy':'Archived temperatures retained in the main reanalysis to reproduce the submitted fixed models; converged values reported only as solver sensitivity.','default_decisions_preserved':True},indent=2),encoding='utf-8')
print(pd.DataFrame(temps).to_string(index=False))
