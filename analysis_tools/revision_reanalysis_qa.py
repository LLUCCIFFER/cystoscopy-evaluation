"""Check weighted bootstrap primitives against explicit integer-row expansion."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from revision_reanalysis import metric_samples,logistic_calibration,METRICS
from revision_evidence_audit import binary_metrics

checks=[]
for seed in range(20):
    rng=np.random.default_rng(seed)
    y=rng.integers(0,2,30);p=np.round(rng.uniform(.01,.99,30),2)
    w=rng.integers(0,5,30)
    ii=np.repeat(np.arange(30),w)
    expected=binary_metrics(y[ii],p[ii])
    actual=metric_samples(y,p,w)[0]
    for metric in ['auroc','auprc','sensitivity','specificity','macro_f1','balanced_accuracy','ppv','npv','brier','nll']:
        assert np.isclose(actual[METRICS.index(metric)],expected[metric],atol=1e-12), (seed,metric,actual,expected)
        checks.append(f'weighted_vs_expanded_seed{seed}_{metric}')
    p2=metric_samples(y[ii],p[ii],np.ones((1,len(ii))))[0]
    assert np.allclose(actual,p2,atol=1e-12,equal_nan=True)
    checks.append(f'all_weighted_metrics_expansion_seed{seed}')

single=metric_samples(np.zeros(5,int),np.linspace(.1,.9,5),np.ones((1,5)))[0]
assert np.isnan(single[METRICS.index('auroc')]) and np.isnan(single[METRICS.index('auprc')])
assert np.isfinite(single[METRICS.index('brier')]) and np.isfinite(single[METRICS.index('nll')])
checks.append('one_class_resample_handling')

rng=np.random.default_rng(123)
x=rng.normal(size=10000);truth=1/(1+np.exp(-(.4+.7*x)));y=rng.binomial(1,truth)
f=logistic_calibration(y,1/(1+np.exp(-x)))
assert f['converged'] and abs(f['intercept']-.4)<.06 and abs(f['slope']-.7)<.06
checks.append('logistic_intercept_slope_known_simulation')

d=Path(__file__).resolve().parents[1]/'revision_analysis'
pts=pd.read_csv(d/'seedwise_and_mean_cluster_intervals.csv')
for dataset in ['cystods_internal','ebtc_external']:
    m=pts[(pts.dataset==dataset)&(pts.method=='B3')&pts.seed.isin(['42','2025','3407'])].groupby('metric').estimate.mean()
    fixed=pts[(pts.dataset==dataset)&(pts.method=='B3')&(pts.seed=='fixed_three_mean')&pts.grouping.isin(['official_pid','case_proxy'])].set_index('metric').estimate
    assert np.allclose(m.sort_index(),fixed.sort_index(),atol=1e-12)
    checks.append('primary_estimand_alignment_'+dataset)
out={'checks_passed':len(checks),'checks':checks,'all_passed':True}
(d/'analysis_qa.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Passed',len(checks),'checks: weighted metrics, outcome-degenerate handling, logistic calibration, fixed-model estimand.')
