"""Reviewer-requested conditional cluster inference from fixed saved predictions.

No model selection, temperature fitting, threshold refitting, or training occurs.
The primary estimand is the arithmetic mean of metrics of three fixed models,
not a metric of an ensemble probability and not a future-training expectation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'implementation'
BASE=ROOT/'outputs/v2/final_validation/frozen_baselines'
DEFAULT_OUT=Path(__file__).resolve().parents[1]/'revision_analysis'
SEEDS=(42,2025,3407)
METRICS=('auroc','auprc','sensitivity','specificity','macro_f1','balanced_accuracy','ppv','npv','brier','nll','positive_ece','top_ece')
INPUTS=set()

def read(path):
    path=Path(path);INPUTS.add(path);return pd.read_csv(path)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def norm_case(value):
    m=re.search(r'case_0*(\d+)',str(value),flags=re.I)
    if not m:raise ValueError(f'No case ID: {value}')
    return f'case_{int(m.group(1)):03d}'

def norm_case_pt(value):
    for pattern in (r'pt_0*(\d+)',r'pt0*(\d+)',r'cys_pt0*(\d+)'):
        m=re.search(pattern,str(value),flags=re.I)
        if m:return f'{norm_case(value)}_pt_{int(m.group(1)):03d}'
    return norm_case(value)+'_pt_unknown'

def metadata(dataset):
    name={'cystods_calib':'cystods_calib','cystods_internal':'cystods_test_internal','ebtc_external':'ebtc_external_all','cystods_validation':'cystods_val'}[dataset]
    return read(ROOT/f'outputs/v2/splits/{name}.csv').sort_values('sample_id').reset_index(drop=True)

def predictions(dataset,method='B3',seed=42,raw=False):
    path=next(BASE.glob(f'{method}_*/seed_{seed}/{dataset}/predictions_{"raw_" if raw else ""}default_0.5.csv'))
    return read(path).sort_values('sample_id').reset_index(drop=True)

def weights(groups,reps,rng):
    codes,uniques=pd.factorize(np.asarray(groups).astype(str),sort=True)
    counts=rng.multinomial(len(uniques),np.full(len(uniques),1/len(uniques)),size=reps)
    return counts[:,codes].astype(float),uniques

def metric_samples(y,p,w,threshold=.5):
    y=np.asarray(y,int);p=np.asarray(p,float);w=np.atleast_2d(np.asarray(w,float))
    pos=w@y;neg=w@(1-y);total=pos+neg
    decision=(p>=threshold).astype(float)
    tp=w@(decision*y);tn=w@((1-decision)*(1-y));fp=neg-tn;fn=pos-tp
    div=lambda a,b:np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.asarray(b)!=0)
    sens=div(tp,pos);spec=div(tn,neg)
    order=np.argsort(p,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
    wp=np.add.reduceat(w[:,order]*y[order],starts,axis=1)
    wn=np.add.reduceat(w[:,order]*(1-y[order]),starts,axis=1)
    auc=div(np.sum(wp*(np.cumsum(wn,axis=1)-.5*wn),axis=1),pos*neg)
    cp=np.cumsum(wp[:,::-1],axis=1);cn=np.cumsum(wn[:,::-1],axis=1)
    precision=np.divide(cp,cp+cn,out=np.zeros_like(cp),where=(cp+cn)!=0)
    ap=div(np.sum(precision*wp[:,::-1],axis=1),pos)
    ap[(pos==0)|(neg==0)]=np.nan
    brier=div(w@(2*(p-y)**2),total)
    clipped=np.clip(p,1e-15,1-1e-15)
    nll=div(w@(-y*np.log(clipped)-(1-y)*np.log1p(-clipped)),total)
    def ece(values,targets):
        result=np.zeros(len(w))
        bins=np.minimum((values*15).astype(int),14)
        for b in range(15):
            mask=bins==b
            if mask.any():result+=np.abs(w@(mask*(targets-values)))
        return div(result,total)
    pece=ece(p,y);tece=ece(np.maximum(p,1-p),((p>.5).astype(int)==y).astype(int))
    f1=(div(2*tp,2*tp+fp+fn)+div(2*tn,2*tn+fp+fn))/2
    f1[(pos==0)|(neg==0)]=np.nan
    return np.stack([auc,ap,sens,spec,f1,(sens+spec)/2,div(tp,tp+fp),div(tn,tn+fn),brier,nll,pece,tece],axis=1)

def summarize(samples,point,**keys):
    result=[]
    for j,name in enumerate(METRICS):
        vals=samples[:,j];valid=vals[np.isfinite(vals)]
        lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
        result.append(dict(**keys,metric=name,estimate=float(point[j]),ci_low=float(lo),ci_high=float(hi),valid_replicates=len(valid),invalid_replicates=len(vals)-len(valid)))
    return result

def logistic_calibration(y,p,offset=False):
    y=np.asarray(y,float);x=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
    if len(np.unique(y))!=2:return dict(intercept=np.nan,slope=np.nan,converged=False)
    X=np.ones((len(y),1)) if offset else np.column_stack((np.ones(len(y)),x))
    beta=np.array([0.]) if offset else np.array([0.,1.])
    zoff=x if offset else 0.
    converged=False
    for _ in range(100):
        eta=X@beta+zoff;pr=1/(1+np.exp(-np.clip(eta,-50,50)))
        gradient=X.T@(y-pr);h=X.T@(np.maximum(pr*(1-pr),1e-12)[:,None]*X)
        try:step=np.linalg.solve(h,gradient)
        except np.linalg.LinAlgError:break
        old=float(np.sum(np.logaddexp(0,eta)-y*eta));scale=1.
        while scale>1e-8:
            new=beta+scale*step;e=X@new+zoff
            if np.sum(np.logaddexp(0,e)-y*e)<=old+1e-10:break
            scale/=2
        beta=new
        if np.max(np.abs(scale*step))<1e-8:converged=True;break
        if np.max(np.abs(beta))>100:break
    return dict(intercept=float(beta[0]),slope=1. if offset else float(beta[1]),converged=converged)

def save(frame,out,name):pd.DataFrame(frame).to_csv(out/name,index=False,encoding='utf-8-sig')

def run(reps,out):
    start=time.time();out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261018)
    rows=[];group_rows=[];subrows=[];calrows=[];deltarows=[];reliability=[];loco=[];contrasts=[];checks=[]
    for dataset in ['cystods_internal','ebtc_external']:
        print('Dataset',dataset,flush=True)
        meta=metadata(dataset);y=meta.label_cancer_binary.to_numpy()
        ids=meta.pid.astype(str) if dataset.startswith('cystods') else meta.sample_id.map(norm_case)
        meta['group_id']=ids
        for gid,g in meta.groupby('group_id'):
            group_rows.append(dict(dataset=dataset,group_id=gid,images=len(g),positive_images=int(g.label_cancer_binary.sum()),negative_images=int(sum(g.label_cancer_binary==0)),original_labels=';'.join(sorted(g.label_original.unique())),modalities=';'.join(sorted(g.modality.unique())),stored_pid_values=';'.join(sorted(g.pid.astype(str).unique()))))
        W,_=weights(ids,reps,rng)
        method_boot={};method_points={}
        for method in ['B3','B1','B5']:
            mb=[];mp=[]
            for seed in SEEDS:
                f=predictions(dataset,method,seed)
                assert np.array_equal(f.sample_id,meta.sample_id) and np.array_equal(f.y_true,y)
                b=metric_samples(y,f.prob_1.to_numpy(),W);v=metric_samples(y,f.prob_1.to_numpy(),np.ones((1,len(y))))[0]
                mb.append(b);mp.append(v)
                rows.extend(summarize(b,v,dataset=dataset,method=method,seed=str(seed),grouping='case_proxy' if dataset=='ebtc_external' else 'official_pid',estimand='single_fixed_model'))
            method_boot[method]=np.stack(mb,axis=1);method_points[method]=np.stack(mp)
            rows.extend(summarize(np.mean(method_boot[method],axis=1),np.mean(mp,axis=0),dataset=dataset,method=method,seed='fixed_three_mean',grouping='case_proxy' if dataset=='ebtc_external' else 'official_pid',estimand='mean_of_three_fixed_model_metrics'))
            for j,name in enumerate(METRICS):checks.append(dict(dataset=dataset,method=method,metric=name,seed_sd=float(np.std(np.asarray(mp)[:,j],ddof=1)),seed_min=float(np.min(np.asarray(mp)[:,j])),seed_max=float(np.max(np.asarray(mp)[:,j]))))
        np.savez_compressed(out/f'{dataset}_common_cluster_bootstrap.npz',**method_boot)
        for baseline in ['B1','B5']:
            contrasts.extend(summarize(np.mean(method_boot['B3']-method_boot[baseline],axis=1),np.mean(method_points['B3']-method_points[baseline],axis=0),dataset=dataset,contrast='B3_minus_'+baseline,estimand='paired_mean_metric_difference'))
        for modality in sorted(meta.modality.unique()):
            mask=meta.modality==modality;mm=meta[mask];wg,ugs=weights(mm.group_id,reps,rng)
            positive_groups=mm.groupby('group_id').label_cancer_binary.agg(['min','max'])
            ms=[];pts=[]
            for seed in SEEDS:
                f=predictions(dataset,'B3',seed);pr=f.loc[mask,'prob_1'].to_numpy();yy=y[mask]
                b=metric_samples(yy,pr,wg);v=metric_samples(yy,pr,np.ones((1,len(yy))))[0]
                ms.append(b);pts.append(v)
            subrows.extend(summarize(np.mean(ms,axis=0),np.mean(pts,axis=0),dataset=dataset,modality=modality,images=len(mm),positive_images=int(mm.label_cancer_binary.sum()),negative_images=int(sum(mm.label_cancer_binary==0)),groups=len(ugs),positive_containing_groups=int(sum(positive_groups['max']==1)),negative_containing_groups=int(sum(positive_groups['min']==0)),wholly_negative_groups=int(sum(positive_groups['max']==0)),estimand='subgroup_mean_of_three_fixed_metrics'))
        if dataset=='ebtc_external':
            alt={'case_plus_part':meta.sample_id.map(norm_case_pt),'legacy_pid':meta.pid.fillna('unknown').astype(str)}
            for grouping,groups in alt.items():
                ww,uu=weights(groups,reps,rng);bs=[];ps=[]
                for seed in SEEDS:
                    f=predictions(dataset,'B3',seed);bs.append(metric_samples(y,f.prob_1.to_numpy(),ww));ps.append(metric_samples(y,f.prob_1.to_numpy(),np.ones((1,len(y))))[0])
                rows.extend(summarize(np.mean(bs,axis=0),np.mean(ps,axis=0),dataset=dataset,method='B3',seed='fixed_three_mean',grouping=grouping,estimand='grouping_stress_test_not_verified_independence'))
            for group in sorted(ids.unique()):
                keep=ids!=group;vs=[]
                for seed in SEEDS:
                    f=predictions(dataset,'B3',seed);vs.append(metric_samples(y[keep],f.loc[keep,'prob_1'].to_numpy(),np.ones((1,sum(keep))))[0])
                for j,name in enumerate(METRICS):loco.append(dict(removed_group=group,removed_images=int(sum(~keep)),metric=name,estimate=float(np.mean(vs,axis=0)[j])))
    print('Calibration',flush=True)
    for dataset in ['cystods_calib','cystods_internal','ebtc_external']:
        meta=metadata(dataset);y=meta.label_cancer_binary.to_numpy()
        groups=meta.pid if dataset.startswith('cystods') else meta.sample_id.map(norm_case)
        W,_=weights(groups,reps,rng);differences=[];pointd=[]
        for seed in SEEDS:
            values={};boots={}
            for raw in [True,False]:
                state='before' if raw else 'after';f=predictions(dataset,'B3',seed,raw);p=f.prob_1.to_numpy()
                assert np.array_equal(meta.sample_id,f.sample_id)
                joint=logistic_calibration(y,p);citl=logistic_calibration(y,p,offset=True)
                calrows.append(dict(dataset=dataset,seed=seed,state=state,joint_intercept=joint['intercept'],calibration_slope=joint['slope'],calibration_in_the_large=citl['intercept'],converged=joint['converged'] and citl['converged']))
                values[state]=metric_samples(y,p,np.ones((1,len(y))))[0];boots[state]=metric_samples(y,p,W)
                bins=np.minimum((p*10).astype(int),9)
                for bi in range(10):
                    mask=bins==bi
                    if not mask.any():continue
                    den=W@mask.astype(float);num=W@(mask*y)
                    rates=np.divide(num,den,out=np.full(len(W),np.nan),where=den>0);valid=rates[np.isfinite(rates)]
                    lo,hi=np.quantile(valid,[.025,.975])
                    reliability.append(dict(dataset=dataset,seed=seed,state=state,bin_index=bi,images=int(sum(mask)),groups=int(pd.Series(np.asarray(groups)[mask]).nunique()),mean_probability=float(np.mean(p[mask])),observed_fraction=float(np.mean(y[mask])),ci_low=float(lo),ci_high=float(hi),valid_replicates=len(valid)))
            diff=boots['after']-boots['before'];point=values['after']-values['before'];differences.append(diff);pointd.append(point)
            deltarows.extend(summarize(diff,point,dataset=dataset,seed=str(seed),contrast='after_minus_before'))
        deltarows.extend(summarize(np.mean(differences,axis=0),np.mean(pointd,axis=0),dataset=dataset,seed='fixed_three_mean',contrast='after_minus_before'))
    print('Matched components on validation selection set',flush=True)
    meta=metadata('cystods_validation');y=meta.label_cancer_binary.to_numpy();W,_=weights(meta.pid,reps,rng)
    component=[];component_raw=[]
    comparisons=[('augmentation_with_sampler',SEEDS),('sampling_with_basic',(42,))]
    for contrast,seeds in comparisons:
        differences=[];points=[]
        for seed in seeds:
            if contrast.startswith('augmentation'):
                a=ROOT/f'outputs/v2/runs/cdg_cysto_v2_stage06_augmentation_M1_swin_tiny_ce_class_balanced_sampler_basic_seed{seed}/predictions_val.csv'
                b=ROOT/f'outputs/v2/runs/cdg_cysto_v2_stage06_augmentation_M1_swin_tiny_ce_class_balanced_sampler_modality_aware_v1_seed{seed}/predictions_val.csv'
            else:
                a=ROOT/'outputs/v2/runs/cdg_cysto_v2_stage04_imbalance_M1_swin_tiny_ce_none_seed42/predictions_val.csv'
                b=ROOT/'outputs/v2/runs/cdg_cysto_v2_stage04_imbalance_M1_swin_tiny_ce_class_balanced_sampler_seed42/predictions_val.csv'
            # Legacy stage-04 run suffix is recovered from the saved leaderboard if needed.
            if not b.exists():
                candidates=list((ROOT/'outputs/v2/runs').glob(f'*M1_swin_tiny_ce_class_balanced_sampler_seed{seed}/predictions_val.csv'))
                if not candidates:raise FileNotFoundError(b)
                b=candidates[0]
            aa=read(a).sort_values('sample_id');bb=read(b).sort_values('sample_id')
            assert np.array_equal(aa.sample_id,meta.sample_id) and np.array_equal(bb.sample_id,meta.sample_id)
            ba=metric_samples(y,aa.prob_1.to_numpy(),W);bbt=metric_samples(y,bb.prob_1.to_numpy(),W)
            pa=metric_samples(y,aa.prob_1.to_numpy(),np.ones((1,len(y))))[0];pb=metric_samples(y,bb.prob_1.to_numpy(),np.ones((1,len(y))))[0]
            differences.append(bbt-ba);points.append(pb-pa)
            component.extend(summarize(bbt-ba,pb-pa,contrast=contrast,seed=str(seed),evaluation_set='validation_used_for_selection',n_models=1))
            for j,name in enumerate(METRICS):component_raw.append(dict(contrast=contrast,seed=seed,metric=name,before=float(pa[j]),after=float(pb[j])))
        component.extend(summarize(np.mean(differences,axis=0),np.mean(points,axis=0),contrast=contrast,seed='fixed_model_mean',evaluation_set='validation_used_for_selection',n_models=len(seeds)))
    save(rows,out,'seedwise_and_mean_cluster_intervals.csv');save(group_rows,out,'group_composition.csv');save(subrows,out,'modality_intervals_counts.csv')
    save(calrows,out,'calibration_intercept_slope.csv');save(deltarows,out,'calibration_paired_changes.csv');save(reliability,out,'cancer_reliability_bins.csv')
    save(loco,out,'leave_one_case_out.csv');save(contrasts,out,'paired_comparator_differences.csv');save(checks,out,'training_seed_variability.csv')
    save(component,out,'paired_component_validation.csv');save(component_raw,out,'component_validation_points.csv')
    save([dict(path=str(x.relative_to(ROOT)),sha256=digest(x),bytes=x.stat().st_size) for x in sorted(INPUTS)],out,'input_manifest.csv')
    info=dict(date='2026-10-18',bootstrap_replicates=reps,bootstrap_rng_seed=20261018,trained_seeds=list(SEEDS),elapsed_seconds=time.time()-start,
        estimand='Arithmetic mean of image-level metrics across three fixed trained models; conditional sampling uncertainty. Same cluster draws shared by seeds and paired comparators. Not ensemble probabilities; not uncertainty over future training.',
        missing_outcomes='AUROC/AUPRC and two-class macro-F1 undefined when a bootstrap sample contains one outcome class; discarded for that metric and reported with valid/invalid counts. Sensitivity/specificity/PPV/NPV require their own nonzero denominators. Brier/NLL/ECE remain defined in one-class samples.',
        grouping='Case and case-plus-part are filename-derived proxies. Legacy pid is an intentionally coarse stress test, not verified patient identities. Alternative smaller clusters are not proof of independence.',
        calibration='Joint logistic intercept and slope fit by unpenalized Newton MLE; probabilities clipped to [1e-6,1-1e-6] for logits. Calibration-in-the-large fits an intercept with logit(p) offset. Evaluative fits are diagnostics, not applied recalibration. Changes use common before/after cluster draws.',
        comparisons='Pointwise exploratory percentile intervals, no multiplicity-adjusted or p-value claims. Component comparisons are on the validation selection set; sampling only seed42. Do not infer confirmatory causal component benefit.',
        source_script_sha256=digest(Path(__file__)),outputs_ready=True)
    (out/'analysis_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reps',type=int,default=2000);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();run(args.reps,args.out)
