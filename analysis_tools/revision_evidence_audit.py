"""Independently recompute saved v2 evidence without training or changing source artifacts."""
from __future__ import annotations
import hashlib
import itertools
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'implementation'
OUT=Path(__file__).resolve().parents[1]/'artifact_audit'
BASE=ROOT/'outputs/v2/final_validation/frozen_baselines'
SEEDS=[42,2025,3407]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def binary_metrics(y,p,threshold=0.5):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=p>=threshold
    tp=int(sum(pred & (y==1))); tn=int(sum(~pred & (y==0)))
    fp=int(sum(pred & (y==0))); fn=int(sum(~pred & (y==1)))
    safe=lambda a,b: a/b if b else float('nan')
    sens=safe(tp,tp+fn); spec=safe(tn,tn+fp)
    ranks=pd.Series(p).rank(method='average').to_numpy()
    npos=int(y.sum()); nneg=len(y)-npos
    auc=safe(ranks[y==1].sum()-npos*(npos+1)/2,npos*nneg)
    order=np.argsort(-p,kind='stable'); sp=p[order]; sy=y[order]
    ends=np.r_[np.flatnonzero(np.diff(sp)),len(y)-1]
    cum=np.cumsum(sy)[ends]; recall=cum/npos; prec=cum/(ends+1)
    ap=float(np.sum(np.diff(np.r_[0,recall])*prec))
    conf=np.maximum(p,1-p); correct=((p>0.5).astype(int)==y)
    # Binary argmax chooses class zero at a probability tie.
    ece=0.0
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(conf>lo)&(conf<=hi)
        if mask.any(): ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    return dict(auroc=float(auc),auprc=ap,sensitivity=float(sens),specificity=float(spec),
        macro_f1=float((safe(2*tp,2*tp+fp+fn)+safe(2*tn,2*tn+fp+fn))/2),
        balanced_accuracy=float((sens+spec)/2),ppv=safe(tp,tp+fp),npv=safe(tn,tn+fn),
        ece=float(ece),brier=float(2*np.mean((p-y)**2)),
        nll=float(-np.mean(y*np.log(np.clip(p,1e-15,1))+(1-y)*np.log(np.clip(1-p,1e-15,1)))),
        TP=tp,TN=tn,FP=fp,FN=fn)

def main():
    OUT.mkdir(exist_ok=True)
    checks=[]
    def check(name,ok,detail): checks.append(dict(check=name,status='PASS' if ok else 'REVIEW',detail=detail))
    split_dir=ROOT/'outputs/v2/splits'
    manifest=json.loads((split_dir/'split_manifest.json').read_text(encoding='utf-8'))
    split_rows=[]; frames={}
    for item in manifest['splits']:
        path=ROOT/item['path']; f=pd.read_csv(path); frames[item['split']]=f
        actual=sha(path); check('split_sha256_'+item['split'],actual==item['sha256'],actual)
        pos=int(f.label_cancer_binary.sum())
        case=f.sample_id.str.extract(r'(?i)case_0*(\d+)',expand=False)
        split_rows.append(dict(split=item['split'],images=len(f),patients_pid=f.pid.nunique(),positive=pos,negative=len(f)-pos,
            normalized_cases=int(case.nunique()) if case.notna().any() else None,
            duplicate_sample_ids=int(f.sample_id.duplicated().sum()),duplicate_paths=int(f.image_path.duplicated().sum()),
            stored_md5_missing=int(f.image_md5.isna().sum()) if 'image_md5' in f else None,
            missing_image_files=sum(not (ROOT/str(p)).is_file() for p in f.image_path),
            modalities=json.dumps(f.modality.value_counts().to_dict(),ensure_ascii=False)))
    pair_rows=[]
    names=['cystods_train','cystods_val','cystods_calib','cystods_test_internal']
    for a,b in itertools.combinations(names,2):
        for column in ['pid','sample_id','image_path','image_md5']:
            overlap=set(frames[a][column].dropna().astype(str)) & set(frames[b][column].dropna().astype(str))
            pair_rows.append(dict(split_a=a,split_b=b,field=column,overlap_count=len(overlap)))
            check(f'overlap_{a}_{b}_{column}',not overlap,str(len(overlap)))
    cross=set(pd.concat([frames[n] for n in names]).image_md5)&set(frames['ebtc_external_all'].image_md5)
    check('cross_dataset_stored_md5_overlap',not cross,str(len(cross)))
    ebtc=frames['ebtc_external_all'].copy()
    ebtc['case']=ebtc.sample_id.map(lambda x:'case_'+str(int(re.search(r'(?i)case_0*(\d+)',x).group(1))).zfill(3) if re.search(r'(?i)case_0*(\d+)',x) else 'unknown')
    groups=ebtc.groupby('case').label_cancer_binary.max()
    check('ebtc_case_counts',len(groups)==22 and int(groups.sum())==16,dict(total=len(groups),positive=int(groups.sum()),negative=int(sum(groups==0)),unknown_images=int(sum(ebtc['case']=='unknown'))))
    pd.DataFrame(split_rows).to_csv(OUT/'split_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pair_rows).to_csv(OUT/'split_overlap_audit.csv',index=False,encoding='utf-8-sig')

    runlist=pd.read_csv(BASE/'selected_frozen_runs.csv'); records=[]; checkpoints=[]
    for row in runlist.itertuples(index=False):
        rd=ROOT/row.run_dir; rm=rd/'run_manifest.json'
        data=json.loads(rm.read_text(encoding='utf-8')) if rm.exists() else {}
        req=['run_manifest.json','config_resolved.yaml','checkpoint_best.pt','metrics_epoch.csv','logits_val.csv']
        checkpoints.append(dict(method_id=row.method_id,seed=row.seed,run_dir=str(rd),run_id=data.get('run_id'),git_commit=data.get('git_commit'),
            created_at_utc=data.get('created_at_utc'),missing_artifacts=';'.join(n for n in req if not (rd/n).is_file()),
            checkpoint_bytes=(rd/'checkpoint_best.pt').stat().st_size if (rd/'checkpoint_best.pt').exists() else 0))
        for dataset in ['cystods_internal','ebtc_external']:
            path=next(BASE.glob(f'{row.method_id}_*/seed_{row.seed}/{dataset}/predictions_default_0.5.csv'))
            pred=pd.read_csv(path); m=binary_metrics(pred.y_true,pred.prob_1)
            records.append(dict(method_id=row.method_id,seed=row.seed,dataset=dataset,**m))
            split=frames['cystods_test_internal' if dataset=='cystods_internal' else 'ebtc_external_all']
            joined=pred[['sample_id','y_true']].merge(split[['sample_id','label_cancer_binary']],on='sample_id',how='outer',indicator=True,validate='one_to_one')
            check(f'prediction_alignment_{row.method_id}_{row.seed}_{dataset}',bool((joined['_merge']=='both').all() and (joined.y_true==joined.label_cancer_binary).all()),len(joined))
    rec=pd.DataFrame(records); rec.to_csv(OUT/'recomputed_metrics_by_seed.csv',index=False,encoding='utf-8-sig')
    summary=rec.groupby(['method_id','dataset'])[['auroc','auprc','sensitivity','specificity','macro_f1','ece','brier','nll']].agg(['mean','std'])
    summary.columns=['_'.join(t) for t in summary.columns]
    summary.reset_index().to_csv(OUT/'recomputed_metrics_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checkpoints).to_csv(OUT/'run_artifact_audit.csv',index=False,encoding='utf-8-sig')
    target={'cystods_internal':{'auroc':.925,'auprc':.684,'sensitivity':.858,'specificity':.853,'macro_f1':.728},
            'ebtc_external':{'auroc':.914,'auprc':.949,'sensitivity':.774,'specificity':.887,'macro_f1':.809}}
    for dataset,ms in target.items():
        for metric,value in ms.items():
            actual=float(summary.loc[('B3',dataset),metric+'_mean'])
            check(f'manuscript_primary_{dataset}_{metric}',abs(actual-value)<0.00051,dict(recomputed=actual,submitted=value))
    cals=[]
    for seed in SEEDS:
        sd=BASE/f'B3_CDG_Cysto_v2/seed_{seed}'; temp=float((sd/'temperature.txt').read_text())
        for dataset in ['cystods_calib','cystods_internal','ebtc_external']:
            ps={}
            for state in ['raw','calibrated']:
                f=pd.read_csv(sd/dataset/f'predictions_{"raw_" if state=="raw" else ""}default_0.5.csv')
                ps[state]=f
                cals.append(dict(seed=seed,dataset=dataset,state=state,temperature=temp,**binary_metrics(f.y_true,f.prob_1)))
            logits=pd.read_csv(sd/dataset/'logits_raw.csv')
            arr=logits[['logit_0','logit_1']].to_numpy()/temp; arr-=arr.max(axis=1,keepdims=True)
            probs=np.exp(arr); probs/=probs.sum(axis=1,keepdims=True)
            check(f'temperature_transform_{seed}_{dataset}',bool(np.allclose(probs[:,1],ps['calibrated'].prob_1,atol=1e-7)),float(np.max(np.abs(probs[:,1]-ps['calibrated'].prob_1))))
            check(f'temperature_order_{seed}_{dataset}',bool(np.array_equal(np.argsort(ps['raw'].prob_1,kind='stable'),np.argsort(ps['calibrated'].prob_1,kind='stable'))),'stable probability sort comparison')
    pd.DataFrame(cals).to_csv(OUT/'calibration_recomputed.csv',index=False,encoding='utf-8-sig')
    threshold_rows=[]
    for seed in SEEDS:
        sd=BASE/f'B3_CDG_Cysto_v2/seed_{seed}'
        calib=pd.read_csv(sd/'cystods_calib/predictions_default_0.5.csv')
        grid=np.unique(np.r_[0.,.5,1.,calib.prob_1])
        scores=calib.prob_1.to_numpy(); y=calib.y_true.to_numpy()
        pred=scores[None,:]>=grid[:,None]
        sensitivity=pred[:,y==1].mean(axis=1); specificity=(~pred[:,y==0]).mean(axis=1)
        youden=float(grid[np.argmax(sensitivity+specificity-1)])
        feasible=np.flatnonzero(sensitivity>=.90)
        best=max(feasible,key=lambda i:(specificity[i],grid[i]))
        selections={'default_0.5':.5,'youden':youden,'sensitivity_90':float(grid[best])}
        saved=pd.read_csv(sd/'threshold_metrics.csv')
        for name,value in selections.items():
            old=float(saved.loc[saved.threshold_name==name,'threshold'].iloc[0])
            check(f'calib_threshold_{seed}_{name}',abs(value-old)<1e-12,dict(recomputed=value,saved=old))
            for dataset in ['cystods_internal','ebtc_external']:
                p=pd.read_csv(sd/dataset/f'predictions_{name}.csv')
                check(f'frozen_threshold_predictions_{seed}_{dataset}_{name}',bool(np.array_equal((p.prob_1>=value).astype(int),p.y_pred)),'saved decisions vs CystoDS-only selected threshold')
                threshold_rows.append(dict(seed=seed,dataset=dataset,threshold_name=name,threshold=value,**binary_metrics(p.y_true,p.prob_1,value)))
        p=pd.read_csv(sd/'ebtc_external/predictions_default_0.5.csv')
        official=p.merge(frames['ebtc_external_official_test'][['sample_id']],on='sample_id',validate='one_to_one')
        threshold_rows.append(dict(seed=seed,dataset='ebtc_official_test',threshold_name='default_0.5',threshold=.5,**binary_metrics(official.y_true,official.prob_1)))
    pd.DataFrame(threshold_rows).to_csv(OUT/'thresholds_official_test_recomputed.csv',index=False,encoding='utf-8-sig')
    boots=pd.read_csv(ROOT/'outputs/v2/patient_level_corrected/patient_clustered_bootstrap_replicates.csv')
    bsummary=pd.read_csv(ROOT/'outputs/v2/patient_level_corrected/patient_clustered_bootstrap_summary.csv')
    ci_rows=[]
    for row in bsummary.itertuples(index=False):
        if row.threshold_name!='default_0.5' or row.metric not in ['AUROC','AUPRC','sensitivity','specificity','macro_f1']:continue
        vals=boots[(boots.dataset==row.dataset)&(boots.threshold_name==row.threshold_name)&(boots.metric==row.metric)].value.dropna()
        lo,hi=np.quantile(vals,[.025,.975])
        ci_rows.append(dict(dataset=row.dataset,metric=row.metric,valid_replicates=len(vals),low=lo,high=hi))
        check(f'ci_percentiles_{row.dataset}_{row.metric}',bool(abs(lo-row.ci_low)<1e-10 and abs(hi-row.ci_high)<1e-10),dict(valid_replicates=len(vals),ci_low=lo,ci_high=hi))
    pd.DataFrame(ci_rows).to_csv(OUT/'ci_percentile_recheck.csv',index=False,encoding='utf-8-sig')
    sources=[]
    for sub in ['src','scripts','configs']:
        for path in sorted((ROOT/sub).rglob('*')):
            if path.is_file() and path.suffix in ['.py','.yaml','.yml','.toml','.json']:
                sources.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    for path in [ROOT/'requirements.txt',ROOT/'pyproject.toml',ROOT/'README.md']:
        sources.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    pd.DataFrame(sources).to_csv(OUT/'code_config_manifest.csv',index=False,encoding='utf-8-sig')
    (OUT/'evidence_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(checks=len(checks),pass_count=sum(c['status']=='PASS' for c in checks),review=[c for c in checks if c['status']!='PASS'],
        splits=split_rows,run_count=len(checkpoints),bootstrap_rows=len(boots),code_config_files=len(sources)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
