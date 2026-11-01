"""Reproduce Table R11 from saved raw cancer probabilities; no model training."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import revision_reanalysis as ra
def run(source,output):
 data=pd.read_csv(source);rng=np.random.default_rng(20261101);rows=[]
 for dataset in ['cystods_internal','ebtc_external']:
  sub=data[data.dataset==dataset];ref=sub[(sub.variant=='basic')&(sub.seed==42)].sort_values('sample_id')
  W,_=ra.weights(ref.group_id,2000,rng);y=ref.y_true.to_numpy()
  for contrast,plus,minus,seeds in [('sampling','basic','unsampled',[42]),('augmentation','augmented','basic',[42,2025,3407])]:
   bs=[];ps=[]
   for seed in seeds:
    b=[];p=[]
    for variant in [plus,minus]:
     s=sub[(sub.variant==variant)&(sub.seed==seed)].sort_values('sample_id');assert np.array_equal(ref.sample_id,s.sample_id) and np.array_equal(y,s.y_true) and np.array_equal(ref.group_id,s.group_id)
     b.append(ra.metric_samples(y,s.prob_1.to_numpy(),W));p.append(ra.metric_samples(y,s.prob_1.to_numpy(),np.ones((1,len(y))))[0])
    bs.append(b[0]-b[1]);ps.append(p[0]-p[1])
   rows.extend(ra.summarize(np.mean(bs,axis=0),np.mean(ps,axis=0),dataset=dataset,contrast=contrast,n_models=len(seeds)))
 result=pd.DataFrame(rows);result.to_csv(output,index=False);return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'revision_analysis/heldout_components/heldout_component_predictions.csv');parser.add_argument('--out',type=Path,default=Path('paired_heldout_components_reproduced.csv'));a=parser.parse_args();run(a.input,a.out)
