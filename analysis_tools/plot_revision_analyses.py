"""Python-only publication plots for positive-class calibration and seed uncertainty."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'revision_analysis';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
names={'cystods_calib':'CystoDS calibration','cystods_internal':'CystoDS internal test','ebtc_external':'EBTC public external'}
colors={'before':'#747C88','after':'#007C83'}
f=pd.read_csv(ROOT/'cancer_reliability_bins.csv')
fig,axes=plt.subplots(3,3,figsize=(7.2,7.5),sharex=True,sharey=True,layout='constrained')
for row,dataset in enumerate(names):
    for col,seed in enumerate([42,2025,3407]):
        ax=axes[row,col];ax.plot([0,1],[0,1],ls=':',color='#888',lw=.7)
        for state,marker in [('before','s'),('after','o')]:
            g=f[(f.dataset==dataset)&(f.seed==seed)&(f.state==state)].sort_values('mean_probability')
            ax.plot(g.mean_probability,g.observed_fraction,marker=marker,ms=3,lw=.8,color=colors[state],label=state.capitalize())
            if state=='after':ax.vlines(g.mean_probability,g.ci_low,g.ci_high,color=colors[state],lw=.65,alpha=.5)
        ax.set(xlim=(0,1),ylim=(0,1),title=f'{names[dataset]}\nseed {seed}',xticks=[0,.5,1],yticks=[0,.5,1])
        if col==0:ax.set_ylabel('Observed cancer fraction')
        if row==2:ax.set_xlabel('Mean predicted cancer probability')
        ax.text(-.13,1.15,chr(97+row*3+col),transform=ax.transAxes,weight='bold',fontsize=9)
axes[0,0].legend(loc='upper left',fontsize=6)
for ext in ['svg','pdf','png','tiff']:fig.savefig(OUT/f'Fig_R1_cancer_class_reliability.{ext}',dpi=600)
plt.close(fig)

f=pd.read_csv(ROOT/'seedwise_and_mean_cluster_intervals.csv')
f=f[(f.dataset=='ebtc_external')&(f.method=='B3')&(f.grouping=='case_proxy')]
fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
for k,metric in enumerate(['auroc','sensitivity']):
    ax=axes[k]
    for j,seed in enumerate(['42','2025','3407','fixed_three_mean']):
        g=f[(f.seed.astype(str)==seed)&(f.metric==metric)].iloc[0]
        color='#007C83' if j==3 else '#747C88'
        ax.hlines(j,g.ci_low,g.ci_high,color=color,lw=1.3);ax.plot(g.estimate,j,'o',ms=5,color=color)
    ax.set(yticks=range(4),yticklabels=['Seed 42','Seed 2025','Seed 3407','Mean of fixed models'],xlim=(.45,1),xlabel=metric.upper() if metric=='auroc' else 'Cancer sensitivity',title='EBTC external evaluation')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.text(-.1,1.07,chr(97+k),weight='bold',fontsize=9,transform=ax.transAxes)
for ext in ['svg','pdf','png','tiff']:fig.savefig(OUT/f'Fig_R2_seedwise_cluster_intervals.{ext}',dpi=600)
plt.close(fig)
print('Exported Fig R1 and R2 as SVG, PDF, PNG, TIFF with traceable CSV inputs.')
