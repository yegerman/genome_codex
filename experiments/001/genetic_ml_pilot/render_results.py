from pathlib import Path
import json,matplotlib.pyplot as plt,numpy as np
P=Path(__file__).parent
r=json.load((P/'grouped_results.json').open())
name_map={'majority':'Majority baseline','GC_only':'GC baseline','DNA_kmer_3_4':'DNA 3-4mers','DNA_position_onehot':'DNA positional logistic','DNA_position_SVM':'DNA positional SVM'}
a=r['aggregate_out_of_fold'];labels=[name_map[x['model']] for x in a];vals=[x['balanced_accuracy']*100 for x in a];lower=[(x['balanced_accuracy']-x['group_bootstrap_bacc_ci_95'][0])*100 for x in a];upper=[(x['group_bootstrap_bacc_ci_95'][1]-x['balanced_accuracy'])*100 for x in a]
fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(len(vals));ax.bar(x,vals);ax.errorbar(x,vals,yerr=[lower,upper],fmt='none',capsize=5)
ax.set_ylim(0,107);ax.set_ylabel('Balanced accuracy (%)');ax.set_xticks(x,labels,rotation=20,ha='right');ax.set_title('Group-disjoint, nested cross-validation: E. coli promoter classification')
for k,y in enumerate(vals):ax.text(k,y+4,f'{y:.1f}%',ha='center',fontsize=9)
fig.tight_layout();fig.savefig(P/'results_chart.png',dpi=150);plt.close(fig)

import csv
rows=list(csv.reader((P/'promoters_source.csv').open()));s=[x[2].upper() for x in rows];y=np.array([int(x[0]=='+') for x in rows])
weights=np.load(P/'model_coefficients.npz')['coef'].reshape(57,4)
imp=weights.max(axis=1)-weights.min(axis=1)
fig,ax=plt.subplots(figsize=(11,3.3))
ax.bar(np.arange(-50,7),imp)
ax.axvspan(-36,-30,alpha=.12);ax.axvspan(-12,-7,alpha=.12)
ax.set_xlabel('DNA position relative to start site (+1)');ax.set_ylabel('Weight range (exploratory)');ax.set_title('Full-data model interpretation: position-wise feature importance')
fig.tight_layout();fig.savefig(P/'learned_positions_chart.png',dpi=150);plt.close(fig)
