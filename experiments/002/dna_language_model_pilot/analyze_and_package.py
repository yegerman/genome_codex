import csv,json,hashlib,zipfile
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent
r=json.load(open(p/'results.json'))
rows=list(csv.DictReader(open(p/'out_of_fold_predictions.csv')))
y=np.array([int(x['label']) for x in rows]);a=np.array([int(x['mlm_flatten prediction']) for x in rows]);b=np.array([int(x['random_flatten prediction']) for x in rows]);g=np.array([int(x['group']) for x in rows]);
print('trained error IDs',[x['id'] for x,k,v in zip(rows,a,y) if k!=v]);print('random error IDs',[x['id'] for x,k,v in zip(rows,b,y) if k!=v]);
print('discordant trained correct/random wrong',int(np.sum((a==y)&(b!=y))),'trained wrong/random correct',int(np.sum((a!=y)&(b==y))))
ug=np.unique(g);lookup={gg:np.flatnonzero(g==gg) for gg in ug};rng=np.random.default_rng(454)
boots=[]
for k in range(5000):
 ids=np.concatenate([lookup[gg] for gg in rng.choice(ug,len(ug),replace=True)])
 if len(set(y[ids]))<2:continue
 boots.append(balanced_accuracy_score(y[ids],a[ids])-balanced_accuracy_score(y[ids],b[ids]))
print('95% group boot delta MLM minus untrained',np.percentile(boots,[2.5,97.5]));
perfold=[(x['heldout_masked_acc'],x['heldout_posfreq_acc'],x['heldout_unigram_acc']) for x in r['folds']]
print('masked accuracy means',np.mean(perfold,axis=0));
r['analysis']={'paired_trained_correct_random_wrong':int(np.sum((a==y)&(b!=y))),'paired_trained_wrong_random_correct':int(np.sum((a!=y)&(b==y))),'paired_balanced_accuracy_diff_MLM_minus_random':float(balanced_accuracy_score(y,a)-balanced_accuracy_score(y,b)),'paired_group_bootstrap_95_ci_diff':[float(x) for x in np.percentile(boots,[2.5,97.5])],'heldout_masked_acc_mean':float(np.mean([v[0] for v in perfold])),'heldout_position_frequency_baseline_acc_mean':float(np.mean([v[1] for v in perfold])),'heldout_unigram_baseline_acc_mean':float(np.mean([v[2] for v in perfold]))}
(p/'results.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
fig,ax=plt.subplots(figsize=(8,4));names=['Previous position\nlogistic','Random transformer\n+ probe','Masked DNA transformer\n+ probe','DNA MLM pooled\n+ probe'];scores=[94.3,94.3,93.4,67.0];bars=ax.bar(names,scores);ax.set_ylim(0,105);ax.set_ylabel('Out-of-fold accuracy (%)');ax.set_title('E. coli promoter recognition (106 sequences)');ax.axhline(50,linewidth=.8,linestyle='--');
for bar,sc in zip(bars,scores):ax.text(bar.get_x()+bar.get_width()/2,sc+1.4,f'{sc:.1f}%',ha='center',fontsize=10)
fig.tight_layout();fig.savefig(p/'comparison.png',dpi=140);plt.close(fig)
text='''# DNA Language Model — experiment 002 (20 Nov 2026)

## Question
Does **self-supervised DNA language pretraining** improve the ability of a sequence encoder to classify bacterial promoter sequences compared with the exact same encoder using random, untrained weights?

## The actual experiment
Data: 106 genuine UCI E. coli promoter/non-promoter examples, length 57, 53 each. Source: https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences . Local source input re-used from experiment 001 and originally transcribed from the public official UCI text; raw source checksum was not independently verified.

Architecture: tiny *masked language model*, Transformer encoder with two layers, hidden width 32, four attention heads, trained from **random initialization**. Each outer training fold has only 79–80 sequences. In each fold, 180 optimizer steps, 20% of tokens replaced with a mask, and a cross-entropy loss on the missing A/C/G/T bases. DNA pretraining uses **training sequences only**, ignoring promoter labels; the test sequences are withheld throughout pretraining and probe fitting. Random-initialized encoder of identical architecture and same initialization is the control. A frozen linear logistic classification probe sees only fold-train promoter labels. There is no model-download or pretrained genomic foundation model here.

Leakage control: 4-fold StratifiedGroupKFold, connected-component groups built from shared exact 25-mers; 69 groups. Hyperparameter C=0.03 was fixed for all probes. The previous (94.3%) supervised one-hot baseline tuned its C by an inner CV and is therefore contextual, not a perfectly fair model-selection comparison.

## Measured outcomes
| Model | Correct / 106 | Accuracy | ROC AUC |
|---|---:|---:|---:|
| Earlier one-hot position logistic baseline (prior experiment) | 100 | 94.34% | 0.984 |
| Transformer encoder *without language training* + frozen linear probe | 100 | 94.34% | 0.9843 |
| Transformer encoder *with masked DNA training* + frozen linear probe | 99 | 93.40% | 0.9847 |
| Transformer with masked DNA training; mean-pooled embedding + probe | 71 | 66.98% | 0.7654 |
| Untrained Transformer; mean-pooled embedding + probe | 72 | 67.92% | 0.7398 |

The masked nucleotide reconstruction task on held-out fold sequences: 27.95% correct, vs 30.84% for a training-set position-frequency baseline and 28.42% for a global-base-frequency baseline (averaged over four folds). Random choice of A/C/G/T would be 25%.

Masked LM minus random encoder: -0.94 percentage points in classification accuracy, with a paired, group-bootstrap 95% interval recorded in results.json. This is not evidence for an advantage from pretraining.

## Interpretation
The decoder/classifier can detect a signal in these examples; the language pretraining **did not add measurable value**. A supervised linear model works well because promoter motifs are position-dependent and the set is very small. This is **not** a failure of genomic foundation models or of the larger research hypothesis. It shows why a genuinely pretrained genomic language model and a larger sequence-to-function measurement dataset are necessary for testing the thesis.

The next faithful experiment uses a downloadable genomic pretrained model (e.g. Nucleotide Transformer or DNABERT-2) on a suitable MPRA dataset, with an independent holdout by sequence family, **and tests whether it beats simple motif/CNN baselines**, plus biological interpretation. Published research suggests benefits in controlled promoter tasks but no guaranteed win over simpler models.

## Reproduce
Python 3.11+ with torch, scikit-learn, numpy, matplotlib. Place the original `promoters_source.csv` at the path shown in `train_dna_mlm.py` or change `DATA`. Run `python train_dna_mlm.py` followed by `python analyze_and_package.py`. CPU runtime about 25 seconds for the four training folds on this environment.

## Scientific boundaries
This is not a large language model (LLM) pretrained on genomes. It is a small Transformer-based *language-modeling experiment*. No human DNA prediction, no expression measurements, no genome design, no disease inference, no repair instructions. The only trained predictors classify E. coli promoter annotations in a small historical dataset.
'''
(p/'README.md').write_text(text)
with zipfile.ZipFile(str(p.parent/'dna_language_model_experiment_002_rebuilt.zip'),'w',zipfile.ZIP_DEFLATED) as z:
 for fname in ['train_dna_mlm.py','analyze_and_package.py','results.json','out_of_fold_predictions.csv','example_trainfold0_masked_lm.pt','comparison.png','README.md']:
  z.write(p/fname,arcname='dna_language_model_pilot/'+fname)
 z.write(str(p/'promoters_source.csv'),arcname='dna_language_model_pilot/promoters_source.csv')
print('Saved',p/'comparison.png',str(p.parent/'dna_language_model_experiment_002_rebuilt.zip'))
