import csv,json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict
P=Path(__file__).parent
r=list(csv.reader(open(P/'promoters_source.csv')))
seq=[x[2].upper() for x in r]
y=np.array([int(x[0]=='+') for x in r]);n=len(seq)
def onehot(ss):
 o=np.zeros((len(ss),57,4))
 for i,s in enumerate(ss):
  for j,x in enumerate(s):o[i,j,'ACGT'.index(x)]=1
 return o
X=onehot(seq)
# read group from previous output for consistency
a=[]
with (P/'grouped_predictions.csv').open() as f:
 for row in list(csv.DictReader(f)):a.append(int(row['group']))
groups=np.array(a)
folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2026).split(X,y,groups))
results=[]
for start,end,title in [(0,19,'-50_to_-32'),(12,26,'-38_to_-25'),(25,37,'-25_to_-14'),(35,49,'-15_to_-2'),(0,29,'-50_to_-21'),(29,57,'-21_to_+6'),(0,57,'all_positions')]:
 preds=np.zeros(n,int)
 for tr,te in folds:
  clf=LogisticRegression(max_iter=3000,C=.1,solver='liblinear')
  clf.fit(X[tr,start:end,:].reshape(len(tr),-1),y[tr]);preds[te]=clf.predict(X[te,start:end,:].reshape(len(te),-1))
 res={'region':title,'accuracy':round(float(accuracy_score(y,preds)),4),'balanced_accuracy':round(float(balanced_accuracy_score(y,preds)),4),'n_correct':int(sum(y==preds))};results.append(res)
 print(res)
# Interpretation model trained on all data for explainability, not evaluation.
clf=LogisticRegression(C=.1,max_iter=3000,solver='liblinear').fit(X.reshape(n,-1),y)
w=clf.coef_.reshape(57,4)
rank=np.argsort(w.max(axis=1)-w.min(axis=1))[::-1]
print('TOP SITE range+base',[(int(i-50),round(float(w[i].max()-w[i].min()),3),'ACGT'[w[i].argmax()],'ACGT'[w[i].argmin()]) for i in rank[:18]])
print('class-wise bases at -12 to -7:')
for pos in range(-12,-6):
 for yy in [1,0]:
  letters=[seq[i][pos+50] for i in range(len(y)) if y[i]==yy]
  print(pos,yy,{b:round(letters.count(b)/len(letters),2) for b in 'ACGT'})

np.savez(P/'model_coefficients.npz',coef=clf.coef_.astype('float32'),intercept=clf.intercept_.astype('float32'),alphabet=np.array(list('ACGT')),input_length=np.array([57]))
(P/'region_ablation.json').write_text(json.dumps({'method':'Fixed C=.1 logistic regression on position-aware one-hot; grouped 4-fold CV; feature subsets by positions relative to promoter +1; descriptive not causal.', 'results':results},indent=2))
