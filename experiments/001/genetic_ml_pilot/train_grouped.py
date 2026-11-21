#!/usr/bin/env python3
"""Group-disjoint ML experiment on real UCI E.coli promoter annotations.
The 25mer connected components prevent identical long sequence fragments
from being placed in training and evaluation splits simultaneously.
"""
from collections import defaultdict, Counter
from pathlib import Path
import csv, hashlib, json
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score,accuracy_score,roc_auc_score,confusion_matrix, f1_score
from joblib import dump
P=Path(__file__).parent
src=P/'promoters_source.csv'; rows=list(csv.reader(src.open()))
X=np.array([r[2].upper() for r in rows]);y=np.array([int(r[0]=='+') for r in rows]);ids=np.array([r[1] for r in rows]);n=len(X)
assert n==106 and (y.sum(),len(y)-y.sum())==(53,53) and all(len(s)==57 for s in X)
class GC(BaseEstimator,TransformerMixin):
 def fit(self,X,y=None):return self
 def transform(self,X):return np.array([[sum(b in 'GC' for b in s)/57] for s in X])
class OneHotPosition(BaseEstimator,TransformerMixin):
 def fit(self,X,y=None):return self
 def transform(self,X):
  o=np.zeros((len(X),57,4),dtype='float32');bases={b:i for i,b in enumerate('ACGT')}
  for row,s in enumerate(X):
   for j,b in enumerate(s):o[row,j,bases[b]]=1
  return o.reshape(len(X),-1)

# Graph connected components: if any contiguous exact 25bp fragment is shared,
# both records are assigned to same group (including transitive overlaps).
parent=list(range(n))
def root(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(a,b):
 a,b=root(a),root(b)
 if a!=b:parent[a]=b
kmer_to_rows=defaultdict(set)
for i,s in enumerate(X):
 for j in range(len(s)-25+1):kmer_to_rows[s[j:j+25]].add(i)
for member_ids in kmer_to_rows.values():
 members=list(member_ids)
 for j in members[1:]:union(members[0],j)
groups=np.array([root(i) for i in range(n)])
assert len(set(groups))>60
assert not any(len(set(y[groups==g]))>1 for g in set(groups))

models={
 'majority':(DummyClassifier(strategy='prior'),{}),
 'GC_only':(Pipeline([('gc',GC()),('z',StandardScaler()),('m',LogisticRegression(max_iter=3000))]),{'m__C':[0.01,0.1,1,10]}),
 'DNA_kmer_3_4':(Pipeline([('k',CountVectorizer(analyzer='char',ngram_range=(3,4),lowercase=False)),('m',LogisticRegression(solver='liblinear',max_iter=3000))]),{'m__C':[.01,.1,1,10]}),
 'DNA_position_onehot':(Pipeline([('o',OneHotPosition()),('m',LogisticRegression(solver='liblinear',max_iter=3000))]),{'m__C':[.01,.1,1,10]}),
 'DNA_position_SVM':(Pipeline([('o',OneHotPosition()),('m',SVC())]),{'m__C':[.1,1,10],'m__gamma':['scale',.01,.1]})
}
outer=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2026)
folds=list(outer.split(X,y,groups))
print('outer fold sizes:', [(len(tr),len(te),Counter(y[te])) for tr,te in folds]);
# For stable cross validated estimates fit inner grouped CV separately per outer fold.
records=[];all_oof={name:np.full(n,np.nan) for name in models};scores_oof={name:np.full(n,np.nan) for name in models};predictions=[];first_models={}
for fold,(train_idx,test_idx) in enumerate(folds):
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    # Strict audit: zero identical 25-mers crossing train-test.
    trainkmers=set(k for s in X[train_idx] for k in (s[j:j+25] for j in range(33)))
    assert all(not any(s[j:j+25] in trainkmers for j in range(33)) for s in X[test_idx])
    inn=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42+fold)
    for name,(model,params) in models.items():
      search=GridSearchCV(model,params,scoring='balanced_accuracy',cv=inn,n_jobs=1) if params else None
      fit=search.fit(X[train_idx],y[train_idx],groups=groups[train_idx]).best_estimator_ if search else model.fit(X[train_idx],y[train_idx])
      pred=fit.predict(X[test_idx])
      if hasattr(fit,'decision_function'): scoring=fit.decision_function(X[test_idx])
      elif hasattr(fit,'predict_proba'):scoring=fit.predict_proba(X[test_idx])[:,1]
      else:scoring=pred.astype(float)
      rec={'fold':fold,'model':name,'n_train':len(train_idx),'n_test':len(test_idx),'test_positive':int(y[test_idx].sum()),'test_negative':int(len(test_idx)-y[test_idx].sum()),'test_accuracy':float(accuracy_score(y[test_idx],pred)),'test_balanced_accuracy':float(balanced_accuracy_score(y[test_idx],pred)),'test_auc':float(roc_auc_score(y[test_idx],scoring)) if len(set(y[test_idx]))==2 else None,'confusion_matrix':confusion_matrix(y[test_idx],pred,labels=[0,1]).tolist(),'best_params':search.best_params_ if search else {},'cv_balanced_accuracy':float(search.best_score_) if search else None}
      records.append(rec);all_oof[name][test_idx]=pred;scores_oof[name][test_idx]=scoring
      if fold==0:first_models[name]=fit
      print('fold',fold,name,'n',len(test_idx),'bacc',round(rec['test_balanced_accuracy'],3),'acc',round(rec['test_accuracy'],3),'conf',rec['confusion_matrix'])

# Aggregate OOF and bootstrap by sequence group, not by sample: conservative uncertainty.
rng=np.random.default_rng(318)
unique_groups=list(dict.fromkeys(groups))
groups_to_idx={g:np.where(groups==g)[0] for g in unique_groups}
aggregate=[]
for name in models:
 pred=all_oof[name].astype(int);sc=scores_oof[name]
 assert np.isfinite(sc).all()
 agg={'model':name,'n':n,'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'roc_auc':float(roc_auc_score(y,sc)),'confusion_matrix':confusion_matrix(y,pred).tolist()}
 boots=[]
 for _ in range(2500):
  chosen=rng.choice(unique_groups,size=len(unique_groups),replace=True)
  sample=np.concatenate([groups_to_idx[g] for g in chosen]);
  if len(set(y[sample]))<2:continue
  boots.append(balanced_accuracy_score(y[sample],pred[sample]))
 agg['group_bootstrap_bacc_ci_95']=[float(z) for z in np.quantile(boots,[.025,.975])]
 aggregate.append(agg)
 print('OVERALL',name,agg)

r={
 'experiment':'group_disjoint_cross_validation','dataset':'UCI Molecular Biology Promoter Gene Sequences (1987) 106 examples','source_url':'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data',
 'source_sha256_local_transcription':hashlib.sha256(src.read_bytes()).hexdigest(),
 'record_source':'Hand transcribed from official public UCI browser-visible 106-line text; whitespace normalized, validated 106 rows / 57 ACGT bases / balanced labels; independent byte-level checksum against raw UCI not available because host has no internet',
 'n_examples':n,'n_promoters':int(sum(y)),'n_nonpromoters':int(n-sum(y)),'n_groups_25mer':len(unique_groups),
 'leakage_rule':'No pair of sequences across train/test may share an exact 25bp contiguous DNA fragment, including transitive sharing (connected-component grouping).',
 'method':'4-fold StratifiedGroupKFold outer, 3-fold StratifiedGroupKFold inner hyperparameter search, raw DNA only','folds':records,'aggregate_out_of_fold':aggregate,
 'limitations':['Promoter yes/no data in E.coli, not MPRA quantitative expression in human cells.','Negative samples derive from windows of limited genomic contexts. 25mer grouping removes a specific direct sequence overlap leakage risk, but does not guarantee independence of biological source.','106 examples and four outer folds are insufficient for claims about general genomic language or design.','No DNA foundation model or inverse-design model was trained.']
}
(P/'grouped_results.json').write_text(json.dumps(r,indent=2))
with (P/'grouped_predictions.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['id','label','group','fold']+[f'prediction_{name}' for name in models]);
 fold_of=np.zeros(n,dtype=int)
 for k,(_,ti) in enumerate(folds):fold_of[ti]=k
 for i in range(n):w.writerow([ids[i],y[i],groups[i],fold_of[i]]+[int(all_oof[name][i]) for name in models]);
# Persist one fold's pretrained predictor (for demonstrating inference on E.coli 57nt).
dump(first_models['DNA_kmer_3_4'],P/'trained_kmer_3_4_fold0.joblib')
print('WROTE RESULTS')
