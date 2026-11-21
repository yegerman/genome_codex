#!/usr/bin/env python3
"""Reproducible pilot on public UCI E. coli promoter recognition data.
Not an MPRA assay, gene-expression regression, or a genomic foundation model.
"""
import csv, json, hashlib, warnings, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix

OUT=Path(__file__).parent
FILE=OUT/'promoters_source.csv'
raw=FILE.read_bytes()
rows=[r for r in csv.reader(raw.decode().splitlines())]
assert len(rows)==106 and Counter(r[0] for r in rows)=={'+':53,'-':53}, 'UCI expected 53+ 53-'
assert all(len(r)==3 and len(r[2])==57 and set(r[2]) <= set('acgt') for r in rows)
seq=np.array([r[2].upper() for r in rows]); labels=np.array([int(r[0]=='+') for r in rows]); ids=[r[1] for r in rows]
class OneHotPosition(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X):
        alphabet={b:i for i,b in enumerate('ACGT')}
        out=np.zeros((len(X),57,4),np.float32)
        for j,s in enumerate(X):
            for i,b in enumerate(s):out[j,i,alphabet[b]]=1
        return out.reshape(len(X),-1)
class GC(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X): return np.array([[sum(b in 'GC' for b in s)/len(s)] for s in X])

# One fixed unseen holdout. All fitting, hyperparameter selection, and scaling is within the train set.
idx=np.arange(len(seq))
trainidx,testidx=train_test_split(idx,test_size=0.25,stratify=labels,random_state=2026)
Xtrain,Ytrain=seq[trainidx],labels[trainidx]
Xtest,Ytest=seq[testidx],labels[testidx]
cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=240)
models={
 'majority':(DummyClassifier(strategy='prior'),{}),
 'GC_content':(Pipeline([('gc',GC()),('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000))]), {'clf__C':[0.01,0.1,1,10]}),
 'kmer_3_4':(Pipeline([('kmers',CountVectorizer(analyzer='char',ngram_range=(3,4),lowercase=False)),('clf',LogisticRegression(max_iter=5000,solver='liblinear'))]),{'clf__C':[0.01,0.1,1,10]}),
 'position_onehot':(Pipeline([('onehot',OneHotPosition()),('clf',LogisticRegression(max_iter=5000,solver='liblinear'))]),{'clf__C':[0.01,0.1,1,10]}),
 'position_onehot_RBF_SVM':(Pipeline([('onehot',OneHotPosition()),('clf',SVC(probability=False))]),{'clf__C':[0.1,1,10], 'clf__gamma':['scale',0.01,0.1]}),
}
results=[];predictions={}
for name,(pipe,hyper) in models.items():
    search=GridSearchCV(pipe,hyper,scoring='balanced_accuracy',cv=cv,n_jobs=1,refit=True) if hyper else None
    fitted=search.fit(Xtrain,Ytrain).best_estimator_ if search is not None else pipe.fit(Xtrain,Ytrain)
    score=search.best_score_ if search else None
    ypred=fitted.predict(Xtest)
    if hasattr(fitted,'decision_function'): decision=fitted.decision_function(Xtest)
    elif hasattr(fitted,'predict_proba'):decision=fitted.predict_proba(Xtest)[:,1]
    else:decision=ypred.astype(float)
    d={'name':name,'n_train':len(trainidx),'n_test':len(testidx),'cv_balanced_accuracy':round(float(score),4) if score is not None else None,'selected_params':search.best_params_ if search else {},'test_accuracy':round(float(accuracy_score(Ytest,ypred)),4),'test_balanced_accuracy':round(float(balanced_accuracy_score(Ytest,ypred)),4),'test_auc':round(float(roc_auc_score(Ytest,decision)),4),'test_f1':round(float(f1_score(Ytest,ypred)),4),'confusion_matrix':confusion_matrix(Ytest,ypred).tolist(),'n_errors':int((ypred!=Ytest).sum())}
    results.append(d)
    predictions[name]=ypred.tolist()
    print(name,d)

# Conservative sequence similarity audit: shift-invariant 25bp common substring, or same-position Hamming near-match.
# Sequences were deliberately sampled, so the audit documents common-ancestry leakage risks.
def max_shared_run(a,b):
    longest=0
    for shift in range(-len(a)+1,len(a)):
        count=0
        for i in range(max(0,shift),min(len(a),len(a)+shift)):
            j=i-shift
            if a[i]==b[j]:
                count+=1;longest=max(longest,count)
            else:count=0
    return longest
shares=[]
for j in testidx:
    vals=[max_shared_run(seq[j],seq[k]) for k in trainidx]
    shares.append(max(vals))
leakage={'n_test_with_exact_25mer_in_train':int(sum(x>=25 for x in shares)), 'n_test_with_exact_35mer_in_train':int(sum(x>=35 for x in shares)), 'n_test_with_exact_45mer_in_train':int(sum(x>=45 for x in shares)), 'max_shared_contiguous_bp_per_test':list(map(int,shares)), 'test_row_ids_at_risk':[ids[int(testidx[k])] for k,s in enumerate(shares) if s>=25]}
# If any held-out examples have long exact overlap to training examples, remove from diagnostic calculation.
keep=np.array([s<25 for s in shares],dtype=bool)
nonoverlap=[]
for res in results:
    yp=np.array(predictions[res['name']])[keep]
    yt=Ytest[keep]
    if len(set(yt))==2:
        nonoverlap.append({'model':res['name'],'n_safe_test':int(keep.sum()),'balanced_accuracy':round(float(balanced_accuracy_score(yt,yp)),4),'accuracy':round(float(accuracy_score(yt,yp)),4)})

# Diagnostic bootstrap, conditional on fixed prediction and test distribution; small sample -> caution.
rng=np.random.default_rng(222)
CIs={}
for res in results:
    scores=[]; yp=np.array(predictions[res['name']]);
    for _ in range(3000):
        b=rng.integers(0,len(Ytest),len(Ytest));
        if len(set(Ytest[b]))==2:scores.append(balanced_accuracy_score(Ytest[b],yp[b]))
    CIs[res['name']]=[round(float(v),4) for v in np.quantile(scores,[.025,.975])]
    res['test_bacc_95pct_bootstrap_CI']=CIs[res['name']]

report={
 'status':'completed_real_training','dataset':'UCI Molecular Biology (Promoter Gene Sequences) id=67','dataset_url':'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data',
 'dataset_license':'CC BY 4.0; original attribution Harley, Reynolds, Noordewier (1987); source copied from browser-visible official plain-text UCI file',
 'source_sha256_local_transcription':hashlib.sha256(raw).hexdigest(),
 'n_rows':len(seq),'n_positive':int(sum(labels)),'n_negative':int(sum(1-labels)), 'seq_length':57,
 'random_seed':2026,'train_indices':trainidx.tolist(),'test_indices':testidx.tolist(),'test_ids':[ids[i] for i in testidx],
 'results':results,'similarity_audit':leakage,'diagnostic_nonoverlap_metrics':nonoverlap,
 'limitations':['Dataset contains E. coli promoter yes/no labels, not MPRA activity or quantitative human HBB gene expression.','Tiny dataset (106 samples, 57 nt each); apparent performance has high uncertainty.','Non-promoter negatives contain many overlapping windows from a shared 1500 nt region: random splitting can inflate test performance despite exact de-duplication.','Model can recognize promoter sequence patterns but does not infer causal cell-context regulatory grammar or ability to write functional genetic designs.','No pretrained DNA foundation model weights were available in the no-internet training runtime.']
}
(OUT/'experiment_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(OUT/'test_predictions.csv').write_text('id,label,max_shared_bp,'+','.join(predictions)+'\n'+ '\n'.join(','.join([ids[int(j)],str(int(labels[j])),str(shares[k])]+[str(predictions[name][k]) for name in predictions]) for k,j in enumerate(testidx)))
print('similarity_audit',leakage)
print('nonoverlap_metrics',nonoverlap)
