import json,csv,numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
P=Path(__file__).parent
rows=list(csv.reader((P/'promoters_source.csv').open()))
seq=[r[2].upper() for r in rows];y=np.array([int(r[0]=='+') for r in rows]);n=len(y)
X=np.zeros((n,57,4),dtype=np.float64)
for k,s in enumerate(seq):
 for j,x in enumerate(s):X[k,j,'ACGT'.index(x)]=1
X=X.reshape(n,-1)
groups=np.array([int(r['group']) for r in csv.DictReader((P/'grouped_predictions.csv').open())])
folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2026).split(X,y,groups))
def fit_predict(y):
 pred=np.zeros(n,dtype='int')
 for tr,te in folds:
  model=LogisticRegression(solver='liblinear',max_iter=1000,C=0.1).fit(X[tr],y[tr]);pred[te]=model.predict(X[te])
 return float(balanced_accuracy_score(y,pred))
observed=fit_predict(y)
rng=np.random.default_rng(20261119)
null=np.array([fit_predict(rng.permutation(y)) for _ in range(400)])
p=(1+int((null>=observed).sum()))/(1+len(null))
print('observed',observed,'permutation p',p,'n',len(null),'null mean',null.mean(),'null max',null.max(),'null 95pct',np.quantile(null,[.025,.975]))
(P/'permutation_results.json').write_text(json.dumps({'fixed_model':'position_onehot_logistic_regression_C_0.1','observed_grouped_bacc':observed,'null_permutations':len(null),'p_one_sided_plus_one':p,'null_mean':float(null.mean()),'null_q025_q975':[float(x) for x in np.quantile(null,[.025,.975])],'null_max':float(null.max()),'notes':'Exploratory permutation analysis: fixed folds on real labels, but permuting labels at individual-sequence level may disrupt original sequence-family class structure; not an independent biological validation.'},indent=2))
