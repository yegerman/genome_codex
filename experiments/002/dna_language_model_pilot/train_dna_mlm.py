#!/usr/bin/env python3
"""Tiny transformer masked-language-model study on 106 public E. coli sequences.
The model is trained from scratch, only on each train split's sequences and
without test labels or sequences. It is not a pretrained foundation model.
"""
import csv, json, time, random, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

ROOT=Path(__file__).parent
DATA=ROOT/'promoters_source.csv'
rows=list(csv.reader(DATA.open()))
S=np.array([r[2].upper() for r in rows]); Y=np.array([int(r[0]=='+') for r in rows]); IDs=np.array([r[1] for r in rows]);n=len(S)
assert n==106 and all(len(s)==57 for s in S)
TOK=np.array([["ACGT".index(x) for x in s] for s in S],dtype=np.int64)
parent=list(range(n))
def root(x):
 while parent[x]!=x:
  parent[x]=parent[parent[x]];x=parent[x]
 return x
def union(a,b):
 a,b=root(a),root(b)
 if a!=b:parent[a]=b
kmers=defaultdict(set)
for i,s in enumerate(S):
 for j in range(33):kmers[s[j:j+25]].add(i)
for v in kmers.values():
 v=list(v)
 for k in v[1:]:union(v[0],k)
G=np.array([root(i) for i in range(n)])
folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2026).split(S,Y,G))
assert len(np.unique(G))==69

class DNAEncoder(nn.Module):
 def __init__(self, width=32):
  super().__init__(); self.base=nn.Embedding(5,width); self.pos=nn.Embedding(57,width)
  layer=nn.TransformerEncoderLayer(width,4,dim_feedforward=width*2,dropout=0.05,batch_first=True,norm_first=True,activation='gelu')
  self.enc=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
  self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,4)
 def forward(self,seq):
  pos=torch.arange(seq.shape[1],device=seq.device)[None]
  h=self.norm(self.enc(self.base(seq)+self.pos(pos)))
  return h,self.head(h)

def mask_data(arr, rng, frac=.2):
 m=rng.random(arr.shape)<frac
 for i in range(len(m)):
  if not m[i].any():m[i,rng.integers(0,arr.shape[1])]=True
 input_ids=arr.copy();input_ids[m]=4
 return torch.as_tensor(input_ids,dtype=torch.long), torch.as_tensor(arr,dtype=torch.long), torch.as_tensor(m,dtype=torch.bool)

def embedded_features(model,tokens):
 model.eval();parts=[];pool=[]
 with torch.no_grad():
  for j in range(0,len(tokens),32):
   h,_=model(torch.tensor(tokens[j:j+32],dtype=torch.long))
   parts.append(h.reshape(len(h),-1).numpy());pool.append(h.mean(dim=1).numpy())
 return np.vstack(parts),np.vstack(pool)

def fixed_probe(a,y,b):
 pipe=make_pipeline(StandardScaler(),LogisticRegression(solver='liblinear',C=.03,max_iter=5000))
 pipe.fit(a,y)
 return pipe.predict(b),pipe.decision_function(b)

def masked_accuracy(model,te,seed):
 rng=np.random.default_rng(seed);masked,lab,m=mask_data(te,rng)
 model.eval()
 with torch.no_grad():
  _,logits=model(masked); pred=logits.argmax(dim=-1)
 return float((pred[m]==lab[m]).float().mean().item()),int(m.sum().item())

def train_fold(tr,te,fold,steps=180):
 torch.manual_seed(1200+fold);np.random.seed(1200+fold)
 model=DNAEncoder()
 init=DNAEncoder();init.load_state_dict(model.state_dict())
 enc=nn.ModuleList([model])
 opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
 rng=np.random.default_rng(5000+fold)
 trtok=TOK[tr];tetok=TOK[te]
 losses=[];start=time.monotonic()
 model.train()
 for step in range(steps):
  batch=rng.integers(0,len(trtok),size=32)
  inp,lab,mask=mask_data(trtok[batch],rng)
  _,out=model(inp);loss=nn.functional.cross_entropy(out[mask],lab[mask]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  losses.append(float(loss.detach()))
  if (step+1)%60==0:
   print(f'fold={fold} step={step+1}/{steps} loss={np.mean(losses[-20:]):.4f} secs={time.monotonic()-start:.1f}',flush=True)
 masked_scores=[masked_accuracy(model,tetok,42+fold*50+k)[0] for k in range(4)]
 # Position-aware empirical baseline -- per position training frequencies with laplace smoothing.
 pos_counts=np.zeros((57,4))
 for s in trtok:
  for j,b in enumerate(s): pos_counts[j,b]+=1
 pos_prob=(pos_counts+1)/(len(trtok)+4)
 baseline_scores=[];unigram=[]
 freqs=(np.bincount(trtok.flatten(),minlength=4)+1)/(trtok.size+4)
 for k in range(4):
  rg=np.random.default_rng(42+fold*50+k);_,lab,mask=mask_data(tetok,rg)
  rows,cols=np.where(mask.numpy())
  baseline_scores.append(np.mean(np.argmax(pos_prob[cols],axis=1)==lab.numpy()[rows,cols]))
  unigram.append(np.mean(np.argmax(freqs)==lab.numpy()[rows,cols]))
 trf,trp=embedded_features(model,trtok);tef,tep=embedded_features(model,tetok)
 rf,rp=embedded_features(init,trtok);rtef,rtep=embedded_features(init,tetok)
 result={}
 for label,a,b in [('mlm_flatten',trf,tef),('random_flatten',rf,rtef),('mlm_mean',trp,tep),('random_mean',rp,rtep)]:
  pred,logit=fixed_probe(a,Y[tr],b)
  result[label]={'idx':te.tolist(),'pred':pred.tolist(),'scores':logit.tolist()}
 print(f'fold={fold} heldout_mlm_acc={np.mean(masked_scores):.3f} pos_freq_acc={np.mean(baseline_scores):.3f} outcomes='+str({k:round(accuracy_score(Y[te],v['pred']),3) for k,v in result.items()}),flush=True)
 return {'fold':fold,'n_train':len(tr),'n_test':len(te),'first20_last20_train_loss':[np.mean(losses[:20]),np.mean(losses[-20:])], 'heldout_masked_acc':float(np.mean(masked_scores)),'heldout_posfreq_acc':float(np.mean(baseline_scores)),'heldout_unigram_acc':float(np.mean(unigram)),'probes':result,'elapsed_sec':round(time.monotonic()-start,2)},model

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 summary=[];out={name:{'pred':np.zeros(n,dtype=int),'scores':np.zeros(n)} for name in ['mlm_flatten','random_flatten','mlm_mean','random_mean']}
 for fold,(tr,te) in enumerate(folds):
  assert set(G[tr]).isdisjoint(set(G[te])); print('Running fold',fold,'train',len(tr),'test',len(te),flush=True)
  report,model=train_fold(tr,te,fold,steps=180);summary.append({k:v for k,v in report.items() if k!='probes'})
  for k,v in report['probes'].items():out[k]['pred'][te]=v['pred'];out[k]['scores'][te]=v['scores']
  if fold==0:torch.save(model.state_dict(),ROOT/'example_trainfold0_masked_lm.pt')
 metrics=[]
 for name,v in out.items():
  pred=v['pred'];s=v['scores']
  metrics.append({'model':name,'accuracy':float(accuracy_score(Y,pred)),'balanced_accuracy':float(balanced_accuracy_score(Y,pred)),'roc_auc':float(roc_auc_score(Y,s)),'confusion_matrix':confusion_matrix(Y,pred).tolist(),'correct':int((Y==pred).sum())})
 with (ROOT/'out_of_fold_predictions.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['id','label','group']+[m+' prediction' for m in out]);
  for i in range(n):w.writerow([IDs[i],Y[i],G[i]]+[out[k]['pred'][i] for k in out])
 stats={'experiment':'Tiny masked DNA transformer trained ONLY from scratch on the UCI promoter sequences; NOT an external pretrained DNA foundation model','dataset':'UCI molecular biology promoter gene sequences','n':n,'groups':len(np.unique(G)),'architecture':'2-layer, width 32, 4 attention heads, masked nucleotide reconstruction; mean/flatten representations + frozen linear probe','pretraining':'180 gradient steps/fold, 20% masked bases, only outer-fold train sequences, without labels','probe':'logistic regression with fixed C 0.03 on fold train labels; tested on group-disjoint holdout','comparison_to_prior':'prior stronger baseline was supervised one-hot position linear classifier, 100/106 across same group splits','folds':summary,'metrics':metrics,'limitations':['Only 106 sequences. Not a true LLM or pretrained DNA foundation model.','Cannot infer genome-wide rules or inverse-design reliability.','Masked reconstruction is a proxy for statistical sequence structure, not biological function.','Fixed probe hyperparameter (no nested tuning); prior baseline had nested tuning, so numbers are not a fair head-to-head model selection experiment.']}
 (ROOT/'results.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
