#!/usr/bin/env python3
"""Demonstration inference from a trained full-data UCI E. coli promoter model.
Requires a 57 nt A/C/G/T DNA string; NOT a human gene-expression predictor.
"""
from pathlib import Path
import argparse, numpy as np

def predict(sequence):
 s=''.join(sequence.split()).upper()
 if len(s)!=57 or any(b not in 'ACGT' for b in s):
  raise ValueError('Input must be exactly 57 DNA letters (A/C/G/T).')
 weights=np.load(Path(__file__).parent/'model_coefficients.npz')
 coef=weights['coef'].reshape(-1); intercept=float(weights['intercept'][0])
 x=np.zeros((57,4),np.float64)
 for i,b in enumerate(s):x[i,'ACGT'.index(b)]=1
 z=float(x.ravel()@coef+intercept)
 p=float(1/(1+np.exp(-np.clip(z,-100,100))))
 return {'probability_promoter_model_score':round(p,5),'predicted_class':'promoter' if p>=.5 else 'non_promoter','scope':'UCI E.coli 57nt promoter recognition only','warning':'Score is not calibrated; not a quantitative gene expression / disease / editing prediction.'}

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('sequence',help='57 nt sequence, 5prime to 3prime, aligned as UCI -50 to +6'); args=ap.parse_args()
 import json
 print(json.dumps(predict(args.sequence),indent=2))
