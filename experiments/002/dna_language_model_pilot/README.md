# DNA Language Model — experiment 002 (20 Nov 2026)

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
