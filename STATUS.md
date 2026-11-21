# Status — what is verified, what is not

Updated 2026-11-21. This file separates measured facts from claims that have
not been checked. `HANDOFF.md` holds the research context and the plan;
this file holds the evidence.

## Verified in this repository

**Both historical experiments reproduce exactly.** Re-ran `train_grouped.py`
and `train_dna_mlm.py` from clean copies. Every metric matched the stored
JSON to the digit, including per-fold masked-reconstruction accuracies.

**The dataset is authentic.** `experiments/003/dataset_verification` downloads
the official UCI file and compares it to the transcribed local CSV. All 106
records match, 53 positive and 53 negative. This closes the open provenance
question that `HANDOFF.md` flagged as unverifiable offline.

| | upstream | local |
|---|---|---|
| records | 106 | 106 |
| label counts | 53 / 53 | 53 / 53 |
| record mismatches | — | 0 |

Raw bytes differ by design: the local CSV strips the field padding present in
the upstream file. The comparison is on normalised records.

**The headline accuracy is optimistic.** `experiments/003/leakage_audit`
tested how much the reported number depends on choices that carry no
biological meaning. Relabelling the similarity groups leaves the partition of
sequences completely untouched, yet moves out-of-fold accuracy as much as
re-seeding the split does.

| | accuracy | correct of 106 |
|---|---|---|
| as published | 94.34% | 100 |
| mean over 100 runs | 91.06% | 96.5 |
| range | 86.8% – 95.3% | 92 – 101 |

Standard deviation 1.9 points. The published figure sits at the 97th
percentile of those runs. **Report means over many fold assignments, never a
single split.**

**Model ranking is stable even though the numbers move.** The positional
one-hot model was best in 99 of 100 runs and beat the GC baseline in 100 of
100. The conclusion that position matters survives; the specific percentage
does not.

**The similarity threshold itself is not critical.** Sweeping the grouping
threshold from 10 to 35 bp, on both strands, kept accuracy between 88.7% and
94.3%, with no trend indicating residual overlap drives the result.

**A pretrained genomic language model has now been run, on this small set
only.** `experiments/004` used frozen embeddings from Nucleotide Transformer
v2 50M multi-species (pinned Hub commit, all weights loaded), tuned inside the
training folds, over 20 group-disjoint fold assignments.

| | mean accuracy | sd |
|---|---|---|
| positional one-hot | 91.60% | 1.22 |
| pretrained NT, frozen | 87.03% | 1.93 |
| 3–4-mer counts | 81.93% | 2.10 |

Paired by assignment, the pretrained model beat k-mers 20 of 20 times and lost
to the positional model 20 of 20 times. This is one frozen 50M model on 106
short sequences; it says nothing yet about fine-tuning, larger models or
quantitative activity.

**The textbook −10 box is real signal in this dataset.** `experiments/005`
counted the bacterial promoter boxes across all 106 sequences, with the test
fixed before looking.

| box | promoters | non-promoters | one-sided Fisher p |
|---|---|---|---|
| −10 `TATAAT` | 39/53 (74%) | 3/53 (6%) | 8.9 × 10⁻¹⁴ |
| −35 `TTGACA` | 26/53 (49%) | 19/53 (36%) | 0.12 |

The −10 matches cluster at window positions 36–37, inside the span where the
experiment 001 model puts its strongest weights: a fixed pattern and a learned
model agree on where the signal is. The −35 box is not enriched by presence
alone. Positions were read off this data, so they carry no p-value.

**`dnakit` reads, explains and reviews sequence.** 172 tests, no network. It
translates against the published start of human beta-globin, classifies
synonymous, missense, nonsense, stop-loss, start-loss, frameshift and in-frame
changes, and reports motif matches with Poisson odds of coincidence. It
reports what a sequence says under the genetic code, never what it does in a
cell.

**A quantitative sequence-to-activity dataset exists and joins.**
`experiments/006` downloaded the two ENCODE K562 MPRA experiments behind GEO
GSE266775 and GSE266796, with md5 checks, and joined sequence to measured
activity from the bytes.

| | ENCSR820RPI | ENCSR997GVD |
|---|---|---|
| rows joined to a sequence | 15,000 of 15,000 | 60,745 of 60,747 |
| with a measured log2 fold change | 14,985 | 60,240 |
| reference/alternate pairs measured | none | 29,765 (26,875 differ by one letter) |
| independent units | 1,000 source regions | 27,541 variant ids |

Positive controls sit well above negative controls in both (median log2 fold
change +3.93 against +0.10, and +2.32 against +0.23), so the activity column
measures activity. This closes the question `HANDOFF.md` left open. It is
reporter activity on a plasmid in one cell line, not native gene expression.

**Sequence predicts measured activity, modestly.** `experiments/007` fitted
baselines to the ENCSR997GVD MPRA data under a protocol committed before any
model was fitted, and scored them once on a frozen holdout (chromosomes 7 and
13; 4,335 sequences from 2,155 variants).

| model | holdout Pearson r | R² |
|---|---|---|
| GC content | 0.194 | 0.036 |
| position-by-base ridge | 0.239 | 0.056 |
| CNN, 3 seeds averaged | 0.452 | 0.201 |
| k-mer ridge (1–6-mers) | 0.475 | 0.225 |
| measurement ceiling | about 0.934 | about 0.873 |

Short words beat position by +0.236 (95% interval +0.198 to +0.275), the
opposite of experiment 001. The untuned CNN and k-mer ridge cannot be told
apart (+0.023, interval −0.008 to +0.056). Cross-validation agrees with the
holdout, the 3,456 holdout sequences sharing no 30-mer with development data
score the same, and a second complete run reproduced every number. The best
model explains about a quarter of the variance that measurement error leaves
explainable.

## Not verified

**No pretrained model has been tested on quantitative activity data.**
Experiment 004 used one on 106 bacterial sequences only. Whether pretraining
helps on the MPRA data is unknown.

**Variant effects have not been predicted.** The 26,875 measured one-letter
pairs have not been used to test whether a model predicts what a change does.

**The environment setup script has run on one local machine only.** On
2026-11-21, `./setup.sh` built `.venv` from empty on an Apple M5 Pro (macOS,
Python 3.14.8) in 75 seconds. torch reported `mps`. `make verify`,
`make audit` and `make verify-dataset` all passed with every stored result
unchanged. It has not yet run on a fresh Linux container or the dev container.

## Constraints that shaped the work

The cloud session has 4 CPUs, no GPU, and a host allowlist. Running a real
genomic foundation model is impractical here. A machine with a GPU is the
precondition for the next experiment, not an optimisation.
