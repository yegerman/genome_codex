# Experiment 007: baselines for sequence → measured activity

**Status: executed 2026-11-21 on an Apple M5 Pro (CNN on `mps`), about 6
minutes.** Results in `baseline_results.json`, per-sequence holdout
predictions in `holdout_predictions.csv`, console output in `run_log.txt`.

The protocol was committed before any model was fitted (commit `bdf6176`), so
the split, models and endpoints could not be adjusted to the results. It is
the docstring of `run_baselines.py`.

## Question

How well does a 200-base DNA sequence predict its measured MPRA activity in
K562 cells, and how much of that is simple composition versus position?

## Result on the frozen holdout (chromosomes 7 and 13)

4,335 sequences from 2,155 variants, never used to choose anything.

| model | Pearson r | 95% interval | Spearman | R² | cross-validation r |
|---|---|---|---|---|---|
| mean (floor) | undefined | — | — | 0.000 | — |
| GC content | 0.194 | 0.149 – 0.240 | 0.105 | 0.036 | 0.195 ± 0.016 |
| position-by-base ridge | 0.239 | 0.202 – 0.279 | 0.187 | 0.056 | 0.229 ± 0.014 |
| CNN, 3 seeds averaged | 0.452 | 0.408 – 0.498 | 0.417 | 0.201 | 0.437 ± 0.018 |
| **k-mer ridge (1–6-mers)** | **0.475** | 0.432 – 0.519 | 0.466 | **0.225** | 0.497 ± 0.024 |
| measurement ceiling | about 0.934 | | | about 0.873 | |

Intervals come from 1,000 resamples of whole variants. A single CNN seed
scores 0.418, 0.410 and 0.405 (mean 0.411, sd 0.007); averaging the three
gives 0.452.

## What this shows

- **Sequence predicts activity, modestly.** The best model explains R² 0.225
  where measurement error allows about 0.873: roughly a quarter of what is
  there to be explained.
- **Which short words are present matters far more than which letter sits
  where.** Counting 1- to 6-mers reaches 0.475; position-by-base indicators
  reach 0.239. The paired difference is +0.236 (interval +0.198 to +0.275).
  This is the opposite of experiment 001, where position won. There the
  sequences were aligned to a promoter's start; here a 200-base window is
  centred on a variant, so a given position means nothing consistent.
- **The untuned CNN does not beat k-mer counts.** Paired difference, k-mer
  minus CNN: +0.023 (interval −0.008 to +0.056), so the two cannot be told
  apart on this holdout. The CNN overfits quickly: its best validation epoch
  was between 2 and 11 of 25.
- **GC content alone carries a little signal** (0.194), so any claim about a
  richer model should be read against it.

## Checks on the result itself

- **Cross-validation agrees with the holdout** for every model, so choosing
  settings on the development folds did not inflate the numbers.
- **Shared sequence does not explain it.** 879 of the 4,335 holdout sequences
  (20.3%) share an exact 30-mer with a development sequence on another
  chromosome, on either strand. On the 3,456 that share none, k-mer ridge
  scores 0.472 and the CNN 0.438. No sequence sits on both sides.
- **It reproduces.** A second complete run gave identical holdout numbers for
  every model, the CNN included.
- **Ridge strength was not on a grid edge.** The pre-set rule extended the
  grid once, for the k-mer model (to 1e6). The choices, 1e5 for k-mers and
  1e4 for position, are both interior.

## How the units were formed

| step | rows |
|---|---|
| activity table | 60,747 |
| joined to a sequence, with activity and standard error | 60,240 |
| controls removed | 59,376 |
| 200-base sequences only | 56,205 |
| identical sequences collapsed | 55,008 |
| development / holdout | 50,673 / 4,335 |

## Deviations from the committed protocol

1. **Reporting fix.** The first run printed a Pearson of 0.000 for the mean
   model, because the standard deviation of a constant array came out as a
   tiny non-zero number. A constant prediction has no correlation; the check
   now tests the range. No other number changed between the two runs.
2. **Two paired comparisons were added after the first run** (k-mer versus
   CNN, k-mer versus position). They are stored under `post_hoc` and marked
   as such. The protocol's own endpoints are unchanged.

## What is not claimed

- **Not that CNNs are worse than k-mers.** This CNN is one small untuned
  architecture with no augmentation, trained on about 40,000 sequences.
- **Not gene expression.** The target is reporter activity of a 200-base
  fragment on a plasmid in one cell line.
- **Not a mechanism.** Nothing here says which words matter or why. The
  k-mer weights have not been examined.
- **The ceiling is approximate.** It trusts the reported standard errors.
- **One library, one holdout.** ENCSR820RPI was not used. No pretrained
  model was tested. Variant effects were not predicted.

## Run

```bash
../../../.venv/bin/python -I ../../006/mpra_acquisition/fetch_encode_mpra.py
../../../.venv/bin/python -I run_baselines.py
```
