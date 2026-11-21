# Experiment 007: baselines for sequence → measured activity

**Status: protocol committed, not yet run.** This file was committed before
any model was fitted, so the rules below could not be adjusted to the results.

## Question

How well does a 200-base DNA sequence predict its measured MPRA activity in
K562 cells, and how much of that is simple composition versus position?

## Protocol

The full protocol is the docstring of `run_baselines.py`. In short:

- **Data:** ENCODE ENCSR997GVD from `experiments/006`. Controls removed,
  200-base sequences only, identical sequences collapsed.
- **Split by chromosome.** Chromosomes 7 and 13 are a frozen holdout, used
  once. The other 20 chromosomes form 5 folds of whole chromosomes for
  choosing settings.
- **Leakage audit:** no sequence on both sides (asserted), and holdout
  sequences sharing an exact 30-mer with development data are counted and
  also scored separately.
- **Models:** mean, GC content, ridge on 1- to 6-mer counts, ridge on
  position-by-base indicators, and a small CNN.
- **Primary endpoint:** Pearson r on the holdout, with a 95% interval from
  resampling whole variants.
- **Ceiling:** measurement error caps what any model can reach; it is
  estimated from the reported standard errors.

## Run

```bash
../../../.venv/bin/python -I ../../006/mpra_acquisition/fetch_encode_mpra.py
../../../.venv/bin/python -I run_baselines.py
```
