# Experiment 008: pretrained genomic model vs k-mers on measured activity

**Status: executed 2026-11-21 on an Apple M5 Pro.** Results in
`pretrained_results.json`, console output in `run_log.txt`. The protocol (the
docstring of `run_pretrained.py`) was committed and pushed in `94b440f`
before any embedding was extracted.

## Question

Experiment 007's best model, ridge on 1- to 6-mer counts, reached holdout
Pearson 0.475. Does a pretrained genomic language model, used frozen, read
these sequences better?

## Result on the frozen holdout (chromosomes 7 and 13)

Same 4,335 sequences from 2,155 variants as experiment 007.

| model | Pearson r | 95% interval | R² | no-shared-30-mer subset | CV r |
|---|---|---|---|---|---|
| NT 50M frozen, ridge (layer 6) | 0.383 | 0.338 – 0.429 | 0.142 | 0.379 | 0.405 |
| k-mer ridge (experiment 007) | 0.475 | 0.432 – 0.519 | 0.225 | 0.472 | 0.497 |
| k-mers + NT embedding | **0.484** | 0.442 – 0.525 | **0.233** | 0.481 | 0.504 |

Paired differences, resampling whole variants:

- **Primary: NT minus k-mers = −0.093 (interval −0.122 to −0.063).** The whole
  interval is below zero, so by the pre-registered rule **pretraining hurt**:
  the frozen pretrained representation predicts activity worse than counting
  short words.
- k-mers + NT minus k-mers = +0.008 (interval +0.003 to +0.013). The
  embedding adds something the counts lack, but very little.

Layer chosen by cross-validation: 6 of 12 (CV R² 0.164), ahead of the last
layer (0.135) and the token embeddings (0.051).

## What this shows

- **A small, frozen, general-purpose genomic model is not a shortcut here.**
  It reached about 80% of the k-mer model's Pearson.
- **The middle of the network reads regulatory DNA best**, not its output
  layer, which is tuned to predicting masked tokens.
- **Same story as experiment 004** on bacterial promoters, now on 55,000
  human sequences with measured activity. Two independent datasets, same
  ordering.

## What is not claimed

- **Not that pretraining cannot help.** This is the smallest model in its
  family, frozen, with its tokens averaged. Averaging throws away where in the
  sequence each signal sits. Fine-tuning, a larger model, or keeping per-token
  information could change the answer. None was tested.
- **One dataset, one cell line, reporter activity.** As in experiment 007.

## Implementation notes

- 36 tokens per 200-base sequence; extraction took 150 s on `mps`. Embeddings
  are cached in `data/cache/` (git-ignored, about 560 MB).
- For `kmer_plus_nt`, the two feature blocks were standardised once on all
  development rows rather than inside each fold. This can only affect which
  ridge strength CV picks, not the holdout scoring, and the chosen value
  (1e5) equals the k-mer model's.

Run: `../../../.venv/bin/python -I run_pretrained.py`
