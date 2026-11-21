# Experiment 004: pretrained genomic language model vs tuned baselines

**Status: executed 2026-11-21 on an Apple M5 Pro (CPU inference).** Results are
in `nt_benchmark_results.json`; console output is in `run_log.txt`.

`run_nt_benchmark.py` extracts frozen embeddings from a pretrained Nucleotide
Transformer and compares them with tuned k-mer and positional baselines on the
UCI E. coli promoter set, averaged over 20 fold assignments.

## Result

Mean over 20 group-disjoint fold assignments, 106 sequences each:

| model | mean accuracy | sd | range | mean ROC AUC |
|---|---|---|---|---|
| positional one-hot, logistic | **91.60%** | 1.22 | 89.6 – 94.3% | 0.965 |
| pretrained NT, frozen + logistic | 87.03% | 1.93 | 83.0 – 90.6% | 0.942 |
| k-mer (3–4) counts, logistic | 81.93% | 2.10 | 77.4 – 85.8% | 0.865 |

Paired on the same fold assignment, the pretrained model:

- beat the k-mer baseline in **20 of 20** assignments, by +5.09 pp on average (sd 2.97);
- lost to the positional one-hot model in **20 of 20**, by −4.58 pp on average (sd 1.79).

The inner cross-validation chose per-token (position-preserving)
representations in 76 of 80 outer folds, most often the final layer (39).
Mean-pooled representations were chosen 4 times.

Note: the model labelled `GC_kmer_baseline` in the JSON is the 3–4-mer count
model, not GC content alone. The name comes from the script.

## What this does and does not show

- A frozen, general multi-species genomic language model carries promoter
  information beyond k-mer counts on this data, but a 228-feature positional
  model trained on these 106 sequences alone is better.
- It does **not** show that genomic pretraining is useless. This is one 50M
  model, frozen, no fine-tuning, on 57-bp sequences that the 6-mer tokenizer
  splits into 9 six-mers plus 3 single bases, so motif boundaries do not line
  up with tokens. The dataset is tiny and is a classification task, not
  quantitative activity. The informative test remains a large quantitative MPRA
  dataset, see `HANDOFF.md`.

## How it was run

- Model: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`, Hub commit
  `81b29e5786726d891dbf929404ef20adca5b36f1`, `model.safetensors` (223.6 MB),
  55.9M parameters. All weights loaded: no missing, unexpected or mismatched
  keys. The script asserts this.
- The repo's custom code (`modeling_esm.py`, `esm_config.py`) was read before
  use at that commit. It imports only math, typing, dataclasses, torch and
  transformers, and makes no file, network or process calls.
- That code was written for transformers 4.32 and imports helpers removed in
  transformers 5.0, so the project pins `transformers==4.57.6`.
- Sanity check before the benchmark: masked 6-mer recovery on these sequences
  was 3.9%, against 0.024% uniform chance, confirming trained weights.
- Command, from this directory:

  ```bash
  ../../../.venv/bin/python -I run_nt_benchmark.py
  ```

- Reproducibility: a second run from a clean copy produced a
  `nt_benchmark_results.json` identical byte for byte.
- License of the model is CC-BY-NC-SA-4.0: research use only.

## Context from experiments/003

A single fold assignment moves out-of-fold accuracy by several points. The
published 94.34% sits at the 97th percentile of 100 audit runs (mean 91.1%).
Report means over many assignments, never one split.
