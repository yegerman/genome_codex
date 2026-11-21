# Experiment 004: pretrained genomic language model vs tuned baselines

**Status: script written, NOT executed. There are no results from this experiment.**

`run_nt_benchmark.py` extracts frozen embeddings from a pretrained Nucleotide
Transformer and compares them with tuned k-mer and positional baselines on the
UCI E. coli promoter set, averaged over 20 fold assignments.

## Known issues before running
- The script loads `nucleotide-transformer-v2-50m-multi-species` with
  `trust_remote_code=True`. The repo's custom code was read and only imports
  math, typing, torch and transformers. The auto-mode classifier blocked the run
  in the cloud session anyway, so run it locally or switch to a checkpoint that
  loads without remote code (for example the 2.5B multi-species model, which is
  large and needs a GPU in practice).
- License of the 50M model is CC-BY-NC-SA-4.0, so research use only.
- The cloud session has no GPU and 4 CPUs.
- This 106-sequence dataset cannot show a pretraining benefit on its own. The
  informative test is a large quantitative MPRA dataset, see HANDOFF.md.

## Context from experiments/003
A single fold assignment moves out-of-fold accuracy by several points. The
published 94.34% sits at the 97th percentile of 100 audit runs (mean 91.1%).
Report means over many assignments, never one split.
