# Experiment 008: pretrained genomic model vs k-mers on measured activity

**Status: protocol committed, not yet run.** The full protocol is the
docstring of `run_pretrained.py`, committed before any embedding was
extracted.

Same data, split and frozen holdout as experiment 007. A frozen Nucleotide
Transformer (50M, pinned commit) supplies mean-pooled embeddings; ridge on
them is compared with experiment 007's k-mer ridge (holdout Pearson 0.475).
Pretraining is called helpful only if the variant-level 95% interval of the
paired difference lies entirely above zero.

Run: `../../../.venv/bin/python -I run_pretrained.py`
