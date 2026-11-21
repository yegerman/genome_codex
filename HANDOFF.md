# Genetic Language Research — Claude Code / Codex handoff

**Prepared:** 2026-11-21. **Owner vision (translated from Hebrew):** "Learn the programming language of DNA step by step. Ultimately read genetic code, understand what it expresses, discover mechanisms, identify defects and eventually know what might need correction or be written." The **research goal is not to build a debugger**; the debugger is a tool for understanding the language. Keep a strict boundary between interpretable molecular predictions and proven biological mechanisms, and do not frame sequence design as safe gene editing.

## Where we stand — only completed work

- `prototypes/genetic_debugger_v02.html`: offline HBB DNA→codon→amino-acid educational interpreter, selected variants and qualitative regulatory graph. No biological-expression simulation.
- `prototypes/genetic_debugger_v03.html`: offline Genetic IDE/Language Lab and mutation atlas. It is a deterministic reference tool, not an ML predictor.
- `experiments/001/genetic_ml_pilot`: completed **supervised**, group-split classifier of E. coli promoters vs non-promoters from 106 sequences of length 57, UCI dataset. **94.34%** grouped out-of-fold accuracy for positional one-hot logistic regression (100/106); 84.0% for k-mer logistic, 69.8% GC. Promoter annotation is **not** expression level. 69 similarity components formed by exact shared 25-mers, 4 group-stratified outer folds, 3 inner folds for model selection. 25-mer grouping does not guarantee biological independence; dataset is tiny.
- `experiments/002/dna_language_model_pilot`: completed **small Transformer trained from scratch** with masked-base reconstruction pretraining on fold-train DNA, then frozen embeddings + supervised logistic probe, same group-disjoint evaluation. The trained Transformer gave **93.40%** promoter classification (99/106) vs **94.34%** (100/106) for random/untrained encoder + probe; MLM base reconstruction on held-out data 27.95%, worse than position-frequency 30.84%. This **did not show that self-supervised language pretraining helped** in this tiny study. It was **not a pretrained genomic LLM or foundation model**.
- Original goal of quantitative **K562 MPRA sequence→activity** using GEO GSE266775/GSE266796 was deferred because downloads of the original datasets were blocked in the previous execution environment. Neither K562/HBB expression measurements nor a DNABERT-2/Nucleotide Transformer checkpoint were actually trained or tested. Do **not** present them as completed.

## Critical scientific conclusion

The strongest observed result is that positional DNA motifs help classify this *small, historical E. coli promoter set*. The model's prominent position weights were near known bacterial -35 and -10 promoter motifs, which is biologically plausible but not causal discovery. We have not deciphered DNA semantics, measured functional gene expression, generalized to human HBB, designed or repaired DNA, or demonstrated benefits of DNA foundation-model pretraining.

## What Claude Code + Codex should do next

1. **Reproduce and audit:** inspect the scripts and saved OOF predictions/results; validate dataset provenance, checksum limitations, group assignment and data leakage. Keep outputs separate from reported historical results.
2. **Acquire an actual quantitative sequence→activity dataset:** candidates include public MPRA experiments, e.g. GEO [GSE266775](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE266775) or [GSE266796](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE266796). Confirm availability and match between DNA sequences and measured activity. Do **not** assume GSE files offer a sufficient joined dataset merely based on their metadata.
3. **Establish reliable baselines:** GC, sequence k-mers, supervised positional models/CNN; train/test split by experiment, region, and homologous sequence groups to reduce contamination.
4. **Try an *actually pretrained* DNA foundation model** (e.g. DNABERT-2 or Nucleotide Transformer) with documented checkpoint/version/license and recorded feature extraction. Frozen embeddings + small head is a sensible first experiment; fine-tuning only if dataset/compute allow and evaluation is clean. Use a fair tuned baseline.
5. **Evaluate on untouched holdout:** report exact N, split/groups, Spearman/Pearson/MAE for quantitative activity, uncertainty/CI, training conditions, seeds, failure cases, and ablation. A negative result is valuable if honestly reported.
6. **Pursue understanding, not accuracy alone:** compare known regulatory motifs and cell-type dependence, challenge explanations with independent data. Keep derived rules/hypotheses distinct from causal evidence.
7. **Return actual artifacts** (dataset manifest, preprocessing, code, version-pinned environment, plots, JSON/CSV results, reproducibility steps) and a concise Hebrew explanation. Update ChatGPT Space only if credentials/access are independently available.

**Important:** Do not jump to wet-lab gene transfer, cloning/viral vectors, therapeutic editing or functional DNA design. Keep this scope to public/synthetic data, molecular prediction and evidence evaluation. The long-run user's vision includes writing genetic instructions, but current evidence is nowhere near reliable implementation.

## Source links (may require ChatGPT login; this ZIP is self-contained)

- [Research Space](https://chatgpt.com/space/page_6af8e7aa1f308191878860a04a3bc5bd)
- [Research plan](https://chatgpt.com/space/page_30cab2d0e60c81919a860414c514055d)
- [GLM benchmark plan](https://chatgpt.com/space/page_f24db9d65420819198f051bb42ee5ca0)
- [Experiment 001 report](https://chatgpt.com/space/page_5f9d08a92b78819192ddc574d15e9bba)
- [Experiment 002 report](https://chatgpt.com/space/page_7d2b202022ec8191ad2b323c4feb6857)

## Local layout and reproduction

```text
./HANDOFF.md
./CLAUDE.md                # Claude Code instructions
./AGENTS.md                # Codex instructions
./experiments/001/genetic_ml_pilot/
./experiments/002/dna_language_model_pilot/
./prototypes/genetic_debugger_v02.html
./prototypes/genetic_debugger_v03.html
```

Both experiment directories contain README files. The original datasets/metrics/checkpoints are included. Code in experiment 002 has its legacy `/mnt/data` **input/output paths changed to relative paths** for portability; the research algorithms are unchanged. Check that no other environment assumptions remain before running. Suggested dependencies: Python 3.11+, numpy, scikit-learn, matplotlib, joblib, torch.

```bash
cd experiments/001/genetic_ml_pilot
python train_grouped.py
# From project root:
cd ../../..
cd experiments/002/dna_language_model_pilot
python train_dna_mlm.py
```

Do not overwrite historical results before backing them up. Compare reproduced metrics to the stored JSON and CSV reports.
