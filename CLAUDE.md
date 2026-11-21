# Claude Code guidance — Genetic Language Research

Read `HANDOFF.md` first, then `experiments/001/genetic_ml_pilot/README.md` and `experiments/002/dna_language_model_pilot/README.md`.

**Mission:** Discover generalizable rules of how DNA sequence and cellular context affect biological activity, using actual quantitative molecular measurements. The long-term vision is to read, understand and eventually write genetic code, but do not conflate promoter classification with understanding all DNA semantics.

- Do actual work before reporting findings. Never invent a dataset download, checkpoint, training run, model score or biological conclusion.
- Preserve a frozen holdout and prevent leakage by related sequences/regions/experiments.
- Prefer a real public MPRA sequence→activity benchmark and a genomic *pretrained* language model to another toy from-scratch Transformer; compare with honest baselines.
- Keep computed facts, measured empirical associations, model predictions and untested hypotheses separate.
- Report results in Hebrew to the user; code/docstrings may be English.
- Use Codex for independent code review or reproducibility verification if available, but both agents must use the SAME repository; coordinate changes using Git commits and avoid simultaneous edits to the same file.
- Scope is computational, using public or nonfunctional synthetic data. No wet-lab protocols or functional genome-editing plans.
