# Genetic Language Research — Experiment 001: DNA promoter recognition

**Status: completed training, evaluated and packaged.**

## Goal and scope
The research vision is to learn how DNA sequence encodes biological function and, much later, to understand how to read and write genetic instructions. This experiment is a narrow **first empirical test of machine learning on regulatory DNA**, not a genetic editor and not a universal model of DNA.

We first tried to retrieve human K562 MPRA datasets (GSE266775/GSE266796) but the execution environment cannot resolve external servers. Instead we trained models on the small, genuinely labeled, public **UCI Molecular Biology (Promoter Gene Sequences)** dataset containing 53 E. coli promoters and 53 non-promoter windows (57 nucleotides each). **The experiment does not measure promoter strength or HBB expression.**

### Dataset and provenance
- Official UCI dataset: https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences
- Raw browser-visible official file: https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data
- Attribution: Harley, C., Reynolds, R., and Noordewier, M. (1987), *Molecular Biology (Promoter Gene Sequences)*, UCI Machine Learning Repository, DOI: 10.24432/C5S01D.
- License: CC BY 4.0. This local CSV was transcribed from the 106-line official public file visible in a web reader; tabs and field whitespace were stripped. Count, label counts, length and alphabet were checked, **but raw-byte identity to upstream could not be verified** in the offline runtime.
- `promoters_source.csv`: `+/-`, source ID, and upper/lowercase 57 nt sequence (all ACGT).

### Models and evaluation
1. Majority-prior dummy.
2. GC percentage → logistic regression.
3. Counts of 3–4 base-pair motifs → logistic regression.
4. Position-aware 57×4 one-hot DNA → logistic regression.
5. Position-aware one-hot DNA → RBF SVM.

The evaluation uses **four outer StratifiedGroupKFold splits**. Connected components were built by exact 25-nt substring overlap; two records with a shared contiguous 25-mer **cannot** be in different folds, including connected transitive chains. Hyperparameters were chosen with **three inner grouped folds** on the training portion only. All 106 records were tested exactly once (out-of-fold). Group-bootstrap intervals resample the 69 sequence-similarity components. 

### Results (real execution)

| Model | Correct / 106 | Group-disjoint out-of-fold balanced accuracy | AUC |
|---|---:|---:|---:|
| Majority | 52 | 49.1% | 0.486 |
| GC-only | 74 | 69.8% | 0.734 |
| 3–4-mers logistic | 89 | 84.0% | 0.868 |
| Positional logistic | **100** | **94.3%** | **0.984** |
| Positional SVM | 98 | 92.5% | 0.981 |

Best model confusion matrix (actual rows 0/1; predicted columns 0/1): `[[50, 3], [3, 50]]`. Group-bootstrap 95% interval for best-model balanced accuracy: **89.2%–98.2%**. A separate fixed-model, exploratory 400-permutation comparison produced **p ≈ 0.0025**, but its label permutation breaks original within-group label structure and is not a substitute for independent biological validation.

### What did it learn?
A logistic coefficient analysis trained *on all data only for interpretation* ranked positions near **−36, −35, −34, −33**, and **−12, −11, −10** strongly. The recognized E. coli sigma-70 promoter motifs are historically reported around **−35 (TTGACA)** and **−10 (TATAAT)**. That positional match is scientifically plausible but is **not causal proof** and could include dataset biases. A held-out group-fold feature ablation found a comparable 94.3% classification with only positions −50 to −32, underscoring that this small dataset may be too easy/confounded and cannot establish that the model has captured all promoter biology.

Relevant sources:
- https://www.ncbi.nlm.nih.gov/books/NBK9850/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC99504/

### Important limitations
- Very small, historical bacterial dataset; not quantitative K562 MPRA, not human HBB, not genome-wide transcript regulation.
- Negative class consists of overlapping genomic windows from a limited source. Grouping by common 25mers is stronger than naive random splitting but cannot eliminate **all** shared biological source and sampling bias.
- Reported probabilities from the full-data model are not calibrated, and there is no guarantee of transfer to another gene, genome, or species.
- No DNA foundation model was downloaded or fine-tuned. We trained ordinary supervised DNA-sequence machine learning **from scratch**; avoid calling it an LLM.
- The data do not enable predictions of disease, protein function, gene editing success or safe gene design.

### Reproduce locally
Requires Python 3.10+, NumPy, scikit-learn, joblib and Matplotlib (for figures). No internet required once this ZIP is present.

```bash
python train.py
python train_grouped.py
python permutation.py
python analyze_language.py
python render_results.py
python predict.py 'tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt'
```

- `grouped_results.json` contains all fold metrics, train/test counts, selected parameters and bootstrap CIs.
- `grouped_predictions.csv` contains all 106 true labels and out-of-fold predictions.
- `permutation_results.json` contains the separate random-label experiment.
- `region_ablation.json` and `learned_positions_chart.png` contain the exploratory position analysis.
- `model_coefficients.npz` stores **a different model retrained on all 106 rows** for demonstration inference in `predict.py`; its in-sample training must not be confused with the out-of-fold evaluation.
- `experiment_results.json` and `test_predictions.csv` are from an initial naive holdout deliberately retained to document leakage risks. The **grouped** files supersede them for reported outcomes.

## Correct next research question
On a much larger independent dataset, predict **measured regulatory activity** (MPRA or another standardized molecular endpoint) for sequences not previously seen. Compare grouped holdout performance to the classical k-mer and position-aware baselines. Until then, this experiment shows useful learning of a *small part* of DNA's regulatory syntax — it does not demonstrate general semantic understanding of DNA.
