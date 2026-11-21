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

## Not verified

**No pretrained genomic language model has been run.** `experiments/004`
contains a written benchmark script and nothing else. Its README says so.
Only a tokenizer was ever downloaded. Any statement about how a genomic
foundation model performs on this data would be invention.

**No quantitative sequence-to-activity dataset has been obtained.** The MPRA
accessions named in `HANDOFF.md` have never been downloaded or inspected in
any session. Whether they contain a usable joined sequence-and-activity table
is unknown.

**The environment setup script is unproven.** The pinned versions in
`requirements.txt` are installed and importable in the session that wrote this
file, but the script has not yet run on a fresh container.

## Constraints that shaped the work

The cloud session has 4 CPUs, no GPU, and a host allowlist. Running a real
genomic foundation model is impractical here. A machine with a GPU is the
precondition for the next experiment, not an optimisation.
