# Data — the DNA this project reads

If DNA is a program, this folder is the source code under study, stored next
to the tools that read it.

## Contents

| path | what it is | size |
|---|---|---|
| `mpra/ENCSR997GVD_sequence_activity.tsv.gz` | **Ready to use.** 55,008 human 200-base sequences, each with its measured activity (`log2FoldChange`, `lfcSE`), chromosome, variant and the train/holdout split from experiment 007 | 2.7 MB |
| `mpra/ENCSR997GVD/` | ENCODE files as downloaded: activity table, variant-effect table, sequence FASTA | 25 MB |
| `mpra/ENCSR820RPI/` | ENCODE files as downloaded: mutagenesis library of 1,000 regions | 4 MB |

The UCI *E. coli* promoter set used by experiments 001–005 lives in
`experiments/001/genetic_ml_pilot/promoters_source.csv`.

## Provenance and checks

- Source: ENCODE Project, K562 MPRA experiments ENCSR997GVD and ENCSR820RPI
  (Tewhey lab, JAX), mirrored in GEO as GSE266796 and GSE266775.
- Every downloaded file matches the md5 ENCODE publishes; the checksums are in
  `experiments/006/mpra_acquisition/manifest.json`. Re-run
  `experiments/006/mpra_acquisition/fetch_encode_mpra.py` to re-verify.
- The ready-to-use table is derived by `experiments/007/mpra_baselines`
  (`load_units`): controls removed, 200-base sequences only, identical
  sequences merged.
- The `split` column marks chromosomes 7 and 13 as the frozen `holdout`. Do
  not train on it, or every later comparison against experiment 007 is void.

## Terms of use

ENCODE data are released without restriction under the ENCODE Data Use
Policy. Cite the ENCODE Consortium and the producing lab.

## Not stored here

`data/cache/` holds large derived files (model embeddings) and is git-ignored.
Raw sequencing reads (about 14 GB) are listed in the manifest but never
downloaded.
