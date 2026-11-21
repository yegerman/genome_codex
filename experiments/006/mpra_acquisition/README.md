# Experiment 006: is there a usable sequence → measured activity dataset?

**Status: executed 2026-11-21. Answer: yes.** Results in `inspection.json`,
file checksums in `manifest.json`.

```bash
../../../.venv/bin/python -I fetch_encode_mpra.py   # about 30 MB, md5-checked
../../../.venv/bin/python -I inspect_mpra.py
```

## Why this exists

`HANDOFF.md` named GEO GSE266775 and GSE266796 as candidates for quantitative
sequence-to-activity data, and warned not to assume from their metadata that
they hold a joined table. Earlier sessions could not download them. This
experiment downloads them and checks the join from the bytes.

## What the two accessions are

Both are GEO mirrors of ENCODE MPRA experiments from the Tewhey lab (JAX), run
in K562 cells with an episomal reporter. An MPRA puts thousands of short DNA
sequences in front of a reporter gene and counts how much RNA each one
produces relative to how much DNA went in. The reported `log2FoldChange` is
that ratio: a measured amount, not a label.

| | ENCSR820RPI (GSE266775) | ENCSR997GVD (GSE266796) |
|---|---|---|
| sequences in the library | 15,000 | 59,474 (60,745 library entries) |
| sequence length | 200 bases (3 shorter) | 200 bases for 95%, 130–199 for the rest |
| activity table rows | 15,000 | 60,747 |
| rows joined to a sequence | **15,000 (100%)** | **60,745 (99.997%)** |
| joined rows with a measured activity | **14,985** | **60,240** |
| log2FoldChange range | −4.04 to +7.07, sd 1.58 | −4.88 to +10.46, sd 1.20 |
| reference/alternate pairs, both measured | none | **29,765** |
| independent units behind those rows | **1,000 source regions** (13,058 mutated versions, 10–40 each) plus controls | 27,541 variant ids |

Total: **75,225 sequences with a measured activity.** The row counts overstate
how many independent things were measured; see the last row and the leakage
warnings below.

## Findings

- **The join works, by identifier.** In ENCSR997GVD it only works fully once
  one convention is handled: where several library entries share an identical
  sequence, the FASTA stores it once under a combined header `(id1;id2)`.
  Expanding 1,099 such headers lifts the join from 58,375 to 60,745 rows. The
  two rows left over are a single ALU insertion with no sequence given.
- **The activity column behaves like activity.** Each library carries its own
  controls, and they separate as they should:

  | experiment | group | n | median log2FC | share above 1 |
  |---|---|---|---|---|
  | ENCSR820RPI | positive controls | 91 | +3.93 | 92% |
  | ENCSR820RPI | negative controls | 506 | +0.10 | 3% |
  | ENCSR820RPI | shuffled negative controls | 249 | +0.06 | 8% |
  | ENCSR997GVD | `ctrl_emvar` controls | 269 | +2.32 | 68% |
  | ENCSR997GVD | negative controls | 505 | +0.23 | 6% |

  `ctrl_emvar` is the submitters' label. It reads like "expression-modulating
  variant", a positive control, and the numbers agree, but the files do not
  define it.
- **ENCSR820RPI is a mutagenesis library.** Its 13,058 non-control entries
  are mutated versions of 1,000 genomic regions, with identifiers such as
  `chr1_100759306_100759456_mut_1_rep_1_saturation`. That makes it a direct
  measurement of which letters in a region matter, and it also means the
  entries are far from independent.

- **ENCSR997GVD is a variant-effect library.** 29,765 pairs of sequences are
  measured as reference and alternate allele. In 26,875 pairs the two
  sequences differ by exactly one letter; 2,890 are insertions or deletions.
  That is a measured answer to "what does changing this one letter do", which
  is what `dnakit diff` can only describe for protein-coding sequence.
- **Sequences are clean.** No ambiguous letters in either library.

## What this does not establish

- **Nothing has been modelled.** This experiment inspects; it fits nothing.
- **Reporter activity is not native gene expression.** These are 200-base
  fragments on a plasmid in one cell line (K562), 24 hours after transfection.
- **The library labels are the submitters'.** Groups such as `GATA`, `ALP`,
  `GGT`, `SBP` are reported as they appear in the files. ENCODE lists no
  publication for either experiment, so their design rationale is not
  asserted here.
- **The two large tables were not opened** (ENCFF294GFK, 381 MB; ENCFF227KRF,
  859 MB). They are listed in the manifest. Whether they hold per-replicate
  values is unknown.

## Leakage warnings for whoever models this next

These follow from the structure found above and would silently inflate any
score if ignored:

1. A reference/alternate pair differs by one letter in 200. Both members
   **must** stay on the same side of any split.
2. 1,099 sequences appear under two or more identifiers. Split on sequence,
   not on identifier.
3. In ENCSR820RPI, 10 to 40 entries are mutated versions of the same region
   and differ by a few letters. Split by source region. The honest sample
   size there is 1,000 regions, not 13,058 rows.
4. In ENCSR997GVD, 641 variants are tested in more than one window (`left`,
   `center`, `right`) or haplotype background, sharing most of their
   sequence. Group by variant at least, by chromosome to be safe. Every
   sequence is on the forward strand, so nothing here tests strand.
5. Controls are not test data. Decide before fitting whether they are held
   out, and say so.

## Environment note

Downloads come from `www.encodeproject.org` and redirect to
`encode-public.s3.amazonaws.com`. Neither is on the cloud allowlist in
`ENVIRONMENT.md`, so this experiment currently runs locally only. NCBI's own
file listing for GSE266775 returned 403 on the day of the run, which is why
ENCODE, the original source, is used for both.
