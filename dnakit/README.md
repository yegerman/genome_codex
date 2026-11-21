# dnakit — read DNA from files, write it back out

The repository's experiments measure DNA. This library is the layer under
them: it gets sequence out of real files, interprets it with the standard
genetic code, and writes it back in a format other tools accept.

Install comes with `./setup.sh`. Tests: `make test` (94 checks, no network).

Runnable straight away, against data already in the repository:

```bash
dnakit read     experiments/001/genetic_ml_pilot/promoters_source.csv --limit 3
dnakit promoter experiments/001/genetic_ml_pilot/promoters_source.csv --limit 1
dnakit write    experiments/001/genetic_ml_pilot/promoters_source.csv -o promoters.fasta
```

## Reading

```bash
dnakit read      sequences.fasta            # length, GC, base composition
dnakit translate gene.fasta --stop-at-stop  # protein, one-letter codes
dnakit orfs      genome.fasta --min-aa 50   # open reading frames, both strands
dnakit revcomp   reads.fasta                # the other strand
dnakit promoter  window.fasta               # score with the experiment 001 model
dnakit promoter  region.fasta --scan --top 5
```

Any command takes `--json` for machine-readable output and `--limit N` to stop
after N records. Both work before or after the subcommand.

Formats read: FASTA (`.fa .fasta .fna .ffn .frn`), the ORIGIN block of GenBank
flat files (`.gb .gbk .genbank`), the UCI promoter CSV (`.csv .data`), and
plain sequence text (`.txt .seq`).

## Writing

```bash
dnakit write input.gb -o output.fasta       # convert anything readable to FASTA
dnakit revcomp input.fasta -o flipped.fasta
```

Writing means serialising sequence records to a file. It is not sequence
design for synthesis, and nothing here proposes or evaluates a construct to
be made in a laboratory.

## From Python

```python
from dnakit import read, translate, find_orfs, write_fasta

records = read("gene.fasta")
protein = translate(records[0].sequence, stop_at_stop=True)
orfs = find_orfs(records[0].sequence, min_aa=50)
write_fasta([r.reverse_complemented() for r in records], "flipped.fasta")
```

## The two layers, kept apart

`dnakit.sequence` and `dnakit.io` are **deterministic**. Transcription,
translation, reverse complement and reading frames follow the published
genetic code, so every answer can be checked by hand. The tests verify
translation against the known start of human beta-globin
(`MVHLTPEEKSAVTALWGKVNVDEVGGEALGR`, UniProt P68871).

`dnakit.promoter` is a **model**, in its own module so the two are never
confused. It applies the logistic-regression weights fitted in
`experiments/001` to a 57-base window and reports which positions drove the
score. Its limits, which make the number meaningless if ignored:

- bacterial. The training data is 106 *E. coli* sequences.
- promoter versus not, never an amount of expression.
- not calibrated: `experiments/003` showed the accuracy behind it moves by
  several points depending on an arbitrary fold assignment, so a score near
  the threshold is undecided.
- says nothing about human sequence, variants, disease or editing.

A test asserts that `dnakit.promoter` returns the same number as the original
`experiments/001/genetic_ml_pilot/predict.py`, so this module stays tied to
the stored artifact instead of drifting into a reimplementation.

### What the model reads

Scoring the first UCI sequence, the positions that moved the score most are
15, 16, 17 and 39, 40, 41 of the 57-base window. Those sit where the
bacterial −35 and −10 promoter boxes are expected, which is consistent with
`HANDOFF.md`. That is a correspondence between learned weights and known
biology, not causal evidence, and it comes from the same 106 sequences the
model was fitted on.

## Design notes

- Nothing is silently repaired. An unexpected base raises `ValueError` naming
  its position, because dropping it would shift the reading frame and change
  every downstream amino acid.
- An ambiguity code in a codon translates to `X`, never a guess.
- An ORF without an in-frame stop codon is not reported, since its extent is
  unknown within the fragment given.
- `find_orfs` returns coordinates on the forward strand for both strands, so
  hits can be compared directly.
