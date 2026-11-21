# dnakit — read DNA, say what it does, review it like source code

The repository's experiments measure DNA. This library is the layer under
them: it gets sequence out of real files, interprets it with the standard
genetic code, explains in plain words what a stretch does, checks its syntax,
and writes it back in a format other tools accept.

Install comes with `./setup.sh`. Tests: `make test` (172 checks, no network).

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

## Explaining — what does this DNA do?

```bash
dnakit explain gene.fasta --organism eukaryote      # plain-words reading
dnakit explain gene.fasta --marks --all             # mark signals, list every step
```

It answers in sentences: how long the sequence is, whether it holds a
complete protein recipe, which control signals appear, and — stated
explicitly every time — what the reading cannot tell you. Then it prints the
sequence as an annotated instruction listing:

```
   pos  codon  aa   what this instruction does
     0  ATG    M    START — begin building the protein here
     3  GTG    V    add Val
    93  TAA    *    STOP — the protein is finished and released
```

## Reviewing — is there a bug?

DNA has real syntax, and breaking it produces defects that read like software
bugs. `lint` checks the syntax; `diff` compares a variant to a reference.

```bash
dnakit lint gene.fasta                         # syntax check
dnakit diff variant.fasta -r reference.fasta   # what a change does
```

| DNA change | what `diff` calls it | the software equivalent |
|---|---|---|
| letter swap, same amino acid | `synonymous` | reformatting: output identical |
| letter swap, different amino acid | `missense` | one value changed |
| letter swap creating a stop | `nonsense` | an early `return` |
| stop codon destroyed | `stop lost` | a block that never closes |
| 1 or 2 letters added or removed | `frameshift` | **off-by-one: everything after is corrupted** |
| 3 letters added or removed | `in-frame` | one step inserted or deleted, rest intact |

The frameshift is the one worth seeing. Deleting a single letter from the
beta-globin fragment turns `MVHLTPEEKSAVTALWGKVNVDEVGGEALGR` into
`MVHLTPRRSLPLLPCGAR*TWMKLVVRPWAG` — correct up to the deletion, unrelated
after it.

**Scope.** `diff` reports what two strings differ by and what that does to
the protein under the genetic code. It does not propose edits to anyone's
genome, design a correction, or claim a difference is a disease. Naming the
reference letter at a position is a statement about two strings.

Amino acids are numbered from 1 at the initiator methionine. Clinical names
usually number the mature protein, after that methionine is removed, a shift
of one — so the change this tool calls `E7V` in beta-globin is the one
published as β6 Glu→Val. Every comparison prints this note.

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

## How far the code analogy actually goes

It holds in places and breaks in others. Where it breaks matters more.

| software | DNA | holds? |
|---|---|---|
| function | gene: defined start, defined end, produces one output | yes |
| entry point | ribosome binding site, where reading begins | yes |
| `if` condition | transcription-factor site: run this gene only if the factor is present | yes |
| comment, stripped before running | intron: transcribed, then cut out before translation | yes |
| off-by-one parser bug | frameshift | yes, exactly |
| reusable module | protein domain, reused across unrelated proteins | roughly |
| class and inheritance | gene family from duplication then divergence | loosely; copy-and-diverge, with no inheritance mechanism |
| one file, one meaning | the same letters are read in up to 6 frames, on 2 strands, often overlapping | **no** |
| deterministic execution | rates vary by cell type, moment and chance | **no** |
| every line executes | most DNA codes for no protein at all | **no** |

The last three are why this library reports what a sequence *says*, never
what it does in a living cell. That difference is the whole discipline.

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
