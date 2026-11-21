# Experiment 005: do the textbook promoter boxes appear in real promoters?

**Status: executed 2026-11-21.** Results in `enrichment_results.json`.
Run from this directory: `../../../.venv/bin/python -I run_enrichment.py`.

## Why this exists

`dnakit explain` can find a −10 or −35 box in a sequence. On a single
57-base sequence it correctly refuses to call that a finding: a six-letter
pattern allowing one mismatch turns up by chance about once in that length.
Reading the first UCI promoter, it found an **exact** `TATAAT` and reported it
as 38% likely to be coincidence — statistically right, and still the real −10
box of that promoter.

So a single sequence has no statistical power by construction. This
experiment asks the same question where it does: across all 106 sequences.

## Test, fixed before looking

Presence of each box anywhere on the given strand, 53 promoters against 53
non-promoters, one-sided Fisher exact test. Positions are described, not
tested (see below).

## Result

| box | consensus | promoters | non-promoters | Fisher p |
|---|---|---|---|---|
| −10 (Pribnow) | `TATAAT` ≤1 mismatch | **39/53 (74%)** | 3/53 (6%) | **8.9 × 10⁻¹⁴** |
| −35 | `TTGACA` ≤1 mismatch | 26/53 (49%) | 19/53 (36%) | 0.12 |

- **The −10 box is real signal here.** It is twelve times more common in
  promoters, and 22 of its 39 promoter matches start at position 36 or 37 of
  the window. The positional model from experiment 001 put its strongest
  weights at positions 39–41 (1-based), inside that same span. Two methods,
  one a fixed pattern and one learned, point at the same place.
- **The −35 box is not enriched by presence alone.** With one mismatch
  allowed, `TTGACA` is loose enough to appear by chance in a third of
  non-promoters. In promoters, though, 19 of its 26 matches start at
  position 14 — where the experiment 001 model's other weight cluster sits
  (positions 15–17, 1-based).

## What is not claimed

- **No p-value for position.** Position 14 and 36 were read off this data.
  Testing them on the same 106 sequences would be circular. The positional
  concentration is a description pointing at the next test, not a result.
- **Not causal.** Enrichment shows these patterns co-occur with the promoter
  label in this dataset. It does not show the cell uses them, though the
  textbook mechanism is that it does.
- **Small, single dataset.** 106 sequences from a 1987 *E. coli* collection.

## What it means for dnakit

The motif reader reads real biology: when tested with enough sequences to
have power, the box it reports is strongly enriched exactly where the
independent model says the signal is. Its single-sequence verdict of "not
beyond chance" is a statement about power, not about whether the box is
there.
