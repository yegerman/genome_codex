#!/usr/bin/env python3
"""Do the textbook promoter boxes actually appear in real promoters?

dnakit can read a sequence and report a -10 or -35 box. On one 57-base
sequence it cannot say whether that match means anything: a six-letter
pattern allowing one mismatch occurs by chance about once in that length. So
the single-sequence reading is statistically powerless by construction.

This experiment asks the question where it has power: across all 106 UCI
sequences, is each box more common in the 53 promoters than in the 53
non-promoters? If dnakit's motif search reads real biology, the answer should
be yes, and the matches should sit at a consistent position.

Pre-registered test, decided before looking: presence of each box anywhere on
the given strand, promoters versus non-promoters, one-sided Fisher exact
test. Positions are reported as description only. Choosing a position from
this data and then testing it on the same data would be circular, so no
p-value is given for position.
"""
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from dnakit import motifs, read_uci_promoters  # noqa: E402

SOURCE = ROOT / "experiments" / "001" / "genetic_ml_pilot" / "promoters_source.csv"
BOXES = ("-10 box (Pribnow box)", "-35 box")


def fisher_one_sided(hits_a, n_a, hits_b, n_b):
    """P(group A has at least hits_a) under the hypergeometric null."""
    n, k = n_a + n_b, hits_a + hits_b
    total = comb(n, k)
    return sum(comb(n_a, x) * comb(n_b, k - x)
               for x in range(hits_a, min(n_a, k) + 1)) / total


def analyse(records, name):
    motif = motifs.BY_NAME[name]
    groups = {"promoter": [], "non_promoter": []}
    for record in records:
        group = "promoter" if record.description == "class=+" else "non_promoter"
        starts = [h["start"] for h in
                  motifs.find(record.sequence, motif, both_strands=False)]
        groups[group].append(starts)

    counts = {g: {"n": len(rows), "with_match": sum(1 for s in rows if s)}
              for g, rows in groups.items()}
    p = fisher_one_sided(counts["promoter"]["with_match"],
                         counts["promoter"]["n"],
                         counts["non_promoter"]["with_match"],
                         counts["non_promoter"]["n"])

    positions = {
        g: dict(sorted(Counter(s for row in rows for s in row).items()))
        for g, rows in groups.items()
    }
    top = Counter(s for row in groups["promoter"] for s in row).most_common(3)
    return {
        "motif": name,
        "consensus": motif.consensus,
        "max_mismatches": motif.max_mismatches,
        "strand": "given strand only",
        "counts": counts,
        "fraction_with_match": {
            g: c["with_match"] / c["n"] for g, c in counts.items()},
        "fisher_one_sided_p": p,
        "enriched_in_promoters_at_0.05": p < 0.05,
        "match_start_positions": positions,
        "most_common_starts_in_promoters": [
            {"start": s, "count": c} for s, c in top],
        "position_note": ("descriptive only; a position chosen from this "
                          "data cannot be tested on the same data"),
    }


def main():
    records = read_uci_promoters(SOURCE)
    assert len(records) == 106
    results = [analyse(records, name) for name in BOXES]

    report = {
        "question": ("Are the textbook bacterial promoter boxes more common in "
                     "the 53 UCI promoters than in the 53 non-promoters?"),
        "dataset": "UCI promoter gene sequences, 106 x 57 bases",
        "test": "one-sided Fisher exact, presence anywhere on the given strand",
        "results": results,
    }
    (HERE / "enrichment_results.json").write_text(json.dumps(report, indent=2))

    for r in results:
        c = r["counts"]
        print(f"{r['motif']}  ({r['consensus']}, <= {r['max_mismatches']} mismatch)")
        print(f"  promoters:     {c['promoter']['with_match']:>2}/{c['promoter']['n']}"
              f"  ({r['fraction_with_match']['promoter']:.0%})")
        print(f"  non-promoters: {c['non_promoter']['with_match']:>2}/{c['non_promoter']['n']}"
              f"  ({r['fraction_with_match']['non_promoter']:.0%})")
        print(f"  one-sided Fisher p = {r['fisher_one_sided_p']:.2e}"
              f"  -> {'enriched' if r['enriched_in_promoters_at_0.05'] else 'not enriched'}")
        print("  most common starts in promoters: "
              + ", ".join(f"{t['start']} (x{t['count']})"
                          for t in r["most_common_starts_in_promoters"]))
        print()
    print("wrote enrichment_results.json")


if __name__ == "__main__":
    main()
