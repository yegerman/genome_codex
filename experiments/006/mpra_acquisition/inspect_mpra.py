#!/usr/bin/env python3
"""Do the ENCODE MPRA files give a usable sequence -> measured activity table?

HANDOFF.md warns not to assume this from metadata. This script answers it
from the bytes: it joins each experiment's activity table to its element
reference FASTA by identifier and reports exactly what matches, what does
not, and what the joined data look like.

It only inspects. It fits nothing and draws no biological conclusion.
Run fetch_encode_mpra.py first.
"""
import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from dnakit import read  # noqa: E402

DATA = ROOT / "data" / "mpra"
EXPERIMENTS = {
    "ENCSR820RPI": {"table": "ENCFF157VVX.tsv", "fasta": "ENCFF798CYL.fasta.gz"},
    "ENCSR997GVD": {"table": "ENCFF612YRJ.tsv", "fasta": "ENCFF621JOY.fasta.gz"},
}


def number(text):
    """Parse a numeric field; the tables write missing values as NA."""
    if text in ("", "NA", "NaN", "."):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_table(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def five_numbers(values):
    values = sorted(values)
    if not values:
        return None
    quantiles = statistics.quantiles(values, n=4)
    return {"n": len(values), "min": values[0], "q1": quantiles[0],
            "median": quantiles[1], "q3": quantiles[2], "max": values[-1],
            "mean": statistics.fmean(values), "sd": statistics.pstdev(values)}


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def inspect(name, files):
    rows = load_table(DATA / name / files["table"])
    records = read(DATA / name / files["fasta"])

    table_ids = [row["ID"] for row in rows]
    direct_ids = {record.id for record in records}
    direct_join = len(set(table_ids) & direct_ids)

    # Where several library entries share one identical sequence, the FASTA
    # stores it once under a combined header "(id1;id2)". The activity table
    # keeps one row per entry, so the header is expanded to recover the join.
    sequences, composite_headers = {}, 0
    for record in records:
        if record.id.startswith("(") and record.id.endswith(")"):
            composite_headers += 1
            for part in record.id[1:-1].split(";"):
                sequences[part] = record.sequence
        else:
            sequences[record.id] = record.sequence
    fasta_ids = list(sequences)
    table_set, fasta_set = set(table_ids), set(fasta_ids)
    joined_ids = table_set & fasta_set

    lengths = Counter(len(record.sequence) for record in records)
    non_acgt = sum(1 for record in records if set(record.sequence) - set("ACGT"))
    duplicate_sequences = len(records) - len({r.sequence for r in records})

    joined = [row for row in rows if row["ID"] in joined_ids]
    with_activity = [row for row in joined
                     if number(row["log2FoldChange"]) is not None]
    activity = [number(row["log2FoldChange"]) for row in with_activity]
    standard_errors = [number(row["lfcSE"]) for row in with_activity
                       if number(row["lfcSE"]) is not None]
    plasmid = [number(row["DNA_mean"]) for row in with_activity
               if number(row["DNA_mean"]) is not None]

    report = {
        "table_file": files["table"],
        "fasta_file": files["fasta"],
        "table_columns": list(rows[0].keys()),
        "table_rows": len(rows),
        "table_unique_ids": len(table_set),
        "fasta_records": len(records),
        "fasta_combined_headers": composite_headers,
        "fasta_ids_after_expanding": len(fasta_set),
        "ids_in_both_before_expanding": direct_join,
        "ids_in_both": len(joined_ids),
        "table_ids_without_sequence": len(table_set - fasta_set),
        "sequences_without_table_row": len(fasta_set - table_set),
        "examples_table_only": sorted(table_set - fasta_set)[:5],
        "examples_fasta_only": sorted(fasta_set - table_set)[:5],
        "sequence_lengths": dict(sorted(lengths.items())),
        "sequences_with_non_acgt": non_acgt,
        "duplicate_sequences": duplicate_sequences,
        "joined_rows": len(joined),
        "joined_rows_with_activity": len(with_activity),
        "log2FoldChange": five_numbers(activity),
        "lfcSE": five_numbers(standard_errors),
        "DNA_mean_plasmid_count": five_numbers(plasmid),
        "project_values": dict(Counter(row.get("project", "")
                                       for row in joined).most_common(12)),
        "chromosomes": len({row["chr"] for row in joined}),
    }

    # Does the activity column behave like activity? The libraries carry
    # their own controls, so positive controls should sit above negative ones.
    by_project = {}
    for row in with_activity:
        by_project.setdefault(row.get("project", ""), []).append(
            number(row["log2FoldChange"]))
    report["activity_by_project"] = {
        project: {"n": len(values),
                  "median_log2FoldChange": statistics.median(values),
                  "fraction_above_1": sum(v > 1 for v in values) / len(values)}
        for project, values in sorted(by_project.items(),
                                      key=lambda item: -len(item[1]))
        if len(values) >= 50
    }

    # How many independent things were measured? Two structures make the row
    # count overstate it, and both matter for any train/test split.
    #  - identifiers like chr1_100_250_mut_3_rep_1_saturation are mutated
    #    versions of one source region;
    #  - one variant can be tested in several windows or haplotype backgrounds.
    region_pattern = re.compile(r"^(chr[^_]+_\d+_\d+)_")
    regions = Counter()
    for row in joined:
        match = region_pattern.match(row["ID"])
        if match:
            regions[match.group(1)] += 1
    if regions:
        per_region = sorted(regions.values())
        report["mutagenesis_source_regions"] = {
            "entries_derived_from_a_region": sum(per_region),
            "distinct_source_regions": len(regions),
            "entries_per_region_min": per_region[0],
            "entries_per_region_median": per_region[len(per_region) // 2],
            "entries_per_region_max": per_region[-1],
        }

    configurations = {}
    for row in joined:
        configurations.setdefault(row["SNP"], set()).add(
            (row.get("window"), row.get("strand"), row.get("haplotype")))
    report["distinct_variant_ids"] = len(configurations)
    report["variants_tested_in_several_configurations"] = sum(
        1 for found in configurations.values() if len(found) > 1)
    report["window_values"] = dict(Counter(row.get("window", "")
                                           for row in joined).most_common(6))
    report["strand_values"] = dict(Counter(row.get("strand", "")
                                           for row in joined).most_common(6))

    # Reference/alternate pairs: identifiers differ only in the allele field,
    # e.g. 1:100004726:G:A:R:wC and 1:100004726:G:A:A:wC.
    measured = {row["ID"]: number(row["log2FoldChange"]) for row in with_activity}
    pairs, distances = 0, Counter()
    for identifier in measured:
        parts = identifier.split(":")
        if len(parts) < 6 or parts[4] != "R":
            continue
        partner = ":".join(parts[:4] + ["A"] + parts[5:])
        if partner not in measured:
            continue
        pairs += 1
        ref, alt = sequences[identifier], sequences[partner]
        distances["different length" if len(ref) != len(alt)
                  else f"{hamming(ref, alt)} letter(s) differ"] += 1
    report["reference_alternate_pairs_with_activity"] = pairs
    report["pair_sequence_differences"] = dict(distances.most_common(8))
    return report


def main():
    reports = {name: inspect(name, files) for name, files in EXPERIMENTS.items()}
    (HERE / "inspection.json").write_text(json.dumps(reports, indent=2) + "\n")

    for name, r in reports.items():
        print(f"===== {name}")
        print(f"  activity table rows      {r['table_rows']:>7}  "
              f"(unique IDs {r['table_unique_ids']})")
        print(f"  FASTA sequences          {r['fasta_records']:>7}  "
              f"({r['fasta_combined_headers']} combined headers, "
              f"{r['fasta_ids_after_expanding']} IDs once expanded)")
        print(f"  IDs present in both      {r['ids_in_both']:>7}  "
              f"(was {r['ids_in_both_before_expanding']} before expanding)")
        print(f"  table rows w/o sequence  {r['table_ids_without_sequence']:>7}  "
              f"e.g. {r['examples_table_only'][:2]}")
        print(f"  sequences w/o table row  {r['sequences_without_table_row']:>7}  "
              f"e.g. {r['examples_fasta_only'][:2]}")
        print(f"  sequence lengths         {r['sequence_lengths']}")
        print(f"  non-ACGT / duplicates    {r['sequences_with_non_acgt']} / "
              f"{r['duplicate_sequences']}")
        print(f"  joined rows with activity {r['joined_rows_with_activity']}")
        a = r["log2FoldChange"]
        if a:
            print(f"  log2FoldChange           min {a['min']:.2f}  median "
                  f"{a['median']:.2f}  max {a['max']:.2f}  sd {a['sd']:.2f}")
        print("  activity by library group (n, median log2FC, share above 1):")
        for project, s in r["activity_by_project"].items():
            print(f"    {project:26} {s['n']:>6}  {s['median_log2FoldChange']:+6.2f}"
                  f"  {s['fraction_above_1']:.0%}")
        regions = r.get("mutagenesis_source_regions")
        if regions:
            print(f"  source regions           "
                  f"{regions['entries_derived_from_a_region']} entries come "
                  f"from {regions['distinct_source_regions']} regions "
                  f"({regions['entries_per_region_min']}-"
                  f"{regions['entries_per_region_max']} each)")
        print(f"  variants                 {r['distinct_variant_ids']} ids, "
              f"{r['variants_tested_in_several_configurations']} tested in "
              f"several configurations; windows {r['window_values']}; "
              f"strands {r['strand_values']}")
        print(f"  ref/alt pairs measured   "
              f"{r['reference_alternate_pairs_with_activity']}  "
              f"{r['pair_sequence_differences']}")
        print()
    print("wrote inspection.json")


if __name__ == "__main__":
    main()
