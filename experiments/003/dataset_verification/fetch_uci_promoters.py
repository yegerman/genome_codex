#!/usr/bin/env python3
"""Fetch the official UCI promoter dataset and verify the local transcription.

The handoff records that `promoters_source.csv` was hand-transcribed from a
browser-visible copy of the UCI file and that byte-level identity to upstream
could NOT be verified in the previous offline runtime. This script closes that
gap: it downloads the official file, records its checksum, parses it with the
same normalisation the transcription used, and compares record-by-record.

Source: UCI Molecular Biology (Promoter Gene Sequences), id=67.
Harley, C., Reynolds, R., Noordewier, M. (1987). DOI 10.24432/C5S01D. CC BY 4.0.
"""
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "molecular-biology/promoter-gene-sequences/promoters.data")
OUT = Path(__file__).parent
LOCAL = OUT.parent.parent / "001" / "genetic_ml_pilot" / "promoters_source.csv"


def normalise(raw_text):
    """Parse the upstream file the way the local CSV was transcribed."""
    records = []
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"unexpected field count: {fields[:2]}")
        label, name, seq = fields
        records.append((label, name, "".join(seq.split()).lower()))
    return records


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "research-script"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()

    (OUT / "promoters.data").write_bytes(raw)
    upstream_sha = hashlib.sha256(raw).hexdigest()
    local_bytes = LOCAL.read_bytes()
    local_sha = hashlib.sha256(local_bytes).hexdigest()

    upstream = normalise(raw.decode("utf-8", errors="strict"))
    local = [(r[0], r[1], r[2].lower())
             for r in csv.reader(local_bytes.decode().splitlines()) if r]

    # Compare as ordered records, then as sets, so a reordering is reported
    # separately from a content difference.
    same_order = upstream == local
    mismatches = [
        {"index": i, "upstream": u, "local": l}
        for i, (u, l) in enumerate(zip(upstream, local)) if u != l
    ]
    upstream_only = sorted(set(upstream) - set(local))
    local_only = sorted(set(local) - set(upstream))

    report = {
        "source_url": URL,
        "upstream_sha256_raw_file": upstream_sha,
        "local_csv_sha256": local_sha,
        "upstream_n_records": len(upstream),
        "local_n_records": len(local),
        "upstream_label_counts": {
            lab: sum(1 for r in upstream if r[0] == lab)
            for lab in sorted({r[0] for r in upstream})
        },
        "identical_as_ordered_records": same_order,
        "n_record_mismatches": len(mismatches),
        "record_mismatches": mismatches[:20],
        "n_upstream_only": len(upstream_only),
        "n_local_only": len(local_only),
        "upstream_only_ids": [r[1] for r in upstream_only][:20],
        "local_only_ids": [r[1] for r in local_only][:20],
        "verdict": (
            "local transcription matches upstream record-for-record"
            if same_order and not upstream_only and not local_only
            else "DIFFERENCES FOUND - see mismatch fields"
        ),
        "note": (
            "Raw bytes differ by design: the local CSV strips tab/whitespace "
            "field padding present in the upstream file. The comparison is on "
            "normalised (label, id, sequence) records."
        ),
    }
    (OUT / "dataset_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k != "record_mismatches"}, indent=2))


if __name__ == "__main__":
    main()
