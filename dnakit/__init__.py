"""dnakit — read DNA, say what it does, and check it like source code.

Reading files: parse FASTA, GenBank ORIGIN blocks and the UCI promoter CSV.
Reading biology: transcription, translation, reading frames and strand
handling under the standard genetic code.
Explaining: `explain` narrates a sequence in plain words and renders it as an
annotated instruction listing.
Reviewing: `review` lints a sequence's syntax and diffs a variant against a
reference, classifying what a change does to the protein.
Writing: serialise records to FASTA. File output, not sequence design.

Three grades of certainty run through the package and are never mixed:
`sequence` and `io` are deterministic and hand-checkable; `motifs` reports
pattern matches with the odds of seeing them by chance; `promoter` is a model
output carrying the limits recorded in STATUS.md. They live in separate
modules for that reason.
"""
from .io import (Record, format_fasta, read, read_fasta, read_genbank,
                 read_plain, read_uci_promoters, write_fasta)
from .sequence import (CODON_TABLE, DNA_BASES, STOP_CODONS, base_counts,
                       clean, codons, describe, find_orfs, gc_content,
                       reverse_complement, transcribe, translate, validate)

__version__ = "0.2.0"

__all__ = [
    "CODON_TABLE", "DNA_BASES", "Record", "STOP_CODONS", "base_counts",
    "clean", "codons", "describe", "find_orfs", "format_fasta", "gc_content",
    "read", "read_fasta", "read_genbank", "read_plain", "read_uci_promoters",
    "reverse_complement", "transcribe", "translate", "validate",
    "write_fasta", "__version__",
]
