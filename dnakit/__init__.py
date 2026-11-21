"""dnakit — read DNA from sequence files and write it back out.

Reading: parse FASTA, GenBank ORIGIN blocks and the UCI promoter CSV, then
interpret the sequence with the standard genetic code (transcription,
translation, reading frames, strand handling) and with the positional model
trained in experiments/001.

Writing: serialise sequence records to FASTA. That is file output, not
sequence design for synthesis.

Everything in `sequence` is deterministic and hand-checkable. Everything in
`promoter` is a model output carrying the limits recorded in STATUS.md, and
is kept in a separate module so the two are never confused.
"""
from .io import (Record, format_fasta, read, read_fasta, read_genbank,
                 read_plain, read_uci_promoters, write_fasta)
from .sequence import (CODON_TABLE, DNA_BASES, STOP_CODONS, base_counts,
                       clean, codons, describe, find_orfs, gc_content,
                       reverse_complement, transcribe, translate, validate)

__version__ = "0.1.0"

__all__ = [
    "CODON_TABLE", "DNA_BASES", "Record", "STOP_CODONS", "base_counts",
    "clean", "codons", "describe", "find_orfs", "format_fasta", "gc_content",
    "read", "read_fasta", "read_genbank", "read_plain", "read_uci_promoters",
    "reverse_complement", "transcribe", "translate", "validate",
    "write_fasta", "__version__",
]
