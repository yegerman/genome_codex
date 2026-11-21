"""Read DNA out of sequence files, and write it back in standard formats.

Supported for reading: FASTA, the ORIGIN block of GenBank flat files, the
UCI promoter CSV used by experiments 001 and 002, and plain sequence text.
Supported for writing: FASTA.

"Writing DNA" here means serialising sequence records to a file that other
tools can read. It is not sequence design for synthesis.

A record is a small, immutable object: identifier, description, sequence.
"""
import csv
from dataclasses import dataclass, replace
from pathlib import Path

from .sequence import clean, describe, reverse_complement, validate


@dataclass(frozen=True)
class Record:
    """One named sequence, as it appeared in its source file."""
    id: str
    sequence: str
    description: str = ""
    source: str = ""

    def __len__(self):
        return len(self.sequence)

    def validated(self, allow_ambiguous=True):
        """Return a copy whose sequence is checked, or raise ValueError."""
        return replace(self, sequence=validate(self.sequence, allow_ambiguous))

    def reverse_complemented(self):
        """The same record read from the other strand."""
        return replace(self, sequence=reverse_complement(self.sequence),
                       description=(self.description + " (reverse complement)").strip())

    def summary(self):
        return {"id": self.id, "description": self.description,
                "source": self.source, **describe(self.sequence)}


def read_fasta(path):
    """Parse a FASTA file into records.

    The identifier is the first whitespace-delimited word after '>', the rest
    of that line is the description. An empty record raises ValueError rather
    than passing a zero-length sequence downstream.
    """
    path = Path(path)
    records, header, chunks = [], None, []

    def flush():
        if header is None:
            return
        ident, _, description = header.partition(" ")
        sequence = clean("".join(chunks))
        if not sequence:
            raise ValueError(f"{path}: record {ident!r} holds no sequence")
        records.append(Record(id=ident, sequence=sequence,
                              description=description.strip(), source=str(path)))

    with path.open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                header, chunks = line[1:].strip(), []
            elif line.strip():
                if header is None:
                    raise ValueError(f"{path}: sequence data before any '>' header")
                chunks.append(line)
    flush()

    if not records:
        raise ValueError(f"{path}: no FASTA records found")
    return records


def read_genbank(path):
    """Read the ORIGIN block of a GenBank flat file.

    Only the sequence and its LOCUS/ACCESSION/DEFINITION identifiers are
    read; feature annotations are deliberately not parsed, so nothing in the
    file's own commentary is passed off as something this code verified.
    """
    path = Path(path)
    locus = accession = ""
    definition_parts, sequence_parts = [], []
    in_origin = in_definition = False

    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if in_origin:
                if stripped == "//":
                    in_origin = False
                    continue
                sequence_parts.append(clean(stripped))
                continue
            if line.startswith("ORIGIN"):
                in_origin, in_definition = True, False
            elif line.startswith("LOCUS"):
                parts = line.split()
                locus = parts[1] if len(parts) > 1 else ""
            elif line.startswith("ACCESSION"):
                parts = line.split()
                accession = parts[1] if len(parts) > 1 else ""
            elif line.startswith("DEFINITION"):
                in_definition = True
                definition_parts.append(stripped[len("DEFINITION"):].strip())
            elif in_definition and line.startswith(" "):
                definition_parts.append(stripped)
            else:
                in_definition = False

    sequence = "".join(sequence_parts)
    if not sequence:
        raise ValueError(f"{path}: no ORIGIN sequence block found")
    return [Record(id=accession or locus or path.stem,
                   sequence=sequence,
                   description=" ".join(definition_parts).rstrip("."),
                   source=str(path))]


def read_uci_promoters(path):
    """Read the UCI promoter CSV: class label, identifier, 57-base sequence.

    The label is kept in the description as `class=+` or `class=-`, since it
    is an annotation from the dataset's authors rather than a property this
    code measured.
    """
    path = Path(path)
    records = []
    with path.open() as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected 3 fields, found {len(row)}")
            label, identifier, sequence = (field.strip() for field in row)
            records.append(Record(id=identifier, sequence=clean(sequence),
                                  description=f"class={label}", source=str(path)))
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def read_plain(path):
    """Read a file holding nothing but sequence text."""
    path = Path(path)
    sequence = clean(path.read_text())
    if not sequence:
        raise ValueError(f"{path}: file holds no sequence characters")
    return [Record(id=path.stem, sequence=sequence, source=str(path))]


READERS = {
    ".fa": read_fasta, ".fasta": read_fasta, ".fna": read_fasta,
    ".ffn": read_fasta, ".frn": read_fasta,
    ".gb": read_genbank, ".gbk": read_genbank, ".genbank": read_genbank,
    ".csv": read_uci_promoters,
    ".txt": read_plain, ".seq": read_plain, ".data": read_uci_promoters,
}


def read(path):
    """Read any supported file, choosing the parser by extension."""
    suffix = Path(path).suffix.lower()
    try:
        reader = READERS[suffix]
    except KeyError:
        raise ValueError(
            f"{path}: unsupported extension {suffix!r}. "
            f"Supported: {', '.join(sorted(READERS))}") from None
    return reader(path)


def format_fasta(records, line_width=60):
    """Render records as FASTA text, wrapped at `line_width` bases."""
    if line_width < 1:
        raise ValueError("line_width must be at least 1")
    if isinstance(records, Record):
        records = [records]
    lines = []
    for record in records:
        header = f">{record.id}"
        if record.description:
            header += f" {record.description}"
        lines.append(header)
        sequence = record.sequence
        lines.extend(sequence[i:i + line_width]
                     for i in range(0, len(sequence), line_width))
    return "\n".join(lines) + "\n"


def write_fasta(records, path, line_width=60):
    """Write records to a FASTA file and return how many were written."""
    if isinstance(records, Record):
        records = [records]
    records = list(records)
    if not records:
        raise ValueError("refusing to write a FASTA file with no records")
    Path(path).write_text(format_fasta(records, line_width))
    return len(records)
