"""Core DNA sequence operations: reading a sequence as biological content.

Scope: this module interprets DNA that already exists in a file or database.
It transcribes, translates and locates reading frames using the standard
genetic code. It does not design sequences for synthesis and makes no claim
about the function or phenotype of anything it reads.

Every operation here is deterministic and checkable by hand, so results from
this module are computed facts rather than model predictions.
"""
from itertools import product

DNA_BASES = "ACGT"
# IUPAC ambiguity codes, accepted when reading real files but rejected by
# operations that need an unambiguous base.
IUPAC = set("ACGTURYSWKMBDHVN")

COMPLEMENT = str.maketrans("ACGTURYSWKMBDHVNacgturyswkmbdhvn",
                           "TGCAAYRSWMKVHDBNtgcaayrswmkvhdbn")

# The standard genetic code (NCBI translation table 1). Codons are generated
# in TCAG order so this table can be checked against the published one by
# reading the amino-acid string straight across.
_AMINO_ACIDS = ("FFLLSSSSYY**CC*W"
                "LLLLPPPPHHQQRRRR"
                "IIIMTTTTNNKKSSRR"
                "VVVVAAAADDEEGGGG")
CODON_TABLE = {"".join(codon): aa for codon, aa in
               zip(product("TCAG", repeat=3), _AMINO_ACIDS)}
assert len(CODON_TABLE) == 64

START_CODON = "ATG"
STOP_CODONS = frozenset(c for c, aa in CODON_TABLE.items() if aa == "*")

AMINO_ACID_NAMES = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    "*": "Stop",
}


def clean(sequence):
    """Strip whitespace, digits and gap characters; uppercase the rest.

    Sequence files carry line numbers, spacing and alignment gaps. Those are
    layout, not content, so they are removed before anything is interpreted.
    """
    return "".join(c for c in str(sequence).upper()
                   if not c.isspace() and not c.isdigit() and c not in ".-")


def validate(sequence, allow_ambiguous=False):
    """Return the cleaned sequence, or raise ValueError naming the offender.

    Silently dropping an unexpected character would change the reading frame
    and every downstream amino acid, so this refuses instead.
    """
    seq = clean(sequence)
    allowed = IUPAC if allow_ambiguous else set(DNA_BASES)
    for index, base in enumerate(seq):
        if base not in allowed:
            raise ValueError(
                f"position {index} holds {base!r}, which is not "
                f"{'an IUPAC code' if allow_ambiguous else 'A, C, G or T'}")
    return seq


def gc_content(sequence):
    """Fraction of G and C. Returns 0.0 for an empty sequence."""
    seq = validate(sequence, allow_ambiguous=True)
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def reverse_complement(sequence):
    """The other strand, read 5' to 3'.

    DNA is double stranded, so a feature absent from the given strand may sit
    on its partner. Reading only one strand misses half the molecule.
    """
    return validate(sequence, allow_ambiguous=True).translate(COMPLEMENT)[::-1]


def transcribe(sequence):
    """DNA to messenger RNA: thymine becomes uracil."""
    return validate(sequence, allow_ambiguous=True).replace("T", "U")


def codons(sequence, frame=0):
    """Split into whole three-base codons from `frame` (0, 1 or 2).

    A trailing partial codon is dropped, because two bases do not specify an
    amino acid.
    """
    if frame not in (0, 1, 2):
        raise ValueError(f"frame must be 0, 1 or 2, got {frame!r}")
    seq = validate(sequence, allow_ambiguous=True)[frame:]
    return [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def translate(sequence, frame=0, stop_at_stop=False):
    """Protein sequence in one-letter codes; '*' marks a stop codon.

    A codon holding an ambiguity code becomes 'X': unknown, not guessed.
    Set stop_at_stop=True to end translation at the first stop codon.
    """
    protein = []
    for codon in codons(sequence, frame):
        amino = CODON_TABLE.get(codon, "X")
        if amino == "*" and stop_at_stop:
            break
        protein.append(amino)
    return "".join(protein)


def find_orfs(sequence, min_aa=25, both_strands=True):
    """Locate open reading frames: ATG through to an in-frame stop codon.

    An ORF is a stretch that *could* encode a protein. Whether it is actually
    transcribed and translated in a living cell is a separate question this
    function cannot answer.

    Returns dicts sorted longest first, with `start`/`end` as 0-based
    half-open coordinates on the strand given here, always counted on the
    forward strand so they can be compared directly.
    """
    forward = validate(sequence, allow_ambiguous=True)
    found = []
    strands = [("+", forward)]
    if both_strands:
        strands.append(("-", reverse_complement(forward)))

    for strand, seq in strands:
        for frame in range(3):
            position = frame
            while position + 3 <= len(seq):
                if seq[position:position + 3] != START_CODON:
                    position += 3
                    continue
                for end in range(position + 3, len(seq) - 2, 3):
                    if seq[end:end + 3] in STOP_CODONS:
                        protein = translate(seq[position:end])
                        if len(protein) >= min_aa:
                            start_fwd = (position if strand == "+"
                                         else len(seq) - end - 3)
                            end_fwd = (end + 3 if strand == "+"
                                       else len(seq) - position)
                            found.append({
                                "strand": strand,
                                "frame": frame,
                                "start": start_fwd,
                                "end": end_fwd,
                                "length_nt": end + 3 - position,
                                "length_aa": len(protein),
                                "protein": protein,
                            })
                        position = end + 3
                        break
                else:
                    # No in-frame stop: an ORF running off the end of the
                    # fragment is not reported, since its extent is unknown.
                    break
    return sorted(found, key=lambda orf: orf["length_aa"], reverse=True)


def base_counts(sequence):
    """Count every base present, ambiguity codes included."""
    seq = validate(sequence, allow_ambiguous=True)
    return {base: seq.count(base) for base in sorted(set(seq))}


def describe(sequence):
    """A plain summary of what a sequence is, with nothing inferred."""
    seq = validate(sequence, allow_ambiguous=True)
    return {
        "length_nt": len(seq),
        "gc_content": gc_content(seq),
        "base_counts": base_counts(seq),
        "ambiguous_bases": sum(1 for b in seq if b not in DNA_BASES),
    }
