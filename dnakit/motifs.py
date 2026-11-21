"""Short DNA signals with documented meanings, and how often they lie.

A motif is a short consensus sequence that the cell's machinery recognises.
Finding one is a string match, which is certain; concluding that it *is* that
signal is not, because a six-letter pattern turns up by chance roughly once
every few hundred bases.

So every search here reports two numbers: how many matches were found, and
how many a random sequence of the same length would be expected to contain.
A finding that does not beat chance is reported as such rather than dressed
up, which is the whole point of this module.

Consensus sequences below are the textbook ones. Each carries the organism it
applies to: a bacterial promoter signal found in human DNA is almost
certainly coincidence, and vice versa.
"""
from dataclasses import dataclass
from math import comb, exp, factorial

from .sequence import DNA_BASES, reverse_complement, validate

BACTERIA = "bacteria"
EUKARYOTE = "eukaryote"
ANY = "any"


@dataclass(frozen=True)
class Motif:
    """One recognised signal: what it reads as, and what it means."""
    name: str
    consensus: str
    max_mismatches: int
    organism: str
    plain: str       # what it does, in plain words
    detail: str      # the mechanism, one sentence
    typical_position: str = ""


CATALOG = (
    Motif("start codon", "ATG", 0, ANY,
          "START — begin building the protein here",
          "The ribosome starts translating at this codon, which also codes "
          "for the amino acid methionine.",
          "the first codon of a protein-coding stretch"),
    Motif("stop codon TAA", "TAA", 0, ANY,
          "STOP — the protein ends here",
          "No amino acid matches this codon, so the ribosome releases the "
          "finished protein.",
          "the last codon of a protein-coding stretch"),
    Motif("stop codon TAG", "TAG", 0, ANY,
          "STOP — the protein ends here",
          "No amino acid matches this codon, so the ribosome releases the "
          "finished protein.",
          "the last codon of a protein-coding stretch"),
    Motif("stop codon TGA", "TGA", 0, ANY,
          "STOP — the protein ends here",
          "No amino acid matches this codon, so the ribosome releases the "
          "finished protein.",
          "the last codon of a protein-coding stretch"),
    Motif("-35 box", "TTGACA", 1, BACTERIA,
          "SWITCH, part 1 — where the copying machine first grips",
          "One of the two elements a bacterial sigma factor recognises when "
          "positioning RNA polymerase.",
          "about 35 bases before the start of the copy"),
    Motif("-10 box (Pribnow box)", "TATAAT", 1, BACTERIA,
          "SWITCH, part 2 — where the two DNA strands are prised apart",
          "This AT-rich element is easier to unwind, so it is where the "
          "double helix opens for copying.",
          "about 10 bases before the start of the copy"),
    Motif("Shine-Dalgarno sequence", "AGGAGG", 1, BACTERIA,
          "DOCK — where the ribosome attaches before reading",
          "Pairs with the 16S ribosomal RNA to position the ribosome on the "
          "message.",
          "a few bases before a bacterial start codon"),
    Motif("TATA box", "TATAAA", 1, EUKARYOTE,
          "SWITCH — where the copying machine is assembled",
          "Bound by the TATA-binding protein, which recruits the rest of the "
          "transcription machinery.",
          "about 25 to 35 bases before the start of the copy"),
    Motif("CAAT box", "CCAAT", 0, EUKARYOTE,
          "VOLUME — a control element that raises copying rate",
          "A binding site for transcription factors that strengthen a "
          "promoter.",
          "roughly 75 to 80 bases before the start of the copy"),
    Motif("GC box", "GGGCGG", 0, EUKARYOTE,
          "VOLUME — a control element that raises copying rate",
          "A binding site for the Sp1 transcription factor.",
          "upstream of many housekeeping genes"),
    Motif("poly(A) signal", "AATAAA", 0, EUKARYOTE,
          "END OF MESSAGE — cut the copy here and cap its tail",
          "Directs cleavage of the transcript and addition of the poly(A) "
          "tail that stabilises it.",
          "near the end of a gene's transcript"),
    Motif("splice donor (GT-AG rule)", "GTAAGT", 1, EUKARYOTE,
          "CUT FROM HERE — the start of a stretch removed before reading",
          "Introns begin with an invariant GT. The spliceosome cuts here, "
          "removes the intron and joins the coding pieces together.",
          "at the start of an intron"),
    Motif("Kozak core", "ACCATGG", 1, EUKARYOTE,
          "START, emphasised — a strongly recognised beginning",
          "The bases immediately around a start codon affect how reliably "
          "the ribosome begins there.",
          "surrounding a eukaryotic start codon"),
)

BY_NAME = {motif.name: motif for motif in CATALOG}


def neighbourhood_size(length, max_mismatches):
    """How many distinct strings lie within `max_mismatches` of a pattern."""
    if not 0 <= max_mismatches <= length:
        raise ValueError("max_mismatches must be between 0 and the length")
    return sum(comb(length, i) * 3 ** i for i in range(max_mismatches + 1))


def expected_by_chance(motif, sequence_length, both_strands=True):
    """Matches expected in random DNA of this length, bases equally likely.

    This is the number a finding has to beat before it means anything. The
    estimate assumes a uniform base composition; real DNA is skewed, so treat
    it as an order of magnitude rather than a p-value.
    """
    width = len(motif.consensus)
    if sequence_length < width:
        return 0.0
    positions = sequence_length - width + 1
    probability = (neighbourhood_size(width, motif.max_mismatches)
                   / len(DNA_BASES) ** width)
    return positions * probability * (2 if both_strands else 1)


def chance_probability(found, expected):
    """P(at least `found` matches) when matches arrive at rate `expected`.

    Comparing a count to its mean is not enough: two matches where chance
    gives 0.8 sounds convincing and happens 19% of the time. This is the
    Poisson upper tail, which is the number that should decide whether a
    motif hit is worth reporting at all.
    """
    if found <= 0:
        return 1.0
    if expected <= 0:
        return 0.0
    below = sum(exp(-expected) * expected ** i / factorial(i)
                for i in range(found))
    return max(0.0, min(1.0, 1.0 - below))


# A hit has to be this unlikely by chance before it is called a real signal.
SIGNIFICANCE = 0.05


def _mismatches(text, pattern):
    return sum(1 for a, b in zip(text, pattern) if a != b)


def find(sequence, motif, both_strands=True):
    """Every position where `motif` matches, within its mismatch budget.

    Coordinates are 0-based and always on the forward strand, so a hit on the
    reverse strand reports the forward-strand span it covers.
    """
    forward = validate(sequence, allow_ambiguous=False)
    width = len(motif.consensus)
    hits = []
    strands = [("+", forward)]
    if both_strands:
        strands.append(("-", reverse_complement(forward)))

    for strand, text in strands:
        for start in range(len(text) - width + 1):
            window = text[start:start + width]
            mismatches = _mismatches(window, motif.consensus)
            if mismatches > motif.max_mismatches:
                continue
            start_fwd = start if strand == "+" else len(text) - start - width
            hits.append({
                "motif": motif.name,
                "consensus": motif.consensus,
                "match": window,
                "mismatches": mismatches,
                "strand": strand,
                "start": start_fwd,
                "end": start_fwd + width,
                "organism": motif.organism,
                "plain": motif.plain,
                "detail": motif.detail,
                "exact": mismatches == 0,
            })
    return sorted(hits, key=lambda hit: (hit["start"], hit["mismatches"]))


def find_all(sequence, organism=ANY, both_strands=True, skip_codons=False):
    """Search the whole catalog.

    `organism` filters to signals that apply there, keeping bacterial and
    eukaryotic machinery from being reported in the same breath. Pass
    skip_codons=True to leave out start and stop codons, which are better
    read as part of a reading frame than as isolated hits.
    """
    sequence = validate(sequence, allow_ambiguous=False)
    results = []
    for motif in CATALOG:
        if skip_codons and len(motif.consensus) == 3:
            continue
        if organism != ANY and motif.organism not in (organism, ANY):
            continue
        hits = find(sequence, motif, both_strands)
        if not hits:
            continue
        expected = expected_by_chance(motif, len(sequence), both_strands)
        probability = chance_probability(len(hits), expected)
        for hit in hits:
            hit["expected_by_chance"] = expected
            hit["chance_probability"] = probability
            hit["beats_chance"] = probability < SIGNIFICANCE
        results.extend(hits)
    return sorted(results, key=lambda hit: hit["start"])


def summarise(hits):
    """Group hits by motif, with counts against the chance expectation."""
    grouped = {}
    for hit in hits:
        entry = grouped.setdefault(hit["motif"], {
            "motif": hit["motif"],
            "consensus": hit["consensus"],
            "organism": hit["organism"],
            "plain": hit["plain"],
            "found": 0,
            "exact": 0,
            "expected_by_chance": hit.get("expected_by_chance", 0.0),
            "positions": [],
        })
        entry["found"] += 1
        entry["exact"] += int(hit["exact"])
        entry["positions"].append(hit["start"])
    for entry in grouped.values():
        entry["chance_probability"] = chance_probability(
            entry["found"], entry["expected_by_chance"])
        entry["beats_chance"] = entry["chance_probability"] < SIGNIFICANCE
    return sorted(grouped.values(),
                  key=lambda e: (e["chance_probability"], -e["found"]))
