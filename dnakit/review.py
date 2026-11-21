"""Code review for DNA: check its syntax, and diff a variant against a reference.

DNA has real syntax, and breaking it has consequences that read exactly like
software defects:

  reading frame   codons are read in threes from a fixed start. Insert or
                  delete anything not a multiple of three and every
                  instruction after it is read wrong. This is an off-by-one
                  in a parser, and it is the most destructive class of change.
  start and stop  a protein-coding stretch must open with ATG and close with
                  a stop codon. Missing either is an unterminated block.
  GT-AG rule      introns begin GT and end AG. The spliceosome relies on it
                  the way a parser relies on delimiters.

What this module does: report what a sequence says, and what a change to it
would do to the protein under the standard genetic code. That is a diff and a
linter.

What it does not do, deliberately and by the scope in CLAUDE.md: propose
edits to anyone's genome, design a correction, or suggest that a difference
found here is a disease or a target. Naming the reference letter at a
position is a statement about two strings, not a treatment.
"""
from dataclasses import dataclass, field

from .sequence import (CODON_TABLE, STOP_CODONS, codons, translate, validate)

ERROR, WARNING, NOTE = "error", "warning", "note"

# Ordered most to least disruptive to the resulting protein.
CONSEQUENCE_RANK = {
    "frameshift": 0,
    "start lost": 1,
    "nonsense": 2,
    "stop lost": 3,
    "in-frame deletion": 4,
    "in-frame insertion": 4,
    "missense": 5,
    "synonymous": 6,
    "identical": 7,
}


@dataclass
class Finding:
    """One diagnostic, in the shape a linter would emit."""
    severity: str
    rule: str
    position: int
    message: str
    plain: str
    detail: str = ""

    def as_dict(self):
        return {"severity": self.severity, "rule": self.rule,
                "position": self.position, "message": self.message,
                "plain": self.plain, "detail": self.detail}


def lint(sequence, expect_coding=True):
    """Check a sequence's syntax as if it were a protein-coding stretch.

    Returns findings ordered by position. `expect_coding=False` skips the
    rules that only apply to a coding sequence, leaving the checks that hold
    for any DNA.
    """
    seq = validate(sequence, allow_ambiguous=True)
    findings = []

    ambiguous = [i for i, base in enumerate(seq) if base not in "ACGT"]
    if ambiguous:
        findings.append(Finding(
            WARNING, "ambiguous-base", ambiguous[0],
            f"{len(ambiguous)} ambiguous letter(s), first at {ambiguous[0]}",
            "Some letters are not a definite A, C, G or T, so any "
            "instruction containing one cannot be read.",
            "Ambiguity codes record uncertainty in the source data."))

    if not expect_coding:
        return sorted(findings, key=lambda f: f.position)

    if len(seq) % 3 != 0:
        findings.append(Finding(
            ERROR, "frame-length", len(seq) - len(seq) % 3,
            f"length {len(seq)} is not a multiple of 3 "
            f"({len(seq) % 3} letter(s) left over)",
            "Instructions are read three letters at a time, so a leftover "
            "piece means the sequence is cut off mid-instruction.",
            "Either this is a fragment, or the reading frame is wrong."))

    if not seq.startswith("ATG"):
        findings.append(Finding(
            ERROR, "missing-start", 0,
            f"does not begin with ATG (begins with {seq[:3]!r})",
            "A protein recipe has to open with the START instruction.",
            "If this is a fragment, that is expected."))

    triplets = codons(seq)
    if not triplets:
        return sorted(findings, key=lambda f: f.position)

    if triplets[-1] not in STOP_CODONS:
        findings.append(Finding(
            ERROR, "missing-stop", (len(triplets) - 1) * 3,
            f"does not end with a stop codon (ends with {triplets[-1]!r})",
            "Without a STOP instruction the machinery keeps reading past "
            "the end of the recipe.",
            "In code terms: a block that is never closed."))

    for index, codon in enumerate(triplets[:-1]):
        if codon in STOP_CODONS:
            findings.append(Finding(
                ERROR, "premature-stop", index * 3,
                f"stop codon {codon} at instruction {index + 1} of "
                f"{len(triplets)}",
                f"The recipe stops at step {index + 1} even though it "
                f"continues for {len(triplets) - index - 1} more steps. The "
                f"protein would be cut short.",
                "In code terms: an early return that skips the rest."))
            break

    return sorted(findings, key=lambda f: f.position)


def _trim_common(reference, variant):
    """Longest common prefix and suffix, to localise a single change."""
    prefix = 0
    limit = min(len(reference), len(variant))
    while prefix < limit and reference[prefix] == variant[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < limit - prefix
           and reference[len(reference) - 1 - suffix]
           == variant[len(variant) - 1 - suffix]):
        suffix += 1
    return prefix, suffix


def compare(reference, variant, protein_numbering_note=True):
    """Diff a variant against a reference coding sequence.

    This is a simple comparison, not a sequence aligner: it localises one
    change by trimming the shared start and end. Several scattered
    substitutions are reported individually; several separate indels are
    beyond what it can resolve and are reported as a single span.

    Amino-acid positions are 1-based counting the initiator methionine as 1.
    Clinical literature often numbers the mature protein after that
    methionine is removed, which shifts positions by one, so compare
    conventions before matching a result to a published name.
    """
    ref = validate(reference, allow_ambiguous=True)
    var = validate(variant, allow_ambiguous=True)

    result = {
        "reference_length": len(ref),
        "variant_length": len(var),
        "length_change": len(var) - len(ref),
        "changes": [],
        "consequence": "identical",
        "plain": "",
        "reference_protein": translate(ref),
        "variant_protein": translate(var),
    }
    if protein_numbering_note:
        result["numbering"] = ("amino acids numbered from 1 at the initiator "
                               "methionine; clinical names often number the "
                               "mature protein instead, a shift of one")

    if ref == var:
        result["plain"] = "The two sequences are identical."
        return result

    prefix, suffix = _trim_common(ref, var)
    delta = len(var) - len(ref)

    if delta != 0:
        changed_from = ref[prefix:len(ref) - suffix]
        changed_to = var[prefix:len(var) - suffix]
        kind = ("insertion" if delta > 0 else "deletion")
        if delta % 3 != 0:
            consequence = "frameshift"
            plain = (
                f"{abs(delta)} letter(s) {kind[:-3]}ed at position {prefix}. "
                f"Because {abs(delta)} is not a multiple of 3, every "
                f"instruction after this point is read in the wrong grouping, "
                f"so the rest of the protein is unrelated to the original and "
                f"usually ends early. This is the most destructive kind of "
                f"change: in code terms, an off-by-one that corrupts "
                f"everything downstream.")
        else:
            consequence = f"in-frame {kind}"
            steps = abs(delta) // 3
            plain = (
                f"{abs(delta)} letters {kind[:-3]}ed at position {prefix}. "
                f"Because that is a multiple of 3, the grouping survives: "
                f"{steps} whole instruction(s) are {kind[:-3]}ed and the rest "
                f"of the protein is unchanged.")
        result["changes"].append({
            "type": kind, "position": prefix,
            "reference": changed_from, "variant": changed_to,
            "length_change": delta,
        })
        result["consequence"] = consequence
        result["plain"] = plain
        return result

    # Same length: compare instruction by instruction.
    ref_codons, var_codons = codons(ref), codons(var)
    worst = "identical"
    descriptions = []
    for index, (rc, vc) in enumerate(zip(ref_codons, var_codons)):
        if rc == vc:
            continue
        ref_aa = CODON_TABLE.get(rc, "X")
        var_aa = CODON_TABLE.get(vc, "X")
        aa_position = index + 1
        if ref_aa == var_aa:
            kind = "synonymous"
            plain = (f"instruction {aa_position}: {rc} became {vc}, but both "
                     f"mean {ref_aa}. The protein is unchanged. The genetic "
                     f"code is redundant, so some letter changes are "
                     f"invisible in the output.")
        elif var_aa == "*":
            kind = "nonsense"
            remaining = len(ref_codons) - aa_position
            plain = (f"instruction {aa_position}: {rc} ({ref_aa}) became {vc}, "
                     f"a STOP. The protein ends {remaining} steps early. In "
                     f"code terms: an early return.")
        elif ref_aa == "*":
            kind = "stop lost"
            plain = (f"instruction {aa_position}: the STOP became {vc} "
                     f"({var_aa}), so the machinery reads past the intended "
                     f"end. In code terms: a block that no longer closes.")
        elif aa_position == 1 and ref_aa == "M":
            kind = "start lost"
            plain = (f"instruction 1: the START codon {rc} became {vc}, so "
                     f"there is no longer a defined place to begin.")
        else:
            kind = "missense"
            plain = (f"instruction {aa_position}: {rc} ({ref_aa}) became {vc} "
                     f"({var_aa}). One building block is swapped for a "
                     f"different one; the protein is the same length. Whether "
                     f"that matters depends on what that block does, which "
                     f"this comparison cannot tell you.")

        descriptions.append(plain)
        result["changes"].append({
            "type": kind,
            "position": index * 3,
            "codon_number": aa_position,
            "reference_codon": rc, "variant_codon": vc,
            "reference_aa": ref_aa, "variant_aa": var_aa,
            "notation": f"{ref_aa}{aa_position}{var_aa}",
        })
        if CONSEQUENCE_RANK[kind] < CONSEQUENCE_RANK[worst]:
            worst = kind

    result["consequence"] = worst
    result["plain"] = " ".join(descriptions)
    return result


def review(sequence, reference=None, expect_coding=True):
    """A full review: syntax findings, plus a diff when a reference is given."""
    report = {
        "findings": [f.as_dict() for f in lint(sequence, expect_coding)],
        "comparison": None,
    }
    if reference is not None:
        report["comparison"] = compare(reference, sequence)
    counts = {ERROR: 0, WARNING: 0, NOTE: 0}
    for finding in report["findings"]:
        counts[finding["severity"]] += 1
    report["counts"] = counts
    report["clean"] = counts[ERROR] == 0
    return report
