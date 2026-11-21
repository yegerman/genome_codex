"""Read a stretch of DNA and say, in plain words, what it does.

The analogy this module leans on: DNA is instructions, so it can be read the
way source code is read. A promoter is the condition that decides whether the
routine runs at all; a ribosome binding site is the entry point; ATG is where
the routine begins; each codon is one instruction naming one building block;
a stop codon is the return.

The analogy is a reading aid and it has limits worth stating plainly. A cell
is not a processor: the same stretch can be read in several frames, on either
strand, at different rates in different cell types, and most DNA does not
encode a protein at all. Nothing here measures what a sequence actually does
in a living cell; it reports what the sequence *says* under the standard
genetic code, plus which documented signals appear in it.

Three grades of certainty run through the output, and they are never mixed:

  certain  a fact about the string under the published genetic code, e.g.
           this codon is ATG, which begins a protein and codes methionine.
  pattern  a documented short signal matched here, which may be coincidence;
           reported alongside how many matches chance alone would give.
  model    a score from the logistic model in experiments/001, with the
           limits recorded in STATUS.md.
"""
from . import motifs
from .motifs import ANY, BACTERIA, EUKARYOTE
from .sequence import (AMINO_ACID_NAMES, CODON_TABLE, STOP_CODONS, codons,
                       find_orfs, gc_content, reverse_complement, translate,
                       validate)

CERTAIN, PATTERN, MODEL = "certain", "pattern", "model"


def _instruction(codon, index, total):
    """Plain words for one codon, read as one instruction."""
    amino = CODON_TABLE.get(codon, "X")
    if codon == "ATG" and index == 0:
        return "START — begin building the protein here", "Met", amino
    if codon in STOP_CODONS:
        return "STOP — the protein is finished and released", "---", amino
    if amino == "X":
        return "unreadable — this codon holds an ambiguous letter", "???", amino
    name = AMINO_ACID_NAMES.get(amino, amino)
    where = "first" if index == 0 else ("last" if index == total - 1 else "")
    prefix = f"{where} building block: " if where else "add "
    return f"{prefix}{name}", name, amino


def listing(sequence, frame=0, start=None, end=None):
    """Render DNA as an annotated instruction listing, like a code listing.

    Returns a list of rows; `as_text` renders them. Coordinates are absolute
    positions in the sequence given, so a row can be found in the original.
    """
    seq = validate(sequence, allow_ambiguous=True)
    begin = frame if start is None else start
    finish = len(seq) if end is None else end
    region = seq[begin:finish]
    triplets = codons(region, frame=0)

    rows = []
    for index, codon in enumerate(triplets):
        meaning, name, amino = _instruction(codon, index, len(triplets))
        rows.append({
            "position": begin + index * 3,
            "codon": codon,
            "amino_acid": amino,
            "amino_acid_name": name,
            "means": meaning,
            "certainty": CERTAIN,
        })
    return rows


def listing_as_text(rows, max_rows=None, show_all=False):
    """Format `listing` rows as a plain-text listing with a header."""
    shown = rows if show_all or not max_rows else rows[:max_rows]
    lines = [f"{'pos':>6}  {'codon':<6} {'aa':<4} what this instruction does",
             f"{'-' * 6}  {'-' * 6} {'-' * 4} {'-' * 44}"]
    for row in shown:
        lines.append(f"{row['position']:>6}  {row['codon']:<6} "
                     f"{row['amino_acid']:<4} {row['means']}")
    hidden = len(rows) - len(shown)
    if hidden > 0:
        lines.append(f"{'':>6}  ... {hidden} more instructions "
                     f"(pass --all to see them)")
    return "\n".join(lines)


def annotated_sequence(sequence, hits, width=60):
    """Print the DNA in rows with the signals marked underneath.

    Each marked signal gets a letter; the legend below says what each letter
    is. This is the 'comments beside the code' view.
    """
    seq = validate(sequence, allow_ambiguous=True)
    labels, legend = {}, []
    for hit in hits:
        name = hit["motif"]
        if name not in labels:
            labels[name] = chr(ord("a") + len(labels) % 26)
            legend.append((labels[name], name, hit["plain"]))

    marks = [" "] * len(seq)
    for hit in hits:
        for position in range(hit["start"], min(hit["end"], len(seq))):
            marks[position] = labels[hit["motif"]]

    lines = []
    for offset in range(0, len(seq), width):
        chunk = seq[offset:offset + width]
        mark_chunk = "".join(marks[offset:offset + width])
        lines.append(f"{offset:>6}  {chunk}")
        if mark_chunk.strip():
            lines.append(f"{'':>6}  {mark_chunk}")
    if legend:
        lines.append("")
        lines.append("  legend:")
        for letter, name, plain in legend:
            lines.append(f"    {letter}  {name} — {plain}")
    return "\n".join(lines)


def _overview(seq):
    return {
        "length_nt": len(seq),
        "gc_content": gc_content(seq),
        "max_instructions": len(seq) // 3,
        "reading_frames_possible": 6 if len(seq) >= 3 else 0,
    }


def _describe_orf(orf, total_length):
    """Plain words for one open reading frame."""
    span = f"{orf['start']}..{orf['end']}"
    strand = ("the given strand" if orf["strand"] == "+"
              else "the opposite strand")
    covers = orf["length_nt"] / total_length if total_length else 0
    return {
        "span": span,
        "start": orf["start"],
        "end": orf["end"],
        "strand": orf["strand"],
        "length_nt": orf["length_nt"],
        "length_aa": orf["length_aa"],
        "protein": orf["protein"],
        "fraction_of_sequence": covers,
        "certainty": CERTAIN,
        "plain": (f"positions {span} on {strand} read as a complete recipe: "
                  f"START, then {orf['length_aa']} building blocks, then STOP"),
    }


def explain(sequence, organism=ANY, min_aa=25, both_strands=True,
            use_model=False):
    """Read a sequence and return a structured, plain-words explanation.

    `organism` restricts which control signals are considered, because a
    bacterial promoter pattern appearing in human DNA is almost certainly
    coincidence. `use_model` additionally scores 57-base windows with the
    experiments/001 model, whose limits are attached to the result.
    """
    seq = validate(sequence, allow_ambiguous=True)
    overview = _overview(seq)

    orfs = [_describe_orf(orf, len(seq))
            for orf in find_orfs(seq, min_aa=min_aa, both_strands=both_strands)]

    try:
        signal_hits = motifs.find_all(seq, organism=organism,
                                      both_strands=both_strands,
                                      skip_codons=True)
    except ValueError:
        # Ambiguous letters make exact motif matching meaningless.
        signal_hits = []
    signals = motifs.summarise(signal_hits)

    result = {
        "overview": overview,
        "organism_assumed": organism,
        "open_reading_frames": orfs,
        "signals": signals,
        "signal_hits": signal_hits,
        "certainty_key": {
            CERTAIN: "a fact about the sequence under the standard genetic code",
            PATTERN: "a documented signal matched here; may be coincidence",
            MODEL: "a score from the experiments/001 model, see STATUS.md",
        },
    }

    if use_model:
        result["model"] = _model_section(seq)

    result["narrative"] = narrate(result)
    return result


def _model_section(seq):
    from . import promoter
    if len(seq) < promoter.WINDOW:
        return {"available": False,
                "reason": f"sequence is shorter than the "
                          f"{promoter.WINDOW}-base window the model reads"}
    try:
        hits = promoter.scan(seq, both_strands=True, top=3)
    except ValueError as error:
        return {"available": False, "reason": str(error)}
    return {
        "available": True,
        "certainty": MODEL,
        "scope": promoter.SCOPE,
        "top_windows": hits,
    }


def narrate(result):
    """Turn the structured reading into sentences anyone can follow."""
    overview = result["overview"]
    lines = [
        f"This is {overview['length_nt']} letters of DNA. Read as "
        f"instructions, that is at most {overview['max_instructions']} "
        f"instructions long, and it can be read in "
        f"{overview['reading_frames_possible']} different ways: three "
        f"starting points, on each of the two strands.",
        f"Its composition is {overview['gc_content']:.0%} G and C. "
        + ("That is GC-rich, which makes the double helix harder to pull "
           "apart." if overview["gc_content"] > 0.6 else
           "That is AT-rich, which makes the double helix easier to pull "
           "apart." if overview["gc_content"] < 0.4 else
           "That is a fairly even mix."),
    ]

    orfs = result["open_reading_frames"]
    if orfs:
        best = orfs[0]
        lines.append(
            f"The clearest thing here is a complete recipe for a protein. "
            f"{best['plain'].capitalize()}. In computer terms: a routine with "
            f"a defined beginning and end, {best['length_aa']} steps long.")
        if len(orfs) > 1:
            lines.append(
                f"There are {len(orfs)} such complete recipes in total. "
                f"Overlapping ones are normal: the same letters can be read "
                f"in more than one way, which is a thing DNA does and source "
                f"code does not.")
    else:
        lines.append(
            "No complete protein recipe is present: nothing here runs from a "
            "START instruction to a STOP instruction. Either this stretch "
            "does not code for a protein, or you are looking at only part of "
            "one and the ends are outside the piece given.")

    believable = [s for s in result["signals"] if s["beats_chance"]]
    coincidental = [s for s in result["signals"] if not s["beats_chance"]]

    if believable:
        lines.append("Control signals found more often than chance would "
                     "explain:")
        for signal in believable:
            where = ", ".join(str(p) for p in signal["positions"][:6])
            more = "" if len(signal["positions"]) <= 6 else ", ..."
            lines.append(
                f"  - {signal['motif']} at position {where}{more}: "
                f"{signal['plain']}. Found {signal['found']}, chance alone "
                f"would give about {signal['expected_by_chance']:.1f} "
                f"(coincidence {signal['chance_probability']:.1%}).")
    if coincidental:
        names = ", ".join(
            f"{s['motif']} ({s['found']} hit(s), coincidence "
            f"{s['chance_probability']:.0%})" for s in coincidental)
        lines.append(
            f"Matched here, but no more often than chance predicts, so these "
            f"are not findings: {names}.")
        if overview["length_nt"] < 500:
            lines.append(
                "That is a statement about statistical power, not proof the "
                "signal is absent. A sequence this short cannot show any "
                "single short pattern beats chance, even a real one. Testing "
                "across many sequences can: experiments/005 found the -10 "
                "box in 74% of known promoters against 6% of non-promoters.")
    if not result["signals"]:
        lines.append(
            "No known control signals were found. Short patterns like these "
            "are only searched for in the organism assumed, and most "
            "regulatory DNA is not a short fixed pattern at all.")

    if result["organism_assumed"] == ANY and result["signals"]:
        kinds = {s["organism"] for s in result["signals"]}
        if {BACTERIA, EUKARYOTE} & kinds:
            lines.append(
                "No organism was specified, so signals from both bacterial "
                "and eukaryotic machinery were searched. They do not belong "
                "in the same sequence: a bacterial signal matched inside "
                "human DNA is coincidence, not a finding. Narrow it with "
                "--organism.")

    model = result.get("model")
    if model and model.get("available"):
        best = model["top_windows"][0]
        lines.append(
            f"A trained model also scored this sequence. Its strongest "
            f"window is {best['start']}..{best['end']} on the "
            f"{'given' if best['strand'] == '+' else 'opposite'} strand "
            f"(score {best['logit']:+.2f}). Treat this as a hint only: "
            f"{model['scope']}.")

    lines.append(
        "What this reading cannot tell you: whether any of it is actually "
        "switched on in a living cell, in which cell types, or how much "
        "protein results. Those are measurements, not readings.")
    return lines
