#!/usr/bin/env python3
"""Tests for dnakit, checked against independently known answers.

Run from the repository root:

    .venv/bin/python -I tests/test_dnakit.py

The translation cases use the published start of human beta-globin (HBB,
UniProt P68871), so a wrong codon table fails here rather than silently
producing a plausible-looking protein.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dnakit import io as dnaio          # noqa: E402
from dnakit import promoter             # noqa: E402
from dnakit import sequence as seq      # noqa: E402
from dnakit.io import Record            # noqa: E402

UCI_CSV = ROOT / "experiments" / "001" / "genetic_ml_pilot" / "promoters_source.csv"

# First 31 codons of the human HBB coding sequence.
HBB_CDS_START = ("ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAG"
                 "GTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGG")
HBB_PROTEIN_START = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGR"

_results = []


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}"
          + (f"\n        {detail}" if detail and not condition else ""))


def equal(name, got, want):
    check(name, got == want, f"got {got!r}, want {want!r}")


# --------------------------------------------------------------- genetic code
def test_codon_table():
    equal("codon table has 64 codons", len(seq.CODON_TABLE), 64)
    equal("three stop codons", sorted(seq.STOP_CODONS), ["TAA", "TAG", "TGA"])
    equal("ATG is methionine", seq.CODON_TABLE["ATG"], "M")
    equal("TGG is tryptophan", seq.CODON_TABLE["TGG"], "W")
    equal("TTT is phenylalanine", seq.CODON_TABLE["TTT"], "F")
    equal("GGG is glycine", seq.CODON_TABLE["GGG"], "G")
    equal("AGA is arginine", seq.CODON_TABLE["AGA"], "R")
    # Every codon maps to a single-letter code, and all 20 amino acids appear.
    equal("20 amino acids plus stop",
          len(set(seq.CODON_TABLE.values())), 21)


def test_translation():
    equal("HBB start translates correctly",
          seq.translate(HBB_CDS_START), HBB_PROTEIN_START)
    equal("translation stops at a stop codon",
          seq.translate("ATGAAATAAGGG", stop_at_stop=True), "MK")
    equal("stop shown as asterisk otherwise",
          seq.translate("ATGAAATAAGGG"), "MK*G")
    equal("partial trailing codon dropped",
          seq.translate("ATGAA"), "M")
    equal("frame 1 shifts the reading",
          seq.translate("GATGAAATAA", frame=1), "MK*")
    equal("ambiguous codon becomes X",
          seq.translate("ATGNNNAAA"), "MXK")
    equal("transcription replaces T with U",
          seq.transcribe("ATGC"), "AUGC")


def test_strands():
    equal("reverse complement of ATGC", seq.reverse_complement("ATGC"), "GCAT")
    equal("palindrome is its own revcomp",
          seq.reverse_complement("GAATTC"), "GAATTC")
    equal("double reverse complement is the original",
          seq.reverse_complement(seq.reverse_complement(HBB_CDS_START)),
          HBB_CDS_START)
    equal("N complements to N", seq.reverse_complement("ATGN"), "NCAT")


def test_composition():
    equal("GC of GGCC", seq.gc_content("GGCC"), 1.0)
    equal("GC of ATAT", seq.gc_content("ATAT"), 0.0)
    equal("GC of ATGC", seq.gc_content("ATGC"), 0.5)
    equal("empty sequence has GC 0.0", seq.gc_content(""), 0.0)
    equal("base counts", seq.base_counts("AAGC"), {"A": 2, "C": 1, "G": 1})


def test_cleaning_and_validation():
    equal("whitespace, digits and gaps removed",
          seq.clean("  1 atg c\n-ta.  "), "ATGCTA")
    try:
        seq.validate("ATGX")
        check("invalid base rejected", False, "no error raised")
    except ValueError as error:
        check("invalid base rejected", "position 3" in str(error), str(error))
    equal("IUPAC accepted when asked",
          seq.validate("ATGN", allow_ambiguous=True), "ATGN")
    try:
        seq.validate("ATGN")
        check("N rejected by default", False, "no error raised")
    except ValueError:
        check("N rejected by default", True)


def test_orfs():
    # The HBB fragment with a stop codon appended is a complete ORF.
    construct = HBB_CDS_START + "TAA"
    orfs = seq.find_orfs(construct, min_aa=10, both_strands=False)
    check("ORF found in HBB fragment", len(orfs) == 1, f"found {len(orfs)}")
    if orfs:
        orf = orfs[0]
        equal("ORF protein matches translation", orf["protein"], HBB_PROTEIN_START)
        equal("ORF starts at 0", orf["start"], 0)
        equal("ORF ends after the stop codon", orf["end"], len(construct))
        equal("ORF length in nucleotides", orf["length_nt"], len(construct))

    equal("no ORF below the minimum length",
          seq.find_orfs("ATGAAATAA", min_aa=25, both_strands=False), [])
    equal("ORF running off the end is not reported",
          seq.find_orfs("ATG" + "AAA" * 40, min_aa=5, both_strands=False), [])

    # The same ORF on the other strand is found with matching coordinates.
    flipped = seq.reverse_complement(construct)
    minus = seq.find_orfs(flipped, min_aa=10, both_strands=True)
    on_minus = [o for o in minus if o["strand"] == "-"]
    check("ORF found on the reverse strand", len(on_minus) == 1,
          f"found {len(on_minus)}")
    if on_minus:
        equal("reverse-strand protein matches",
              on_minus[0]["protein"], HBB_PROTEIN_START)
        equal("reverse-strand start coordinate", on_minus[0]["start"], 0)
        equal("reverse-strand end coordinate",
              on_minus[0]["end"], len(construct))


# ------------------------------------------------------------------ file i/o
def test_fasta_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.fasta"
        records = [
            Record(id="HBB_fragment", sequence=HBB_CDS_START,
                   description="first 31 codons"),
            Record(id="short", sequence="ATGC"),
        ]
        written = dnaio.write_fasta(records, path)
        equal("write_fasta reports the count", written, 2)

        back = dnaio.read_fasta(path)
        equal("same number of records read back", len(back), 2)
        equal("identifier survives", back[0].id, "HBB_fragment")
        equal("description survives", back[0].description, "first 31 codons")
        equal("sequence survives", back[0].sequence, HBB_CDS_START)
        equal("second record survives", back[1].sequence, "ATGC")

        text = path.read_text()
        body = [line for line in text.splitlines() if not line.startswith(">")]
        check("lines wrapped at 60 bases",
              all(len(line) <= 60 for line in body),
              f"longest line {max(len(l) for l in body)}")

        wide = Path(tmp) / "wide.fasta"
        dnaio.write_fasta(records, wide, line_width=1000)
        equal("line width is honoured",
              [l for l in wide.read_text().splitlines()
               if not l.startswith(">")][0], HBB_CDS_START)

        # read() dispatches on the extension.
        equal("read() dispatches to FASTA",
              [r.sequence for r in dnaio.read(path)],
              [HBB_CDS_START, "ATGC"])

        try:
            dnaio.write_fasta([], Path(tmp) / "empty.fasta")
            check("empty FASTA refused", False, "no error raised")
        except ValueError:
            check("empty FASTA refused", True)


def test_gzip_reading():
    import gzip

    with tempfile.TemporaryDirectory() as tmp:
        records = [Record(id="HBB_fragment", sequence=HBB_CDS_START,
                          description="gzipped")]
        path = Path(tmp) / "seq.fasta.gz"
        with gzip.open(path, "wt") as handle:
            handle.write(dnaio.format_fasta(records))

        equal("read_fasta reads a gzipped file",
              dnaio.read_fasta(path)[0].sequence, HBB_CDS_START)
        back = dnaio.read(path)
        equal("read() looks past .gz to the real format",
              back[0].sequence, HBB_CDS_START)
        equal("description survives gzip", back[0].description, "gzipped")

        unknown = Path(tmp) / "seq.xyz.gz"
        with gzip.open(unknown, "wt") as handle:
            handle.write("ATGC")
        try:
            dnaio.read(unknown)
            check("unknown format inside .gz rejected", False, "no error raised")
        except ValueError:
            check("unknown format inside .gz rejected", True)


def test_fasta_errors():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "headerless.fasta"
        bad.write_text("ATGC\n")
        try:
            dnaio.read_fasta(bad)
            check("headerless FASTA rejected", False, "no error raised")
        except ValueError:
            check("headerless FASTA rejected", True)

        empty_record = Path(tmp) / "empty_record.fasta"
        empty_record.write_text(">only_a_header\n")
        try:
            dnaio.read_fasta(empty_record)
            check("empty record rejected", False, "no error raised")
        except ValueError:
            check("empty record rejected", True)

        unknown = Path(tmp) / "thing.xyz"
        unknown.write_text("ATGC")
        try:
            dnaio.read(unknown)
            check("unknown extension rejected", False, "no error raised")
        except ValueError:
            check("unknown extension rejected", True)


def test_genbank():
    flat = """LOCUS       TESTSEQ                 24 bp    DNA     linear   TST
DEFINITION  A constructed test record for the
            dnakit GenBank reader.
ACCESSION   TEST00001
ORIGIN
        1 atggtgcatc tgactcctga ggag
//
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.gb"
        path.write_text(flat)
        records = dnaio.read_genbank(path)
        equal("one GenBank record", len(records), 1)
        equal("accession used as id", records[0].id, "TEST00001")
        equal("ORIGIN sequence parsed, numbers stripped",
              records[0].sequence, "ATGGTGCATCTGACTCCTGAGGAG")
        equal("multi-line definition joined",
              records[0].description,
              "A constructed test record for the dnakit GenBank reader")
        equal("GenBank translates to HBB start",
              seq.translate(records[0].sequence), "MVHLTPEE")

        no_origin = Path(tmp) / "no_origin.gb"
        no_origin.write_text("LOCUS  X\n//\n")
        try:
            dnaio.read_genbank(no_origin)
            check("GenBank without ORIGIN rejected", False, "no error raised")
        except ValueError:
            check("GenBank without ORIGIN rejected", True)


def test_uci_csv():
    records = dnaio.read_uci_promoters(UCI_CSV)
    equal("106 UCI records", len(records), 106)
    positives = [r for r in records if r.description == "class=+"]
    negatives = [r for r in records if r.description == "class=-"]
    equal("53 positives", len(positives), 53)
    equal("53 negatives", len(negatives), 53)
    equal("first record identifier", records[0].id, "S10")
    lengths = {len(r) for r in records}
    equal("every sequence is 57 bases", lengths, {57})
    check("all sequences are unambiguous ACGT",
          all(set(r.sequence) <= set("ACGT") for r in records))


def test_record_helpers():
    record = Record(id="r", sequence="atgc", description="d")
    equal("record length", len(record), 4)
    equal("record reverse complement",
          record.reverse_complemented().sequence, "GCAT")
    check("reverse complement noted in description",
          "reverse complement" in record.reverse_complemented().description)
    summary = Record(id="r", sequence="ATGC").summary()
    equal("summary length", summary["length_nt"], 4)
    equal("summary GC", summary["gc_content"], 0.5)


# ------------------------------------------------------------- promoter model
def test_promoter_model():
    model = promoter.load_model()
    equal("weights shaped (57, 4)", model["weights"].shape, (57, 4))
    equal("alphabet is ACGT", model["alphabet"], "ACGT")

    records = dnaio.read_uci_promoters(UCI_CSV)
    first = records[0]
    result = promoter.score(first.sequence, model)

    # The legacy script in experiments/001 must give the same number, which
    # ties this module to the stored artifact rather than a reimplementation.
    legacy = subprocess.run(
        [sys.executable, "-I", "predict.py", first.sequence],
        cwd=ROOT / "experiments" / "001" / "genetic_ml_pilot",
        capture_output=True, text=True, check=True)
    legacy_p = json.loads(legacy.stdout)["probability_promoter_model_score"]
    check("matches experiments/001 predict.py",
          abs(result["probability"] - legacy_p) < 1e-5,
          f"dnakit {result['probability']:.6f} vs predict.py {legacy_p}")

    # Contributions must reconstruct the score exactly.
    contributions = promoter.position_contributions(first.sequence, model)
    equal("one contribution per position", len(contributions), 57)
    total = sum(c["contribution"] for c in contributions) + model["intercept"]
    check("contributions sum to the logit",
          abs(total - result["logit"]) < 1e-9,
          f"sum {total} vs logit {result['logit']}")

    equal("top_positions returns the requested number",
          len(promoter.top_positions(first.sequence, 5, model)), 5)

    # Separation on the training data. This is not an accuracy estimate: the
    # model was fitted on these sequences. It only shows the weights loaded
    # the right way round.
    pos = [promoter.score(r.sequence, model)["logit"]
           for r in records if r.description == "class=+"]
    neg = [promoter.score(r.sequence, model)["logit"]
           for r in records if r.description == "class=-"]
    check("positives score above negatives on training data",
          min(pos) > max(neg),
          f"min positive {min(pos):.2f}, max negative {max(neg):.2f}")

    try:
        promoter.score("ATGC")
        check("wrong window length rejected", False, "no error raised")
    except ValueError as error:
        check("wrong window length rejected", "57" in str(error), str(error))

    try:
        promoter.score("N" * 57)
        check("ambiguous base rejected", False, "no error raised")
    except ValueError:
        check("ambiguous base rejected", True)


def test_promoter_scan():
    records = dnaio.read_uci_promoters(UCI_CSV)
    target = records[0]
    # Embed a known promoter window inside filler and check the scan finds it.
    filler = "ACGT" * 30
    context = filler + target.sequence + filler
    hits = promoter.scan(context, step=1, both_strands=False, top=1)
    equal("best window is the embedded promoter", hits[0]["start"], len(filler))
    equal("best window sequence matches", hits[0]["sequence"], target.sequence)

    both = promoter.scan(context, step=1, both_strands=True)
    equal("both strands scanned", len(both), 2 * (len(context) - 57 + 1))

    try:
        promoter.scan("ACGT")
        check("short sequence rejected by scan", False, "no error raised")
    except ValueError:
        check("short sequence rejected by scan", True)


# ------------------------------------------------------------------------ CLI
def run_cli(*args):
    return subprocess.run([sys.executable, "-I", "-m", "dnakit", *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_cli():
    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "hbb.fasta"
        dnaio.write_fasta([Record(id="HBB_fragment",
                                  sequence=HBB_CDS_START + "TAA")], fasta)

        out = run_cli("read", str(fasta))
        check("cli read succeeds", out.returncode == 0, out.stderr)
        check("cli read shows the length", "96 nt" in out.stdout, out.stdout)

        out = run_cli("translate", str(fasta))
        check("cli translate succeeds", out.returncode == 0, out.stderr)
        check("cli translate gives the HBB protein",
              HBB_PROTEIN_START in out.stdout, out.stdout)

        out = run_cli("orfs", str(fasta), "--min-aa", "10")
        check("cli orfs succeeds", out.returncode == 0, out.stderr)
        check("cli orfs reports one ORF", "1 ORF(s)" in out.stdout, out.stdout)

        converted = Path(tmp) / "converted.fasta"
        out = run_cli("write", str(fasta), "-o", str(converted))
        check("cli write succeeds", out.returncode == 0, out.stderr)
        check("cli write produced a readable file",
              dnaio.read_fasta(converted)[0].sequence == HBB_CDS_START + "TAA")

        out = run_cli("--json", "read", str(fasta))
        check("cli --json emits valid JSON",
              out.returncode == 0 and json.loads(
                  out.stdout[out.stdout.index("["):])[0]["length_nt"] == 96,
              out.stdout)

        out = run_cli("promoter", str(UCI_CSV), "--limit", "1")
        check("cli promoter succeeds", out.returncode == 0, out.stderr)
        check("cli promoter prints the scope note",
              "not expression" in out.stdout, out.stdout)

        out = run_cli("read", str(Path(tmp) / "missing.fasta"))
        check("cli reports a missing file as an error",
              out.returncode == 1 and "error:" in out.stderr, out.stderr)


# ----------------------------------------------------------------- motifs
def test_chance_maths():
    from dnakit import motifs

    equal("exact pattern has one neighbour",
          motifs.neighbourhood_size(6, 0), 1)
    # 6 positions x 3 alternative bases, plus the pattern itself.
    equal("one mismatch gives 19 strings",
          motifs.neighbourhood_size(6, 1), 19)
    equal("three mismatches in a 3-mer covers everything",
          motifs.neighbourhood_size(3, 3), 64)

    # Poisson upper tail: P(X >= 1) with lambda = 1 is 1 - e^-1.
    import math
    check("P(at least 1) at rate 1 is 1 - 1/e",
          abs(motifs.chance_probability(1, 1.0) - (1 - math.exp(-1))) < 1e-12,
          str(motifs.chance_probability(1, 1.0)))
    equal("no matches is never surprising", motifs.chance_probability(0, 5.0), 1.0)
    check("two hits where 0.8 are expected is not significant",
          motifs.chance_probability(2, 0.8) > 0.05,
          f"{motifs.chance_probability(2, 0.8):.3f}")
    check("ten hits where 0.8 are expected is significant",
          motifs.chance_probability(10, 0.8) < 0.05)

    # A 6-mer in 1000 bases, both strands: about 2 x 995 / 4096.
    pribnow = motifs.BY_NAME["-10 box (Pribnow box)"]
    expected = motifs.expected_by_chance(pribnow, 1000, both_strands=True)
    check("chance expectation is in the right range", 8 < expected < 12,
          f"{expected:.2f}")
    equal("a sequence shorter than the motif expects none",
          motifs.expected_by_chance(pribnow, 3), 0.0)


def test_motif_finding():
    from dnakit import motifs

    pribnow = motifs.BY_NAME["-10 box (Pribnow box)"]
    hits = motifs.find("GGGGTATAATGGGG", pribnow, both_strands=False)
    equal("exact Pribnow box found once", len(hits), 1)
    equal("found at the right position", hits[0]["start"], 4)
    equal("recorded as exact", hits[0]["mismatches"], 0)

    # One mismatch is inside this motif's budget, two is not.
    equal("one mismatch accepted",
          len(motifs.find("GGGGTATCATGGGG", pribnow, both_strands=False)), 1)
    equal("two mismatches rejected",
          len(motifs.find("GGGGTACCATGGGG", pribnow, both_strands=False)), 0)

    found = motifs.find_all("GGGGTATAATGGGG", organism=motifs.BACTERIA,
                            both_strands=False, skip_codons=True)
    check("find_all reports the Pribnow box",
          any(h["motif"] == pribnow.name for h in found))
    check("every hit carries a chance probability",
          all("chance_probability" in h for h in found))
    check("eukaryotic motifs excluded when bacteria assumed",
          all(h["organism"] in (motifs.BACTERIA, motifs.ANY) for h in found))


# ----------------------------------------------------------------- explain
def test_explain():
    from dnakit import explain as explainer

    reading = explainer.explain(HBB_CDS_START + "TAA", organism="eukaryote")
    equal("length reported", reading["overview"]["length_nt"], 96)
    equal("six reading frames", reading["overview"]["reading_frames_possible"], 6)
    equal("one complete recipe found",
          len(reading["open_reading_frames"]), 1)
    orf = reading["open_reading_frames"][0]
    equal("recipe is 31 steps", orf["length_aa"], 31)
    equal("recipe is marked certain", orf["certainty"], "certain")
    check("narrative is written", len(reading["narrative"]) >= 4)
    check("narrative states what it cannot tell you",
          any("cannot tell you" in line for line in reading["narrative"]))

    # A sequence with no complete recipe must say so rather than invent one.
    nothing = explainer.explain("ACGT" * 30, organism="eukaryote")
    equal("no recipe found", nothing["open_reading_frames"], [])
    check("absence is stated plainly",
          any("No complete protein recipe" in line
              for line in nothing["narrative"]))


def test_listing():
    from dnakit import explain as explainer

    rows = explainer.listing(HBB_CDS_START + "TAA")
    equal("one row per codon", len(rows), 32)
    equal("first instruction is at position 0", rows[0]["position"], 0)
    equal("first instruction is the start codon", rows[0]["codon"], "ATG")
    check("first instruction explained as START", "START" in rows[0]["means"])
    equal("last instruction is a stop codon", rows[-1]["codon"], "TAA")
    check("last instruction explained as STOP", "STOP" in rows[-1]["means"])
    equal("middle instruction names its building block",
          rows[1]["amino_acid_name"], "Val")
    check("every row is marked certain",
          all(row["certainty"] == "certain" for row in rows))

    text = explainer.listing_as_text(rows, max_rows=3)
    check("listing truncates with a note", "29 more instructions" in text, text)
    check("listing shows all when asked",
          "more instructions" not in
          explainer.listing_as_text(rows, max_rows=3, show_all=True))

    marked = explainer.annotated_sequence(
        "GGGGTATAATGGGG",
        [{"motif": "-10 box", "start": 4, "end": 10, "plain": "switch"}])
    check("annotated sequence marks the motif", "aaaaaa" in marked, marked)
    check("annotated sequence prints a legend", "legend" in marked)


# ------------------------------------------------------------------ review
def test_lint():
    from dnakit import review

    clean = review.lint(HBB_CDS_START + "TAA")
    equal("a well-formed recipe has no findings", clean, [])

    no_start = review.lint("GTG" + HBB_CDS_START[3:] + "TAA")
    check("missing start reported",
          any(f.rule == "missing-start" for f in no_start))

    no_stop = review.lint(HBB_CDS_START)
    check("missing stop reported",
          any(f.rule == "missing-stop" for f in no_stop))

    bad_length = review.lint(HBB_CDS_START + "TA")
    check("length not a multiple of three reported",
          any(f.rule == "frame-length" for f in bad_length))

    early = review.lint(HBB_CDS_START[:18] + "TAA" + HBB_CDS_START[21:] + "TAA")
    premature = [f for f in early if f.rule == "premature-stop"]
    equal("premature stop reported once", len(premature), 1)
    equal("premature stop located", premature[0].position, 18)
    equal("premature stop is an error", premature[0].severity, "error")

    ambiguous = review.lint("ATGNNNTAA")
    check("ambiguous letters warned about",
          any(f.rule == "ambiguous-base" for f in ambiguous))
    equal("ambiguity is a warning, not an error",
          [f.severity for f in ambiguous if f.rule == "ambiguous-base"],
          ["warning"])

    # Non-coding mode drops the coding rules but keeps the universal ones.
    equal("non-coding mode ignores start and stop",
          review.lint("ACGTACGT", expect_coding=False), [])


def test_compare():
    from dnakit import review

    reference = HBB_CDS_START + "TAA"

    def swap(codon_number, new):
        index = (codon_number - 1) * 3
        return reference[:index] + new + reference[index + 3:]

    equal("identical sequences",
          review.compare(reference, reference)["consequence"], "identical")

    # Codon 7 is GAG, coding glutamate.
    equal("reference codon 7", reference[18:21], "GAG")

    silent = review.compare(reference, swap(7, "GAA"))
    equal("same amino acid is synonymous", silent["consequence"], "synonymous")
    equal("protein unchanged by a synonymous change",
          silent["reference_protein"], silent["variant_protein"])

    missense = review.compare(reference, swap(7, "GTG"))
    equal("different amino acid is missense", missense["consequence"], "missense")
    equal("missense notation", missense["changes"][0]["notation"], "E7V")
    equal("protein stays the same length",
          len(missense["variant_protein"]), len(missense["reference_protein"]))

    nonsense = review.compare(reference, swap(7, "TAA"))
    equal("a new stop codon is nonsense", nonsense["consequence"], "nonsense")
    check("nonsense says how much is lost",
          "25 steps early" in nonsense["plain"], nonsense["plain"])

    start_lost = review.compare(reference, swap(1, "AAG"))
    equal("losing the start codon is reported",
          start_lost["consequence"], "start lost")

    stop_lost = review.compare(reference, reference[:-3] + "CAA")
    equal("losing the stop codon is reported",
          stop_lost["consequence"], "stop lost")

    frameshift = review.compare(reference, reference[:18] + reference[19:])
    equal("a one-letter deletion is a frameshift",
          frameshift["consequence"], "frameshift")
    equal("frameshift located", frameshift["changes"][0]["position"], 18)
    check("frameshift protein diverges after the change",
          frameshift["variant_protein"][:6] == "MVHLTP"
          and frameshift["variant_protein"][6:] !=
          frameshift["reference_protein"][6:],
          frameshift["variant_protein"])

    in_frame = review.compare(reference, reference[:18] + reference[21:])
    equal("a three-letter deletion keeps the frame",
          in_frame["consequence"], "in-frame deletion")
    equal("in-frame deletion shortens the protein by one",
          len(in_frame["variant_protein"]),
          len(in_frame["reference_protein"]) - 1)

    insertion = review.compare(reference, reference[:18] + "GGG" + reference[18:])
    equal("a three-letter insertion keeps the frame",
          insertion["consequence"], "in-frame insertion")

    check("numbering convention is stated",
          "initiator methionine" in review.compare(reference, swap(7, "GTG"))["numbering"])


def test_review_combined():
    from dnakit import review

    reference = HBB_CDS_START + "TAA"
    report = review.review(reference)
    check("a clean sequence reviews clean", report["clean"])
    equal("no comparison without a reference", report["comparison"], None)

    broken = reference[:18] + "TAA" + reference[21:]
    report = review.review(broken, reference=reference)
    check("problems are reported", not report["clean"])
    equal("comparison included when a reference is given",
          report["comparison"]["consequence"], "nonsense")


def test_cli_explain_lint_diff():
    with tempfile.TemporaryDirectory() as tmp:
        reference = Path(tmp) / "ref.fasta"
        dnaio.write_fasta([Record(id="ref", sequence=HBB_CDS_START + "TAA")],
                          reference)
        variant = Path(tmp) / "var.fasta"
        dnaio.write_fasta(
            [Record(id="var",
                    sequence=HBB_CDS_START[:18] + "GTG" + HBB_CDS_START[21:]
                    + "TAA")], variant)

        out = run_cli("explain", str(reference), "--organism", "eukaryote")
        check("cli explain succeeds", out.returncode == 0, out.stderr)
        check("cli explain describes the recipe",
              "complete recipe" in out.stdout, out.stdout)
        check("cli explain lists instructions",
              "START" in out.stdout, out.stdout)

        out = run_cli("lint", str(reference))
        check("cli lint succeeds on a clean sequence",
              out.returncode == 0 and "clean" in out.stdout, out.stdout)

        broken = Path(tmp) / "broken.fasta"
        dnaio.write_fasta([Record(id="broken", sequence="ATGAAA")], broken)
        out = run_cli("lint", str(broken))
        check("cli lint reports a missing stop",
              "missing-stop" in out.stdout or "stop codon" in out.stdout,
              out.stdout)

        out = run_cli("diff", str(variant), "-r", str(reference))
        check("cli diff succeeds", out.returncode == 0, out.stderr)
        check("cli diff names the consequence",
              "missense" in out.stdout, out.stdout)
        check("cli diff gives the notation", "E7V" in out.stdout, out.stdout)


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        print(f"\n{test.__name__}")
        test()

    failed = [name for name, ok, _ in _results if not ok]
    print(f"\n{'-' * 60}")
    print(f"{len(_results) - len(failed)} passed, {len(failed)} failed")
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
