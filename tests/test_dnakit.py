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
