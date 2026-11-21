"""Command line for reading and writing DNA.

    python -m dnakit read      FILE              summarise every record
    python -m dnakit translate FILE [--frame N]  protein from a reading frame
    python -m dnakit orfs      FILE [--min-aa N] open reading frames
    python -m dnakit revcomp   FILE              the other strand, as FASTA
    python -m dnakit write     FILE -o OUT.fa    convert any input to FASTA
    python -m dnakit promoter  FILE [--scan]     score with the 001 model

Add --json to any command for machine-readable output.
"""
import argparse
import json
import sys

from . import io as dnaio
from . import sequence as seq
from .io import Record


def _records(path, limit=None):
    records = dnaio.read(path)
    return records[:limit] if limit else records


def cmd_read(args):
    records = _records(args.file, args.limit)
    summaries = [record.summary() for record in records]
    if args.json:
        return summaries
    for summary in summaries:
        header = summary["id"]
        if summary["description"]:
            header += f"  {summary['description']}"
        print(header)
        print(f"  length    {summary['length_nt']} nt")
        print(f"  GC        {summary['gc_content']:.1%}")
        counts = "  ".join(f"{base}={count}"
                           for base, count in summary["base_counts"].items())
        print(f"  bases     {counts}")
        if summary["ambiguous_bases"]:
            print(f"  ambiguous {summary['ambiguous_bases']}")
    print(f"\n{len(summaries)} record(s) from {args.file}")
    return summaries


def cmd_translate(args):
    out = []
    for record in _records(args.file, args.limit):
        protein = seq.translate(record.sequence, frame=args.frame,
                                stop_at_stop=args.stop_at_stop)
        out.append({"id": record.id, "frame": args.frame, "protein": protein,
                    "length_aa": len(protein)})
        if not args.json:
            print(f">{record.id} frame={args.frame} aa={len(protein)}")
            for i in range(0, len(protein), 60):
                print(protein[i:i + 60])
    return out


def cmd_orfs(args):
    out = []
    for record in _records(args.file, args.limit):
        orfs = seq.find_orfs(record.sequence, min_aa=args.min_aa,
                             both_strands=not args.forward_only)
        out.append({"id": record.id, "n_orfs": len(orfs), "orfs": orfs})
        if not args.json:
            print(f"{record.id}: {len(orfs)} ORF(s) of at least {args.min_aa} aa")
            for orf in orfs[:args.limit_orfs]:
                print(f"  {orf['strand']} frame {orf['frame']} "
                      f"{orf['start']}..{orf['end']} "
                      f"{orf['length_aa']} aa  {orf['protein'][:40]}"
                      f"{'...' if orf['length_aa'] > 40 else ''}")
    return out


def cmd_revcomp(args):
    records = [record.reverse_complemented() for record in _records(args.file)]
    text = dnaio.format_fasta(records)
    if args.out:
        dnaio.write_fasta(records, args.out)
        print(f"wrote {len(records)} record(s) to {args.out}")
    else:
        sys.stdout.write(text)
    return [record.summary() for record in records]


def cmd_write(args):
    records = _records(args.file, args.limit)
    count = dnaio.write_fasta(records, args.out, line_width=args.line_width)
    print(f"wrote {count} record(s) to {args.out}")
    return {"written": count, "path": args.out}


def cmd_promoter(args):
    from . import promoter

    model = promoter.load_model()
    out = []
    for record in _records(args.file, args.limit):
        if args.scan:
            hits = promoter.scan(record.sequence, step=args.step,
                                 model=model, top=args.top)
            out.append({"id": record.id, "windows": hits})
            if not args.json:
                print(f"{record.id}: top {len(hits)} window(s)")
                for hit in hits:
                    print(f"  {hit['strand']} {hit['start']}..{hit['end']} "
                          f"logit {hit['logit']:+.2f}  p {hit['probability']:.3f}")
            continue

        result = promoter.score(record.sequence, model)
        result["id"] = record.id
        result["top_positions"] = promoter.top_positions(record.sequence,
                                                         args.top, model)
        out.append(result)
        if not args.json:
            print(f"{record.id}: {result['predicted_class']}  "
                  f"logit {result['logit']:+.2f}  p {result['probability']:.3f}")
            print(f"  top positions (1-based in the {promoter.WINDOW}-base window):")
            for item in result["top_positions"]:
                print(f"    {item['position']:>3} {item['base']} "
                      f"{item['contribution']:+.3f}")
    if not args.json:
        print(f"\nscope: {promoter.SCOPE}")
    return out


def build_parser():
    parser = argparse.ArgumentParser(
        prog="dnakit", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true",
                        help="print the result as JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N records")

    # The same two options are accepted after the subcommand as well, which is
    # where people reach for them. SUPPRESS keeps an unused copy from
    # overwriting a value given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true",
                        default=argparse.SUPPRESS,
                        help="print the result as JSON")
    common.add_argument("--limit", type=int, default=argparse.SUPPRESS,
                        help="only process the first N records")

    subparsers = parser.add_subparsers(dest="command", required=True,
                                       parser_class=lambda **kw:
                                       argparse.ArgumentParser(parents=[common], **kw))

    read = subparsers.add_parser("read", help="summarise each record")
    read.set_defaults(func=cmd_read)

    translate = subparsers.add_parser("translate", help="translate to protein")
    translate.add_argument("--frame", type=int, default=0, choices=(0, 1, 2))
    translate.add_argument("--stop-at-stop", action="store_true",
                           help="end at the first stop codon")
    translate.set_defaults(func=cmd_translate)

    orfs = subparsers.add_parser("orfs", help="find open reading frames")
    orfs.add_argument("--min-aa", type=int, default=25)
    orfs.add_argument("--forward-only", action="store_true")
    orfs.add_argument("--limit-orfs", type=int, default=10)
    orfs.set_defaults(func=cmd_orfs)

    revcomp = subparsers.add_parser("revcomp", help="reverse complement")
    revcomp.add_argument("-o", "--out", default=None)
    revcomp.set_defaults(func=cmd_revcomp)

    write = subparsers.add_parser("write", help="write records as FASTA")
    write.add_argument("-o", "--out", required=True)
    write.add_argument("--line-width", type=int, default=60)
    write.set_defaults(func=cmd_write)

    prom = subparsers.add_parser(
        "promoter", help="score with the experiment 001 positional model")
    prom.add_argument("--scan", action="store_true",
                      help="slide the window along a longer sequence")
    prom.add_argument("--step", type=int, default=1)
    prom.add_argument("--top", type=int, default=10)
    prom.set_defaults(func=cmd_promoter)

    for sub in (read, translate, orfs, revcomp, write, prom):
        sub.add_argument("file", help="FASTA, GenBank, UCI CSV or plain text")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = args.func(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
