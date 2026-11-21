#!/usr/bin/env python3
"""Download the two ENCODE K562 MPRA experiments named in HANDOFF.md.

GEO GSE266775 and GSE266796 are mirrors of ENCODE experiments ENCSR820RPI and
ENCSR997GVD (Tewhey lab, JAX). ENCODE is the original source and publishes an
md5 for every file, so the files are fetched from there and checked.

Only the processed tables and the element reference FASTA files are fetched:
about 30 MB. Raw reads (6-8 GB per experiment) and the two large
quantification tables (381 MB and 859 MB) are listed in the manifest but not
downloaded.

Files land in data/mpra/<experiment>/, which is git-ignored. A manifest with
checksums is written next to this script and is committed, so anyone can
confirm they hold the same bytes.

Data use: ENCODE data are released without restriction under the ENCODE Data
Use Policy; cite the ENCODE Consortium and the producing lab.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
DATA = ROOT / "data" / "mpra"
BASE = "https://www.encodeproject.org"

# md5 values are the ones ENCODE publishes for each file.
FILES = [
    {"experiment": "ENCSR820RPI", "geo": "GSE266775", "accession": "ENCFF157VVX",
     "filename": "ENCFF157VVX.tsv", "role": "element quantifications (summary)",
     "md5": "25da1d675a4f22dc98fcb83f83922abf"},
    {"experiment": "ENCSR820RPI", "geo": "GSE266775", "accession": "ENCFF798CYL",
     "filename": "ENCFF798CYL.fasta.gz",
     "role": "elements reference (tested sequences), from ENCSR252UKW",
     "md5": "3b9068f7dc85b42ad047f73300327829"},
    {"experiment": "ENCSR997GVD", "geo": "GSE266796", "accession": "ENCFF612YRJ",
     "filename": "ENCFF612YRJ.tsv", "role": "element quantifications (summary)",
     "md5": "aba6cbd03f7dbb20e6ddbaaa70d3a551"},
    {"experiment": "ENCSR997GVD", "geo": "GSE266796", "accession": "ENCFF161KFE",
     "filename": "ENCFF161KFE.tsv", "role": "variant effect quantifications, hg19",
     "md5": "2b6ffcd82b63b5116964b0be9a9c6715"},
    {"experiment": "ENCSR997GVD", "geo": "GSE266796", "accession": "ENCFF621JOY",
     "filename": "ENCFF621JOY.fasta.gz",
     "role": "elements reference (tested sequences), from ENCSR804MZR",
     "md5": "c3ac432c2af2baa8342633eb394b14b5"},
]

NOT_DOWNLOADED = [
    {"experiment": "ENCSR820RPI", "accession": "ENCFF294GFK",
     "role": "element quantifications (large table)", "size_mb": 381.3},
    {"experiment": "ENCSR997GVD", "accession": "ENCFF227KRF",
     "role": "element quantifications (large table)", "size_mb": 859.2},
    {"experiment": "ENCSR820RPI", "role": "raw reads, 5 fastq files",
     "size_mb": 7497.3},
    {"experiment": "ENCSR997GVD", "role": "raw reads, 10 fastq files",
     "size_mb": 6184.2},
]


def md5_of(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(entry):
    target = DATA / entry["experiment"] / entry["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/files/{entry['accession']}/@@download/{entry['filename']}"

    if target.exists() and md5_of(target) == entry["md5"]:
        status = "already present, checksum ok"
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "research-script"})
        with urllib.request.urlopen(request, timeout=300) as response:
            target.write_bytes(response.read())
        status = "downloaded"

    actual = md5_of(target)
    if actual != entry["md5"]:
        raise SystemExit(
            f"{entry['filename']}: md5 {actual} does not match the published "
            f"{entry['md5']}. Refusing to continue with unverified data.")
    size = target.stat().st_size
    print(f"  {entry['filename']:24} {size/1e6:7.2f} MB  md5 ok  ({status})")
    return {**entry, "url": url, "bytes": size, "md5_verified": True,
            "path": str(target.relative_to(ROOT))}


def main():
    print("Fetching ENCODE MPRA processed files:")
    fetched = [fetch(entry) for entry in FILES]
    manifest = {
        "source": "ENCODE Project (encodeproject.org), Tewhey lab, JAX",
        "experiments": {
            "ENCSR820RPI": {"geo_mirror": "GSE266775", "assay": "MPRA",
                            "biosample": "K562, episomal, transient transfection",
                            "elements_reference": "ENCSR252UKW"},
            "ENCSR997GVD": {"geo_mirror": "GSE266796", "assay": "MPRA",
                            "biosample": "K562, episomal, transient transfection",
                            "elements_reference": "ENCSR804MZR"},
        },
        "data_use": ("ENCODE Data Use Policy: unrestricted use; cite the "
                     "ENCODE Consortium and the producing lab"),
        "files": fetched,
        "listed_but_not_downloaded": NOT_DOWNLOADED,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote manifest.json")


if __name__ == "__main__":
    main()
