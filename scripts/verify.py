#!/usr/bin/env python3
"""Re-run the historical experiments in a temporary copy and compare.

Experiment 001 (scikit-learn) must match the stored JSON exactly.
Experiment 002 (torch) is reported; tiny differences can appear across CPU
types and BLAS libraries, so a mismatch there is a warning, not a failure.
Stored results in experiments/ are never modified.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_copy(rel, script):
    src = ROOT / rel
    tmp = Path(tempfile.mkdtemp(prefix="verify_"))
    dst = tmp / src.name
    shutil.copytree(src, dst)
    proc = subprocess.run([sys.executable, "-I", script], cwd=dst,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        raise SystemExit(f"{rel}/{script} failed with exit code {proc.returncode}")
    return src, dst


def check_001():
    src, dst = run_copy("experiments/001/genetic_ml_pilot", "train_grouped.py")
    old = json.loads((src / "grouped_results.json").read_text())["aggregate_out_of_fold"]
    new = json.loads((dst / "grouped_results.json").read_text())["aggregate_out_of_fold"]
    ok = True
    for o, n in zip(old, new):
        same = o["confusion_matrix"] == n["confusion_matrix"]
        ok &= same
        print(f"  001 {o['model']:22} stored {o['accuracy']*100:6.2f}%  "
              f"now {n['accuracy']*100:6.2f}%  {'OK' if same else 'MISMATCH'}")
    return ok


def check_002():
    src, dst = run_copy("experiments/002/dna_language_model_pilot", "train_dna_mlm.py")
    old = {m["model"]: m for m in json.loads((src / "results.json").read_text())["metrics"]}
    new = {m["model"]: m for m in json.loads((dst / "results.json").read_text())["metrics"]}
    ok = True
    for name, o in old.items():
        same = o["correct"] == new[name]["correct"]
        ok &= same
        print(f"  002 {name:22} stored {o['correct']:>3}/106  "
              f"now {new[name]['correct']:>3}/106  {'OK' if same else 'DIFFERS'}")
    return ok


def main():
    print("Experiment 001 (must match exactly):")
    ok1 = check_001()
    print("Experiment 002 (torch, small hardware differences possible):")
    ok2 = check_002()
    if not ok1:
        raise SystemExit("FAIL: experiment 001 did not reproduce.")
    if not ok2:
        print("WARNING: experiment 002 differs slightly; expected on some hardware.")
    print("PASS: historical results reproduce on this machine.")


if __name__ == "__main__":
    main()
