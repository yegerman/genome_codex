#!/usr/bin/env python3
"""Stress-test the leakage control used in experiments 001 and 002.

Both experiments group sequences by shared exact 25-mers (connected
components) and report ~94.3% out-of-fold accuracy for a positional one-hot
logistic model. The 25 bp threshold is a choice, not a derived quantity, and
the handoff flags that it "does not guarantee biological independence".

This script asks a falsifiable question: does the headline accuracy depend on
that threshold? It rebuilds the grouping at a range of thresholds, including
thresholds far stricter than 25, and re-runs the same model and the same
nested-tuning protocol at each one. If the result is driven by residual
sequence overlap between folds, accuracy should fall as grouping tightens.

It also measures how much exact sequence identity still crosses fold
boundaries at the original threshold, on both strands.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

HERE = Path(__file__).parent
SRC = HERE.parent.parent / "001" / "genetic_ml_pilot" / "promoters_source.csv"
LENGTH = 57
BASES = "ACGT"
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def load():
    rows = [r for r in csv.reader(SRC.open()) if r]
    seqs = np.array([r[2].upper() for r in rows])
    labels = np.array([int(r[0] == "+") for r in rows])
    ids = np.array([r[1] for r in rows])
    assert len(seqs) == 106 and Counter(labels)[1] == 53
    assert all(len(s) == LENGTH and set(s) <= set(BASES) for s in seqs)
    return seqs, labels, ids


def revcomp(s):
    return "".join(COMPLEMENT[b] for b in reversed(s))


class OneHotPosition(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx = {b: i for i, b in enumerate(BASES)}
        out = np.zeros((len(X), LENGTH, 4), np.float32)
        for row, s in enumerate(X):
            for j, b in enumerate(s):
                out[row, j, idx[b]] = 1
        return out.reshape(len(X), -1)


def longest_common_substring(a, b):
    """Length of the longest exact shared contiguous run between two strings."""
    av = np.frombuffer(a.encode(), dtype=np.uint8)
    bv = np.frombuffer(b.encode(), dtype=np.uint8)
    match = av[:, None] == bv[None, :]
    best = 0
    prev = np.zeros(len(bv) + 1, dtype=np.int32)
    for i in range(len(av)):
        cur = np.zeros(len(bv) + 1, dtype=np.int32)
        cur[1:] = (prev[:-1] + 1) * match[i]
        best = max(best, int(cur.max()))
        prev = cur
    return best


def pairwise_overlap(seqs, use_revcomp):
    """Max shared exact run for every pair, optionally considering both strands."""
    n = len(seqs)
    overlap = np.zeros((n, n), dtype=np.int32)
    rc = [revcomp(s) for s in seqs]
    for i in range(n):
        for j in range(i + 1, n):
            best = longest_common_substring(seqs[i], seqs[j])
            if use_revcomp:
                best = max(best, longest_common_substring(seqs[i], rc[j]))
            overlap[i, j] = overlap[j, i] = best
    return overlap


def components_from_overlap(overlap, threshold):
    """Connected components linking any pair sharing >= threshold bases."""
    n = len(overlap)
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if overlap[i, j] >= threshold:
                a, b = root(i), root(j)
                if a != b:
                    parent[a] = b
    return np.array([root(i) for i in range(n)])


def evaluate(seqs, labels, groups, seed=2026, n_splits=4):
    """Same model and nested-tuning protocol as experiment 001."""
    n_groups = len(set(groups))
    if n_groups < n_splits:
        return None
    pipe = Pipeline([("onehot", OneHotPosition()),
                     ("clf", LogisticRegression(solver="liblinear", max_iter=3000))])
    grid = {"clf__C": [0.01, 0.1, 1, 10]}
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(labels), dtype=int)
    score = np.zeros(len(labels), dtype=float)
    crossing = 0
    for fold, (tr, te) in enumerate(outer.split(seqs, labels, groups)):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + fold)
        search = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=inner, n_jobs=1)
        fitted = search.fit(seqs[tr], labels[tr], groups=groups[tr]).best_estimator_
        pred[te] = fitted.predict(seqs[te])
        score[te] = fitted.decision_function(seqs[te])
    return {
        "n_groups": int(n_groups),
        "largest_group": int(max(Counter(groups).values())),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "roc_auc": float(roc_auc_score(labels, score)),
        "n_correct": int((pred == labels).sum()),
    }


def cross_fold_identity(seqs, labels, groups, overlap, seed=2026, n_splits=4):
    """How much exact sequence identity still crosses the fold boundary."""
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_test_max = np.zeros(len(seqs), dtype=int)
    for tr, te in outer.split(seqs, labels, groups):
        for i in te:
            per_test_max[i] = overlap[i, tr].max()
    return per_test_max


def main():
    seqs, labels, ids = load()
    n = len(seqs)

    same_strand = pairwise_overlap(seqs, use_revcomp=False)
    both_strand = pairwise_overlap(seqs, use_revcomp=True)

    # Expected number of pairs sharing a k-mer purely by chance, for context.
    n_pairs = n * (n - 1) // 2
    positions = (LENGTH - np.arange(5, 40) + 1)
    chance = {
        int(k): float(n_pairs * (positions[k - 5] ** 2) * (0.25 ** k))
        for k in range(8, 31, 2)
    }

    thresholds = [10, 12, 15, 18, 20, 22, 25, 30, 35]
    sweep = []
    for k in thresholds:
        for strand_label, matrix in (("same_strand", same_strand),
                                     ("both_strands", both_strand)):
            groups = components_from_overlap(matrix, k)
            result = evaluate(seqs, labels, groups)
            if result is None:
                sweep.append({"threshold_bp": k, "strand": strand_label,
                              "skipped": "fewer groups than folds"})
                continue
            result.update({"threshold_bp": k, "strand": strand_label})
            sweep.append(result)
            print(f"k={k:>3} {strand_label:<13} groups={result['n_groups']:>3} "
                  f"largest={result['largest_group']:>3} "
                  f"correct={result['n_correct']:>3}/106 "
                  f"acc={result['accuracy']*100:.2f}% auc={result['roc_auc']:.3f}",
                  flush=True)

    # Residual identity across the fold boundary at the original 25 bp rule.
    groups25 = components_from_overlap(same_strand, 25)
    residual = cross_fold_identity(seqs, labels, groups25, same_strand)
    residual_rc = cross_fold_identity(seqs, labels, groups25, both_strand)

    # The negatives are described upstream as windows from limited genomic
    # context; quantify how much they overlap each other versus the positives.
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    def within(idx, matrix):
        vals = [int(matrix[i, j]) for a, i in enumerate(idx) for j in idx[a + 1:]]
        return {
            "n_pairs": len(vals),
            "median_bp": float(np.median(vals)),
            "p95_bp": float(np.percentile(vals, 95)),
            "max_bp": int(max(vals)),
            "n_pairs_ge_25bp": int(sum(v >= 25 for v in vals)),
        }

    report = {
        "question": (
            "Does the ~94.3% out-of-fold accuracy depend on the 25 bp grouping "
            "threshold used in experiments 001 and 002?"
        ),
        "model": "positional one-hot + logistic regression, nested grouped tuning (as in 001)",
        "n": n,
        "expected_chance_pairs_sharing_kmer": chance,
        "threshold_sweep": sweep,
        "residual_identity_at_25bp_rule": {
            "same_strand_max_shared_bp_per_heldout_sequence":
                residual.tolist(),
            "same_strand_n_heldout_sharing_ge_20bp_with_train":
                int((residual >= 20).sum()),
            "same_strand_n_heldout_sharing_ge_15bp_with_train":
                int((residual >= 15).sum()),
            "both_strands_n_heldout_sharing_ge_20bp_with_train":
                int((residual_rc >= 20).sum()),
            "note": (
                "The grouping guarantees no shared 25-mer across folds, so the "
                "maximum here is 24 by construction. The question is how close "
                "to 24 the distribution sits."
            ),
        },
        "within_class_overlap_same_strand": {
            "positives": within(pos_idx, same_strand),
            "negatives": within(neg_idx, same_strand),
        },
    }
    (HERE / "leakage_audit.json").write_text(json.dumps(report, indent=2))
    print("\nwrote leakage_audit.json")


if __name__ == "__main__":
    main()
