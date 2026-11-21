#!/usr/bin/env python3
"""How stable is the headline 94.34% to the arbitrary parts of the protocol?

Experiments 001 and 002 report 100/106 (94.34%) out-of-fold for the positional
one-hot logistic model under a fixed 25-mer grouping and a single outer split
(StratifiedGroupKFold, random_state=2026).

Two things in that protocol are arbitrary and carry no biological meaning:

1. The integer *labels* assigned to the connected components. Union-find emits
   whichever member happens to become the root, and StratifiedGroupKFold packs
   groups into folds in an order that depends on those labels. The partition of
   sequences into groups is identical; only the names change.
2. The outer-split seed.

If the reported accuracy is a stable property of the data and model, it should
barely move under either. This script measures that spread, and checks whether
the *ranking* of the models is more stable than their absolute scores.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
SRC = HERE.parent.parent / "001" / "genetic_ml_pilot" / "promoters_source.csv"
LENGTH, BASES, KMER = 57, "ACGT", 25
N_REPEATS = 50


def load():
    rows = [r for r in csv.reader(SRC.open()) if r]
    return (np.array([r[2].upper() for r in rows]),
            np.array([int(r[0] == "+") for r in rows]),
            np.array([r[1] for r in rows]))


def grouping_as_in_experiment_001(seqs):
    """Reproduce 001's union-find over shared exact 25-mers, verbatim in effect."""
    n = len(seqs)
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kmers = defaultdict(set)
    for i, s in enumerate(seqs):
        for j in range(LENGTH - KMER + 1):
            kmers[s[j:j + KMER]].add(i)
    for members in kmers.values():
        members = list(members)
        for other in members[1:]:
            a, b = root(members[0]), root(other)
            if a != b:
                parent[a] = b
    return np.array([root(i) for i in range(n)])


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


class GC(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([[sum(b in "GC" for b in s) / LENGTH] for s in X])


MODELS = {
    "majority": (DummyClassifier(strategy="prior"), {}),
    "GC_only": (Pipeline([("gc", GC()), ("z", StandardScaler()),
                          ("m", LogisticRegression(max_iter=3000))]),
                {"m__C": [0.01, 0.1, 1, 10]}),
    "DNA_kmer_3_4": (Pipeline([("k", CountVectorizer(analyzer="char",
                                                     ngram_range=(3, 4),
                                                     lowercase=False)),
                               ("m", LogisticRegression(solver="liblinear",
                                                        max_iter=3000))]),
                     {"m__C": [0.01, 0.1, 1, 10]}),
    "DNA_position_onehot": (Pipeline([("o", OneHotPosition()),
                                      ("m", LogisticRegression(solver="liblinear",
                                                               max_iter=3000))]),
                            {"m__C": [0.01, 0.1, 1, 10]}),
}


def run_once(seqs, labels, groups, outer_seed):
    outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=outer_seed)
    scores = {}
    for name, (pipe, grid) in MODELS.items():
        pred = np.zeros(len(labels), dtype=int)
        for fold, (tr, te) in enumerate(outer.split(seqs, labels, groups)):
            assert set(groups[tr]).isdisjoint(set(groups[te]))
            if grid:
                inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                             random_state=42 + fold)
                search = GridSearchCV(pipe, grid, scoring="balanced_accuracy",
                                      cv=inner, n_jobs=1)
                fitted = search.fit(seqs[tr], labels[tr],
                                    groups=groups[tr]).best_estimator_
            else:
                fitted = pipe.fit(seqs[tr], labels[tr])
            pred[te] = fitted.predict(seqs[te])
        scores[name] = {
            "accuracy": float(accuracy_score(labels, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "n_correct": int((pred == labels).sum()),
        }
    return scores


def summarise(runs, key="accuracy"):
    out = {}
    for name in MODELS:
        vals = np.array([r[name][key] for r in runs])
        out[name] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(np.median(vals)),
            "n_correct_min": int(round(vals.min() * 106)),
            "n_correct_max": int(round(vals.max() * 106)),
        }
    return out


def main():
    seqs, labels, ids = load()
    groups = grouping_as_in_experiment_001(seqs)
    partition = [sorted(np.where(groups == g)[0].tolist())
                 for g in sorted(set(groups))]
    assert len(partition) == 69, f"expected 69 groups, got {len(partition)}"

    baseline = run_once(seqs, labels, groups, outer_seed=2026)
    print("as published (group labels from union-find, seed 2026):")
    for name, s in baseline.items():
        print(f"  {name:22} {s['n_correct']:>3}/106  {s['accuracy']*100:.2f}%")

    # (1) Identical partition, shuffled group NAMES. The biology is untouched.
    rng = np.random.default_rng(7)
    relabel_runs = []
    for _ in range(N_REPEATS):
        order = rng.permutation(len(partition))
        relabelled = np.zeros(len(seqs), dtype=int)
        for new_name, members in zip(order, partition):
            relabelled[members] = new_name
        relabel_runs.append(run_once(seqs, labels, relabelled, outer_seed=2026))
    print(f"\nrelabelling groups only, {N_REPEATS} draws, same partition and seed:")
    relabel_summary = summarise(relabel_runs)
    for name, s in relabel_summary.items():
        print(f"  {name:22} mean={s['mean']*100:.2f}% sd={s['sd']*100:.2f} "
              f"range={s['n_correct_min']}-{s['n_correct_max']}/106")

    # (2) Original group labels, varying the outer-split seed.
    seed_runs = [run_once(seqs, labels, groups, outer_seed=s)
                 for s in range(N_REPEATS)]
    print(f"\nvarying outer split seed, {N_REPEATS} seeds, original group labels:")
    seed_summary = summarise(seed_runs)
    for name, s in seed_summary.items():
        print(f"  {name:22} mean={s['mean']*100:.2f}% sd={s['sd']*100:.2f} "
              f"range={s['n_correct_min']}-{s['n_correct_max']}/106")

    # Does the model ranking survive, even though the numbers move?
    all_runs = relabel_runs + seed_runs
    best_counts = Counter(max(MODELS, key=lambda m: r[m]["accuracy"])
                          for r in all_runs)
    beats_kmer = sum(r["DNA_position_onehot"]["accuracy"] >
                     r["DNA_kmer_3_4"]["accuracy"] for r in all_runs)
    beats_gc = sum(r["DNA_position_onehot"]["accuracy"] >
                   r["GC_only"]["accuracy"] for r in all_runs)

    published = baseline["DNA_position_onehot"]["accuracy"]
    pooled = np.array([r["DNA_position_onehot"]["accuracy"] for r in all_runs])
    report = {
        "question": ("Is the published 94.34% a stable property, or one draw "
                     "from a spread created by arbitrary protocol choices?"),
        "partition_n_groups": len(partition),
        "published_point_estimate": {
            "accuracy": published,
            "n_correct": baseline["DNA_position_onehot"]["n_correct"],
            "note": "group labels as emitted by union-find, outer seed 2026",
        },
        "relabel_only": relabel_summary,
        "vary_outer_seed": seed_summary,
        "pooled_positional_model": {
            "n_runs": len(pooled),
            "mean_accuracy": float(pooled.mean()),
            "sd_accuracy": float(pooled.std(ddof=1)),
            "percentile_of_published": float((pooled <= published).mean() * 100),
            "range_n_correct": [int(round(pooled.min() * 106)),
                                int(round(pooled.max() * 106))],
        },
        "ranking_stability": {
            "times_each_model_was_best": dict(best_counts),
            "positional_beats_kmer": f"{beats_kmer}/{len(all_runs)}",
            "positional_beats_gc": f"{beats_gc}/{len(all_runs)}",
        },
        "interpretation": (
            "The partition of sequences into similarity groups is identical in "
            "every relabelling run. Any movement there is protocol noise, not "
            "biology."
        ),
    }
    (HERE / "fold_stability.json").write_text(json.dumps(report, indent=2))
    print("\npublished estimate sits at the "
          f"{report['pooled_positional_model']['percentile_of_published']:.0f}th "
          "percentile of the pooled runs")
    print("wrote fold_stability.json")


if __name__ == "__main__":
    main()
