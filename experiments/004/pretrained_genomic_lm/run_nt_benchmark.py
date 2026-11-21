#!/usr/bin/env python3
"""Does a genuinely pretrained genomic language model beat simple baselines?

Experiment 002 trained a tiny Transformer from scratch on 106 sequences and
found no benefit from masked pretraining. That is a statement about a toy
model, not about genomic foundation models. This experiment runs the test the
handoff actually asks for: frozen embeddings from a real pretrained genomic
language model, against tuned classical baselines, on the same data and the
same leakage control.

Model: InstaDeepAI/nucleotide-transformer-v2-50m-multi-species.
Pretrained on 850 genomes spanning multiple species including prokaryotes,
which matters because the evaluation data is E. coli. License CC-BY-NC-SA-4.0,
so this is a research-only comparison. The custom modeling code in that repo
was read before use: it is a vendored ESM definition importing only math,
typing, torch and transformers, with no side effects.

Two corrections to the earlier protocol, both justified by the audit in
experiments/003:

1. Every model is tuned the same way. Experiment 002 compared a fixed-C probe
   against a nested-tuned baseline, which the handoff flags as unfair. Here the
   probe's regularisation AND its representation (which layer, mean-pooled or
   per-token) are selected inside the training folds only.
2. Results are averaged over many fold assignments rather than reported from
   one. The audit showed that relabelling the similarity groups, which leaves
   the partition of sequences untouched, moves accuracy by several points. A
   single split is not a measurement.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
SRC = HERE.parent.parent / "001" / "genetic_ml_pilot" / "promoters_source.csv"
MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
LENGTH, BASES, KMER = 57, "ACGT", 25
N_ASSIGNMENTS = 20
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def load():
    rows = [r for r in csv.reader(SRC.open()) if r]
    return (np.array([r[2].upper() for r in rows]),
            np.array([int(r[0] == "+") for r in rows]),
            np.array([r[1] for r in rows]))


def grouping(seqs):
    """Connected components over shared exact 25-mers, as in experiment 001."""
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


def extract_embeddings(seqs):
    """Frozen per-layer representations. The model is never updated."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    enc = tok(list(seqs), return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    mask = enc["attention_mask"].unsqueeze(-1).float()
    n_layers = len(out.hidden_states)
    mean_pooled, per_token = {}, {}
    for layer in range(n_layers):
        h = out.hidden_states[layer]
        mean_pooled[layer] = ((h * mask).sum(1) / mask.sum(1)).numpy()
        per_token[layer] = (h * mask).reshape(len(seqs), -1).numpy()
    meta = {
        "model_id": MODEL_ID,
        "license": "cc-by-nc-sa-4.0 (research use only)",
        "n_hidden_states": n_layers,
        "hidden_size": int(out.hidden_states[-1].shape[-1]),
        "n_tokens_per_sequence": int(enc["input_ids"].shape[1]),
        "tokenizer": "6-mer, as published",
        "frozen": True,
    }
    return mean_pooled, per_token, meta


def tune_and_predict_embedding(reps, labels, tr, te, inner):
    """Pick representation and C inside the training folds, then predict."""
    best = (-np.inf, None)
    for rep_name, matrix in reps.items():
        Xtr = matrix[tr]
        for C in C_GRID:
            pipe = Pipeline([("z", StandardScaler()),
                             ("m", LogisticRegression(solver="liblinear",
                                                      C=C, max_iter=5000))])
            scores = []
            for itr, ite in inner:
                pipe.fit(Xtr[itr], labels[tr][itr])
                scores.append(balanced_accuracy_score(labels[tr][ite],
                                                      pipe.predict(Xtr[ite])))
            mean = float(np.mean(scores))
            if mean > best[0]:
                best = (mean, (rep_name, C))
    rep_name, C = best[1]
    matrix = reps[rep_name]
    pipe = Pipeline([("z", StandardScaler()),
                     ("m", LogisticRegression(solver="liblinear", C=C,
                                              max_iter=5000))])
    pipe.fit(matrix[tr], labels[tr])
    return pipe.predict(matrix[te]), pipe.decision_function(matrix[te]), rep_name, C


def run_assignment(seqs, labels, groups, reps, seed):
    outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    classical = {
        "GC_kmer_baseline": (Pipeline([
            ("k", CountVectorizer(analyzer="char", ngram_range=(3, 4),
                                  lowercase=False)),
            ("m", LogisticRegression(solver="liblinear", max_iter=3000))]),
            {"m__C": C_GRID}),
        "positional_onehot": (Pipeline([
            ("o", OneHotPosition()),
            ("m", LogisticRegression(solver="liblinear", max_iter=3000))]),
            {"m__C": C_GRID}),
    }
    pred = {name: np.zeros(len(labels), dtype=int) for name in classical}
    score = {name: np.zeros(len(labels)) for name in classical}
    pred["pretrained_nt_frozen"] = np.zeros(len(labels), dtype=int)
    score["pretrained_nt_frozen"] = np.zeros(len(labels))
    chosen = []

    for fold, (tr, te) in enumerate(outer.split(seqs, labels, groups)):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                        random_state=42 + fold)
        inner = list(inner_cv.split(seqs[tr], labels[tr], groups[tr]))

        for name, (pipe, grid) in classical.items():
            search = GridSearchCV(pipe, grid, scoring="balanced_accuracy",
                                  cv=inner, n_jobs=1)
            fitted = search.fit(seqs[tr], labels[tr]).best_estimator_
            pred[name][te] = fitted.predict(seqs[te])
            score[name][te] = fitted.decision_function(seqs[te])

        p, s, rep_name, C = tune_and_predict_embedding(reps, labels, tr, te, inner)
        pred["pretrained_nt_frozen"][te] = p
        score["pretrained_nt_frozen"][te] = s
        chosen.append({"fold": fold, "representation": rep_name, "C": C})

    return {
        name: {
            "accuracy": float(accuracy_score(labels, pred[name])),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred[name])),
            "roc_auc": float(roc_auc_score(labels, score[name])),
            "n_correct": int((pred[name] == labels).sum()),
        } for name in pred
    }, chosen


def main():
    seqs, labels, ids = load()
    groups = grouping(seqs)
    partition = [sorted(np.where(groups == g)[0].tolist())
                 for g in sorted(set(groups))]
    assert len(partition) == 69

    print(f"extracting frozen embeddings from {MODEL_ID} ...", flush=True)
    mean_pooled, per_token, meta = extract_embeddings(seqs)
    print(f"  {meta['n_hidden_states']} hidden states, "
          f"hidden size {meta['hidden_size']}, "
          f"{meta['n_tokens_per_sequence']} tokens/sequence", flush=True)

    # Offer a spread of depths; the inner CV decides which is used.
    layers = sorted({0, meta["n_hidden_states"] // 4,
                     meta["n_hidden_states"] // 2,
                     3 * meta["n_hidden_states"] // 4,
                     meta["n_hidden_states"] - 1})
    reps = {}
    for layer in layers:
        reps[f"mean_layer{layer}"] = mean_pooled[layer]
        reps[f"token_layer{layer}"] = per_token[layer]
    print(f"  candidate representations: {len(reps)} "
          f"(layers {layers}, mean-pooled and per-token)", flush=True)

    rng = np.random.default_rng(11)
    runs, all_chosen = [], []
    for i in range(N_ASSIGNMENTS):
        order = rng.permutation(len(partition))
        relabelled = np.zeros(len(seqs), dtype=int)
        for new_name, members in zip(order, partition):
            relabelled[members] = new_name
        result, chosen = run_assignment(seqs, labels, relabelled, reps,
                                        seed=2026 + i)
        runs.append(result)
        all_chosen.extend(chosen)
        print(f"  assignment {i+1}/{N_ASSIGNMENTS}: " + "  ".join(
            f"{k}={v['n_correct']}/106" for k, v in result.items()), flush=True)

    names = list(runs[0].keys())
    summary = {}
    for name in names:
        vals = np.array([r[name]["accuracy"] for r in runs])
        aucs = np.array([r[name]["roc_auc"] for r in runs])
        summary[name] = {
            "mean_accuracy": float(vals.mean()),
            "sd_accuracy": float(vals.std(ddof=1)),
            "min_accuracy": float(vals.min()),
            "max_accuracy": float(vals.max()),
            "mean_roc_auc": float(aucs.mean()),
            "mean_n_correct": float(vals.mean() * 106),
        }

    # Paired comparison: same fold assignment, pretrained minus each baseline.
    paired = {}
    nt = np.array([r["pretrained_nt_frozen"]["accuracy"] for r in runs])
    for name in names:
        if name == "pretrained_nt_frozen":
            continue
        other = np.array([r[name]["accuracy"] for r in runs])
        diff = nt - other
        paired[name] = {
            "mean_difference_pp": float(diff.mean() * 100),
            "sd_difference_pp": float(diff.std(ddof=1) * 100),
            "pretrained_wins": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "pretrained_losses": int((diff < 0).sum()),
            "n_assignments": len(diff),
        }

    rep_counts = defaultdict(int)
    for c in all_chosen:
        rep_counts[c["representation"]] += 1

    report = {
        "question": ("Does a pretrained multi-species genomic language model, "
                     "used as a frozen feature extractor, beat tuned classical "
                     "baselines on E. coli promoter recognition?"),
        "model": meta,
        "dataset": "UCI Molecular Biology promoter gene sequences, 106 sequences",
        "protocol": {
            "leakage_control": "connected components over shared exact 25-mers, 69 groups",
            "outer_cv": "4-fold StratifiedGroupKFold",
            "inner_cv": "3-fold StratifiedGroupKFold, on training folds only",
            "tuning": ("every model tunes C; the pretrained model also selects "
                       "its layer and pooling inside the training folds"),
            "n_fold_assignments": N_ASSIGNMENTS,
            "why_repeated": ("experiments/003 showed a single fold assignment "
                             "moves accuracy by several points"),
        },
        "summary": summary,
        "paired_vs_pretrained": paired,
        "representations_selected_by_inner_cv": dict(rep_counts),
        "per_assignment": runs,
    }
    (HERE / "nt_benchmark_results.json").write_text(json.dumps(report, indent=2))

    print("\n--- mean over", N_ASSIGNMENTS, "fold assignments ---")
    for name in names:
        s = summary[name]
        print(f"{name:24} {s['mean_accuracy']*100:6.2f}% "
              f"(sd {s['sd_accuracy']*100:4.2f})  auc {s['mean_roc_auc']:.3f}")
    print("\n--- paired, pretrained minus baseline ---")
    for name, p in paired.items():
        print(f"vs {name:22} {p['mean_difference_pp']:+6.2f} pp   "
              f"wins {p['pretrained_wins']}  ties {p['ties']}  "
              f"losses {p['pretrained_losses']}")
    print("\nwrote nt_benchmark_results.json")


if __name__ == "__main__":
    main()
