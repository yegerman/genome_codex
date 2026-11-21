#!/usr/bin/env python3
"""Does a pretrained genomic language model beat k-mers on measured activity?

Experiment 007 set the bar: ridge on 1- to 6-mer counts reached holdout
Pearson 0.475 on ENCSR997GVD. This experiment feeds the same sequences, split
and holdout through a pretrained Nucleotide Transformer, frozen, and asks
whether its representation predicts activity better.

PROTOCOL. Written and committed before any embedding was extracted.

Data and split: identical to experiment 007, imported from its script, so the
units, the five chromosome folds and the frozen holdout (chromosomes 7, 13)
are the same rows.

Model: InstaDeepAI/nucleotide-transformer-v2-50m-multi-species, Hub commit
81b29e57 (read and pinned in experiment 004), frozen, never updated. Each
200-base sequence is tokenised as published (6-mers). For hidden-state layers
0, 3, 6, 9 and 12, token states are mean-pooled over real tokens, giving five
512-number representations per sequence. Cached in data/cache/.

Models compared
  nt_ridge        ridge on the standardised mean-pooled embedding; the layer
                  and the ridge strength are chosen together by mean R2 across
                  the five development folds.
  kmer_ridge      experiment 007's model, refitted here with its chosen alpha
                  (1e5) so predictions sit side by side.
  kmer_plus_nt    ridge on k-mer counts and the chosen layer, each block
                  standardised; alpha chosen by the same CV.
Ridge grid {0.1 ... 1e5} in decades, extended by a decade at an edge, at most
three times, as in experiment 007.

Endpoints
  Primary: paired difference in holdout Pearson, nt_ridge minus kmer_ridge,
  with a 95% interval from 1,000 resamples of whole variants. Pretraining is
  called helpful only if the whole interval is above zero.
  Secondary: the same for kmer_plus_nt minus kmer_ridge; Spearman, R2, MAE;
  cross-validation means; the no-shared-30-mer holdout subset.

Not in this experiment: fine-tuning, larger models, variant effects.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "007" / "mpra_baselines"))

import run_baselines as base  # noqa: E402

MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
MODEL_REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"
LAYERS = (0, 3, 6, 9, 12)
KMER_ALPHA = 1e5
CACHE = ROOT / "data" / "cache" / "nt50m_meanpool_ENCSR997GVD.npz"
BATCH = 128


def extract(units):
    """Mean-pooled hidden states for every unit, cached on disk."""
    ids = np.array([u["id"] for u in units])
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=False)
        if np.array_equal(cached["ids"], ids):
            print(f"using cached embeddings {CACHE.name}", flush=True)
            return {layer: cached[f"layer{layer}"] for layer in LAYERS}, {
                "cached": True}

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = base.pick_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                              trust_remote_code=True)
    model, loading = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        use_safetensors=True, output_loading_info=True)
    assert not any(loading.values()), f"incomplete weight load: {loading}"
    model.eval().to(device)

    out = {layer: np.zeros((len(units), 512), dtype=np.float32) for layer in LAYERS}
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(units), BATCH):
            batch = [u["sequence"] for u in units[start:start + BATCH]]
            encoded = tokenizer(batch, return_tensors="pt", padding=True)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded, output_hidden_states=True).hidden_states
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            for layer in LAYERS:
                pooled = (hidden[layer] * mask).sum(1) / mask.sum(1)
                out[layer][start:start + len(batch)] = pooled.float().cpu().numpy()
            if start // BATCH % 50 == 0:
                print(f"  embedded {start + len(batch)}/{len(units)} "
                      f"({time.time() - started:.0f}s)", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, ids=ids, **{f"layer{layer}": out[layer] for layer in LAYERS})
    return out, {"cached": False, "device": device,
                 "tokens_per_sequence": int(encoded["input_ids"].shape[1]),
                 "seconds": round(time.time() - started, 1)}


def main():
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    started = time.time()
    units, flow = base.load_units()
    embeddings, extraction = extract(units)
    is_hold = np.array([u["chr"] in base.HOLDOUT_CHROMOSOMES for u in units])
    dev = [u for u, h in zip(units, is_hold) if not h]
    holdout = [u for u, h in zip(units, is_hold) if h]
    fold_of, _ = base.assign_folds(units)
    folds = np.array([fold_of[u["chr"]] for u in dev])
    clean, audit = base.leakage_audit(dev, holdout)
    y_dev = np.array([u["y"] for u in dev])
    y_hold = np.array([u["y"] for u in holdout])
    groups = [u["variant"] for u in holdout]
    print(f"development {len(dev)}, holdout {len(holdout)}", flush=True)

    results, predictions = {}, {}

    def finish(name, cv, prediction, extra=None):
        entry = {"cross_validation": cv,
                 "holdout": base.score(y_hold, prediction),
                 "holdout_no_shared_kmer": base.score(y_hold[clean], prediction[clean]),
                 "holdout_pearson_ci95": base.bootstrap_pearson(y_hold, prediction, groups)}
        entry.update(extra or {})
        results[name], predictions[name] = entry, prediction
        print(f"  {name:13} holdout Pearson {entry['holdout']['pearson']:.3f}  "
              f"R2 {entry['holdout']['r2']:+.3f}  ({time.time() - started:.0f}s)",
              flush=True)

    def scaled_ridge(alpha):
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    # nt_ridge: choose layer and alpha together
    best = None
    by_layer = {}
    for layer in LAYERS:
        X = embeddings[layer][~is_hold].astype(np.float64)
        alpha, tried = base.ridge_cross_validate(scaled_ridge, X, y_dev, folds)
        cv_r2 = float(np.mean([f["r2"] for f in tried[alpha]]))
        by_layer[layer] = {"alpha": alpha, "mean_cv_r2": cv_r2,
                           "mean_cv_pearson": float(np.mean([f["pearson"] for f in tried[alpha]]))}
        print(f"  layer {layer:>2}: alpha {alpha:g}  CV R2 {cv_r2:.4f}", flush=True)
        if best is None or cv_r2 > best[2]:
            best = (layer, alpha, cv_r2, tried[alpha])
    layer, alpha, _, per_fold = best
    X_dev = embeddings[layer][~is_hold].astype(np.float64)
    X_hold = embeddings[layer][is_hold].astype(np.float64)
    finish("nt_ridge", base.fold_summary(per_fold),
           scaled_ridge(alpha).fit(X_dev, y_dev).predict(X_hold),
           {"layer": layer, "alpha": alpha, "by_layer": by_layer})

    # kmer_ridge: experiment 007's model
    vectoriser = CountVectorizer(analyzer="char", ngram_range=base.KMER_RANGE,
                                 lowercase=False)
    kmer_dev = vectoriser.fit_transform([u["sequence"] for u in dev]).astype(np.float64)
    kmer_hold = vectoriser.transform([u["sequence"] for u in holdout]).astype(np.float64)
    kmer_model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=KMER_ALPHA))
    per_fold = [base.score(y_dev[folds == k],
                           make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=KMER_ALPHA))
                           .fit(kmer_dev[folds != k], y_dev[folds != k])
                           .predict(kmer_dev[folds == k])) for k in range(base.N_FOLDS)]
    finish("kmer_ridge", base.fold_summary(per_fold),
           kmer_model.fit(kmer_dev, y_dev).predict(kmer_hold), {"alpha": KMER_ALPHA})

    # kmer_plus_nt: both blocks, each standardised on development data
    kmer_scale = StandardScaler(with_mean=False).fit(kmer_dev)
    nt_scale = StandardScaler().fit(X_dev)
    both_dev = sparse.hstack([kmer_scale.transform(kmer_dev),
                              sparse.csr_matrix(nt_scale.transform(X_dev))]).tocsr()
    both_hold = sparse.hstack([kmer_scale.transform(kmer_hold),
                               sparse.csr_matrix(nt_scale.transform(X_hold))]).tocsr()
    alpha_both, tried = base.ridge_cross_validate(lambda a: Ridge(alpha=a),
                                                  both_dev, y_dev, folds)
    finish("kmer_plus_nt", base.fold_summary(tried[alpha_both]),
           Ridge(alpha=alpha_both).fit(both_dev, y_dev).predict(both_hold),
           {"alpha": alpha_both, "nt_layer": layer})

    comparisons = {
        "nt_ridge_minus_kmer_ridge": base.paired_bootstrap_difference(
            y_hold, predictions["nt_ridge"], predictions["kmer_ridge"], groups),
        "kmer_plus_nt_minus_kmer_ridge": base.paired_bootstrap_difference(
            y_hold, predictions["kmer_plus_nt"], predictions["kmer_ridge"], groups),
    }
    primary = comparisons["nt_ridge_minus_kmer_ridge"]
    verdict = ("pretraining helped" if primary["ci95_low"] > 0 else
               "pretraining hurt" if primary["ci95_high"] < 0 else
               "no detectable difference")

    report = {
        "question": "Does a frozen pretrained genomic LM beat k-mer counts on MPRA activity?",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "frozen": True,
                  "pooling": "mean over real tokens", "layers_offered": list(LAYERS),
                  "license": "cc-by-nc-sa-4.0 (research use only)", **extraction},
        "unit_flow": flow, "leakage_audit": audit,
        "models": results, "comparisons": comparisons, "primary_verdict": verdict,
        "seconds": round(time.time() - started, 1),
    }
    (HERE / "pretrained_results.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n--- frozen holdout, chromosomes 7 and 13 ---")
    for name, e in results.items():
        h, ci, cv = e["holdout"], e["holdout_pearson_ci95"], e["cross_validation"]
        print(f"{name:13} Pearson {h['pearson']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]  "
              f"R2 {h['r2']:+.3f}  clean {e['holdout_no_shared_kmer']['pearson']:.3f}  "
              f"CV {cv['pearson_mean']:.3f}")
    for name, d in comparisons.items():
        print(f"{name}: {d['difference_in_pearson']:+.3f} "
              f"[{d['ci95_low']:+.3f}, {d['ci95_high']:+.3f}]")
    print(f"primary verdict: {verdict}")


if __name__ == "__main__":
    main()
