#!/usr/bin/env python3
"""Baselines: how well does DNA sequence predict measured MPRA activity?

Data: ENCODE ENCSR997GVD, K562, obtained and verified in experiments/006.
Target: log2FoldChange, the measured reporter activity of a 200-base sequence.

PROTOCOL. Written and committed before any model was fitted. Deviations, if
any, are listed in the README under "Deviations".

Units
  1. Rows that join to a sequence and have a measured log2FoldChange and lfcSE.
  2. Library controls (project starting "ctrl") are removed from training and
     evaluation alike.
  3. Only 200-base sequences are kept, so every model sees the same rows.
  4. Identical sequences are collapsed to one unit with the mean activity.

Split, by chromosome, because allele pairs, windows and duplicated entries of
one locus always share a chromosome:
  - Frozen holdout: chromosomes 7 and 13, chosen before looking at any
    outcome. Used once, for the final numbers. Never used to choose anything.
  - Development: the other 20 chromosomes, in 5 folds of whole chromosomes,
    assigned largest-first to the currently smallest fold.

Leakage audit, reported not hidden:
  - No sequence may appear on both sides of the holdout split (asserted).
  - Holdout sequences sharing an exact 30-mer with any development sequence,
    on either strand, are counted. Metrics are reported on the whole holdout
    and on the subset with no shared 30-mer.

Models
  mean         predicts the development mean. The floor.
  gc           linear regression on GC fraction.
  kmer_ridge   ridge on standardised counts of all 1- to 6-mers (5,460).
  onehot_ridge ridge on the 800 position-by-base indicators.
  cnn          three convolution blocks (128 filters; widths 9, 7, 5), max
               and mean pooling, one hidden layer of 128. AdamW 1e-3, weight
               decay 1e-4, batch 256, at most 25 epochs, early stopping with
               patience 4 on a validation fold.
  Ridge strength is chosen by mean R2 across the 5 development folds from
  {0.1 ... 1e5} in decades; if the choice lands on an edge the grid grows by
  a decade, at most three times.
  CNN cross-validation: fold k is the test fold, fold k+1 the validation
  fold. CNN holdout: fold 0 validates, folds 1-4 train, seeds 11, 12, 13;
  the three seeds are reported singly and averaged as an ensemble.

Endpoints
  Primary: Pearson r between predicted and measured activity on the holdout.
  Secondary: Spearman, R2, mean absolute error; the same on the no-shared-
  30-mer subset; mean and sd across development folds.
  Uncertainty: 95% interval from 1,000 bootstrap resamples of whole variants
  (not rows), since alleles and windows of one variant are not independent.
  Ceiling: measurement error bounds any model. With reliability
  1 - mean(lfcSE^2) / var(activity), the best achievable R2 is about the
  reliability and the best Pearson about its square root. Approximate: it
  trusts the reported standard errors.

Not in this experiment: variant-effect prediction, pretrained models, the
second library (ENCSR820RPI). Each is a separate step.
"""
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from dnakit import read, reverse_complement  # noqa: E402

DATA = ROOT / "data" / "mpra" / "ENCSR997GVD"
TABLE, FASTA = "ENCFF612YRJ.tsv", "ENCFF621JOY.fasta.gz"

HOLDOUT_CHROMOSOMES = ("7", "13")
N_FOLDS = 5
LENGTH = 200
KMER_RANGE = (1, 6)
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
MAX_GRID_EXTENSIONS = 3
LEAK_K = 30
SEEDS = (11, 12, 13)
BOOTSTRAPS = 1000
CNN = {"filters": 128, "widths": (9, 7, 5), "hidden": 128, "dropout": 0.3,
       "lr": 1e-3, "weight_decay": 1e-4, "batch": 256, "max_epochs": 25,
       "patience": 4}


# ------------------------------------------------------------------ data
def number(text):
    if text in ("", "NA", "NaN", "."):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_units():
    """Apply the unit rules in order and record how many survive each."""
    with (DATA / TABLE).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    sequences = {}
    for record in read(DATA / FASTA):
        ids = (record.id[1:-1].split(";")
               if record.id.startswith("(") and record.id.endswith(")")
               else [record.id])
        for identifier in ids:
            sequences[identifier] = record.sequence

    flow = {"table_rows": len(rows)}
    kept = [r for r in rows if r["ID"] in sequences
            and number(r["log2FoldChange"]) is not None
            and number(r["lfcSE"]) is not None]
    flow["joined_with_measured_activity_and_se"] = len(kept)
    kept = [r for r in kept if not r["project"].startswith("ctrl")]
    flow["after_removing_controls"] = len(kept)
    kept = [r for r in kept if len(sequences[r["ID"]]) == LENGTH]
    flow["after_requiring_200_bases"] = len(kept)

    by_sequence = defaultdict(list)
    for row in kept:
        by_sequence[sequences[row["ID"]]].append(row)

    units = []
    for sequence, members in by_sequence.items():
        chromosomes = {m["chr"] for m in members}
        assert len(chromosomes) == 1, "one sequence on two chromosomes"
        activities = [float(m["log2FoldChange"]) for m in members]
        errors = [float(m["lfcSE"]) for m in members]
        units.append({
            "id": members[0]["ID"],
            "variant": members[0]["SNP"],
            "chr": members[0]["chr"],
            "sequence": sequence,
            "y": sum(activities) / len(activities),
            "se": math.sqrt(sum(e * e for e in errors)) / len(errors),
        })
    flow["unique_sequences"] = len(units)
    units.sort(key=lambda u: u["id"])
    return units, flow


def assign_folds(units):
    """Whole chromosomes into folds, largest first into the smallest fold."""
    sizes = Counter(u["chr"] for u in units
                    if u["chr"] not in HOLDOUT_CHROMOSOMES)
    fold_of, totals = {}, [0] * N_FOLDS
    for chromosome, count in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        fold = totals.index(min(totals))
        fold_of[chromosome] = fold
        totals[fold] += count
    return fold_of, totals


def leakage_audit(dev, holdout):
    """Which holdout sequences share an exact 30-mer with development data?"""
    shared_exact = {u["sequence"] for u in dev} & {u["sequence"] for u in holdout}
    assert not shared_exact, "a sequence sits on both sides of the split"

    owners = defaultdict(set)
    for index, unit in enumerate(holdout):
        sequence = unit["sequence"]
        for start in range(LENGTH - LEAK_K + 1):
            owners[sequence[start:start + LEAK_K]].add(index)

    touched = set()
    for unit in dev:
        for strand in (unit["sequence"], reverse_complement(unit["sequence"])):
            for start in range(LENGTH - LEAK_K + 1):
                hit = owners.get(strand[start:start + LEAK_K])
                if hit:
                    touched |= hit
    clean = np.array([i not in touched for i in range(len(holdout))])
    return clean, {
        "identical_sequences_across_split": 0,
        "kmer_length": LEAK_K,
        "strands_checked": "both",
        "holdout_sequences": len(holdout),
        "holdout_sharing_a_kmer_with_development": len(touched),
        "holdout_sharing_fraction": len(touched) / len(holdout),
        "holdout_clean": int(clean.sum()),
    }


# -------------------------------------------------------------- features
def one_hot(sequences):
    lookup = np.full(256, -1, dtype=np.int64)
    for index, base in enumerate("ACGT"):
        lookup[ord(base)] = index
    codes = lookup[np.frombuffer("".join(sequences).encode(), dtype=np.uint8)]
    assert (codes >= 0).all(), "non-ACGT letter in a sequence"
    codes = codes.reshape(len(sequences), LENGTH)
    out = np.zeros((len(sequences), LENGTH, 4), dtype=np.float32)
    rows = np.arange(len(sequences))[:, None]
    out[rows, np.arange(LENGTH)[None, :], codes] = 1.0
    return out


def gc_fraction(sequences):
    return np.array([[(s.count("G") + s.count("C")) / len(s)] for s in sequences])


# --------------------------------------------------------------- metrics
def score(y, predicted):
    from scipy.stats import rankdata

    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = float(((y - predicted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    out = {"n": int(len(y)),
           "r2": 1.0 - residual / total,
           "mae": float(np.abs(y - predicted).mean()),
           "pearson": None, "spearman": None}
    if predicted.std() > 0:
        out["pearson"] = float(np.corrcoef(y, predicted)[0, 1])
        out["spearman"] = float(np.corrcoef(rankdata(y), rankdata(predicted))[0, 1])
    return out


def bootstrap_pearson(y, predicted, groups):
    """95% interval resampling whole variants, which are the independent unit."""
    if np.std(predicted) == 0:
        return None
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    blocks = [np.array(v) for v in members.values()]
    rng = np.random.default_rng(2026)
    values = []
    for _ in range(BOOTSTRAPS):
        picks = rng.integers(0, len(blocks), len(blocks))
        index = np.concatenate([blocks[i] for i in picks])
        values.append(np.corrcoef(y[index], predicted[index])[0, 1])
    low, high = np.percentile(values, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "resamples": BOOTSTRAPS,
            "unit": "variant", "n_variants": len(blocks)}


def noise_ceiling(y, se):
    reliability = 1.0 - float(np.mean(se ** 2)) / float(np.var(y))
    return {"reliability": reliability,
            "max_r2": reliability,
            "max_pearson": math.sqrt(max(reliability, 0.0)),
            "mean_standard_error": float(np.mean(se)),
            "sd_activity": float(np.std(y)),
            "note": "approximate; trusts the reported lfcSE"}


def fold_summary(per_fold):
    keys = ("pearson", "spearman", "r2", "mae")
    out = {"per_fold": per_fold}
    for key in keys:
        values = [f[key] for f in per_fold if f[key] is not None]
        if values:
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


# ---------------------------------------------------------------- models
def ridge_cross_validate(make_model, X, y, folds):
    """Choose alpha by mean R2 across development folds; grow the grid if
    the best value sits on an edge."""
    grid, tried = list(ALPHAS), {}
    for _ in range(MAX_GRID_EXTENSIONS + 1):
        for alpha in grid:
            if alpha in tried:
                continue
            per_fold = []
            for fold in range(N_FOLDS):
                train, test = folds != fold, folds == fold
                model = make_model(alpha).fit(X[train], y[train])
                per_fold.append(score(y[test], model.predict(X[test])))
            tried[alpha] = per_fold
        best = max(tried, key=lambda a: np.mean([f["r2"] for f in tried[a]]))
        ordered = sorted(tried)
        if best == ordered[0]:
            grid = [ordered[0] / 10.0]
        elif best == ordered[-1]:
            grid = [ordered[-1] * 10.0]
        else:
            break
    return best, tried


def build_cnn():
    import torch
    from torch import nn

    class SequenceCNN(nn.Module):
        def __init__(self):
            super().__init__()
            filters, widths = CNN["filters"], CNN["widths"]
            layers, channels = [], 4
            for position, width in enumerate(widths):
                layers += [nn.Conv1d(channels, filters, width, padding=width // 2),
                           nn.BatchNorm1d(filters), nn.ReLU()]
                if position < len(widths) - 1:
                    layers.append(nn.MaxPool1d(2))
                channels = filters
            self.features = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.Linear(2 * filters, CNN["hidden"]), nn.ReLU(),
                nn.Dropout(CNN["dropout"]), nn.Linear(CNN["hidden"], 1))

        def forward(self, x):
            hidden = self.features(x)
            pooled = torch.cat([hidden.amax(dim=2), hidden.mean(dim=2)], dim=1)
            return self.head(pooled).squeeze(1)

    return SequenceCNN()


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_cnn(X_train, y_train, X_val, y_val, seed, device):
    """Train with early stopping on the validation fold; return a predictor."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, sd = float(y_train.mean()), float(y_train.std())

    def to_tensor(X):
        return torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1))).to(device)

    train_x, val_x = to_tensor(X_train), to_tensor(X_val)
    train_y = torch.from_numpy(((y_train - mean) / sd).astype(np.float32)).to(device)

    model = build_cnn().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=CNN["lr"],
                                  weight_decay=CNN["weight_decay"])
    loss_function = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)

    def predict(tensor):
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(tensor), 2048):
                chunks.append(model(tensor[start:start + 2048]).float().cpu().numpy())
        return np.concatenate(chunks) * sd + mean

    best_r, best_state, best_epoch, stale, history = -np.inf, None, 0, 0, []
    for epoch in range(1, CNN["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator).to(device)
        for start in range(0, len(order), CNN["batch"]):
            batch = order[start:start + CNN["batch"]]
            optimiser.zero_grad()
            loss = loss_function(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimiser.step()
        val_r = float(np.corrcoef(y_val, predict(val_x))[0, 1])
        history.append(val_r)
        if val_r > best_r + 1e-4:
            best_r, best_epoch, stale = val_r, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= CNN["patience"]:
                break
    model.load_state_dict(best_state)

    def predictor(X):
        return predict(to_tensor(X))

    return predictor, {"best_epoch": best_epoch, "epochs_run": len(history),
                       "best_validation_pearson": best_r,
                       "validation_pearson_by_epoch": history}


# ------------------------------------------------------------------ main
def main():
    import sklearn
    import torch
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    started = time.time()
    units, flow = load_units()
    dev = [u for u in units if u["chr"] not in HOLDOUT_CHROMOSOMES]
    holdout = [u for u in units if u["chr"] in HOLDOUT_CHROMOSOMES]
    fold_of, fold_sizes = assign_folds(units)
    folds = np.array([fold_of[u["chr"]] for u in dev])
    print(f"units {len(units)}: development {len(dev)}, holdout {len(holdout)}",
          flush=True)
    print(f"fold sizes {fold_sizes}", flush=True)

    clean, audit = leakage_audit(dev, holdout)
    print(f"leakage audit: {audit['holdout_sharing_a_kmer_with_development']} of "
          f"{len(holdout)} holdout sequences share a {LEAK_K}-mer with "
          f"development ({audit['holdout_sharing_fraction']:.1%})", flush=True)

    y_dev = np.array([u["y"] for u in dev])
    y_hold = np.array([u["y"] for u in holdout])
    se_hold = np.array([u["se"] for u in holdout])
    groups_hold = [u["variant"] for u in holdout]
    dev_sequences = [u["sequence"] for u in dev]
    hold_sequences = [u["sequence"] for u in holdout]

    ceiling = {"holdout": noise_ceiling(y_hold, se_hold),
               "development": noise_ceiling(y_dev, np.array([u["se"] for u in dev]))}
    print(f"noise ceiling on holdout: Pearson about "
          f"{ceiling['holdout']['max_pearson']:.3f}", flush=True)

    results, predictions = {}, {}

    def finish(name, cv, holdout_prediction, extra=None):
        entry = {"cross_validation": cv,
                 "holdout": score(y_hold, holdout_prediction),
                 "holdout_no_shared_kmer": score(y_hold[clean],
                                                 holdout_prediction[clean]),
                 "holdout_pearson_ci95": bootstrap_pearson(
                     y_hold, holdout_prediction, groups_hold)}
        entry.update(extra or {})
        results[name] = entry
        predictions[name] = holdout_prediction
        h = entry["holdout"]
        pearson = "  n/a " if h["pearson"] is None else f"{h['pearson']:.3f}"
        print(f"  {name:13} holdout Pearson {pearson}  R2 {h['r2']:+.3f}  "
              f"({time.time() - started:.0f}s)", flush=True)

    # mean: the floor
    per_fold = [score(y_dev[folds == k],
                      np.full((folds == k).sum(), y_dev[folds != k].mean()))
                for k in range(N_FOLDS)]
    finish("mean", fold_summary(per_fold), np.full(len(holdout), y_dev.mean()))

    # GC content
    gc_dev, gc_hold = gc_fraction(dev_sequences), gc_fraction(hold_sequences)
    per_fold = [score(y_dev[folds == k],
                      LinearRegression().fit(gc_dev[folds != k], y_dev[folds != k])
                      .predict(gc_dev[folds == k])) for k in range(N_FOLDS)]
    finish("gc", fold_summary(per_fold),
           LinearRegression().fit(gc_dev, y_dev).predict(gc_hold))

    # k-mer counts
    vectoriser = CountVectorizer(analyzer="char", ngram_range=KMER_RANGE,
                                 lowercase=False)
    kmer_dev = vectoriser.fit_transform(dev_sequences).astype(np.float64)
    kmer_hold = vectoriser.transform(hold_sequences).astype(np.float64)

    def kmer_model(alpha):
        return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=alpha))

    alpha, tried = ridge_cross_validate(kmer_model, kmer_dev, y_dev, folds)
    finish("kmer_ridge", fold_summary(tried[alpha]),
           kmer_model(alpha).fit(kmer_dev, y_dev).predict(kmer_hold),
           {"alpha": alpha, "n_features": int(kmer_dev.shape[1]),
            "mean_cv_r2_by_alpha": {str(a): float(np.mean([f["r2"] for f in v]))
                                    for a, v in sorted(tried.items())}})

    # positional one-hot
    hot_dev, hot_hold = one_hot(dev_sequences), one_hot(hold_sequences)
    flat_dev = hot_dev.reshape(len(dev), -1).astype(np.float64)
    flat_hold = hot_hold.reshape(len(holdout), -1).astype(np.float64)
    alpha, tried = ridge_cross_validate(lambda a: Ridge(alpha=a), flat_dev,
                                        y_dev, folds)
    finish("onehot_ridge", fold_summary(tried[alpha]),
           Ridge(alpha=alpha).fit(flat_dev, y_dev).predict(flat_hold),
           {"alpha": alpha, "n_features": int(flat_dev.shape[1]),
            "mean_cv_r2_by_alpha": {str(a): float(np.mean([f["r2"] for f in v]))
                                    for a, v in sorted(tried.items())}})

    # CNN
    device = pick_device()
    print(f"  cnn on {device}", flush=True)
    per_fold, training_logs = [], []
    for fold in range(N_FOLDS):
        validation = (fold + 1) % N_FOLDS
        train = (folds != fold) & (folds != validation)
        predictor, log = train_cnn(hot_dev[train], y_dev[train],
                                   hot_dev[folds == validation],
                                   y_dev[folds == validation], SEEDS[0], device)
        per_fold.append(score(y_dev[folds == fold], predictor(hot_dev[folds == fold])))
        training_logs.append({"test_fold": fold, "validation_fold": validation, **log})
        print(f"    cv fold {fold}: Pearson {per_fold[-1]['pearson']:.3f} "
              f"(best epoch {log['best_epoch']}, {time.time() - started:.0f}s)",
              flush=True)

    seed_predictions, seed_scores, seed_logs = [], [], []
    for seed in SEEDS:
        predictor, log = train_cnn(hot_dev[folds != 0], y_dev[folds != 0],
                                   hot_dev[folds == 0], y_dev[folds == 0],
                                   seed, device)
        prediction = predictor(hot_hold)
        seed_predictions.append(prediction)
        seed_scores.append({"seed": seed, **score(y_hold, prediction)})
        seed_logs.append({"seed": seed, **log})
        print(f"    seed {seed}: holdout Pearson {seed_scores[-1]['pearson']:.3f} "
              f"({time.time() - started:.0f}s)", flush=True)
    ensemble = np.mean(seed_predictions, axis=0)
    pearsons = [s["pearson"] for s in seed_scores]
    finish("cnn", fold_summary(per_fold), ensemble,
           {"holdout_is": "average of the three seeds' predictions",
            "single_seeds": seed_scores,
            "single_seed_pearson_mean": float(np.mean(pearsons)),
            "single_seed_pearson_sd": float(np.std(pearsons, ddof=1)),
            "cv_training": training_logs, "holdout_training": seed_logs,
            "settings": CNN, "device": device})

    report = {
        "question": ("How well does a 200-base DNA sequence predict its "
                     "measured MPRA activity in K562?"),
        "dataset": {"experiment": "ENCSR997GVD", "table": TABLE, "fasta": FASTA,
                    "target": "log2FoldChange"},
        "unit_flow": flow,
        "split": {"holdout_chromosomes": list(HOLDOUT_CHROMOSOMES),
                  "development_units": len(dev), "holdout_units": len(holdout),
                  "holdout_variants": len(set(groups_hold)),
                  "fold_of_chromosome": fold_of, "fold_sizes": fold_sizes},
        "leakage_audit": audit,
        "noise_ceiling": ceiling,
        "models": results,
        "environment": {"numpy": np.__version__, "scikit_learn": sklearn.__version__,
                        "torch": torch.__version__, "device": device,
                        "seconds": round(time.time() - started, 1)},
    }
    (HERE / "baseline_results.json").write_text(json.dumps(report, indent=2) + "\n")

    with (HERE / "holdout_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        names = list(predictions)
        writer.writerow(["id", "variant", "chr", "no_shared_kmer",
                         "measured_log2FoldChange", "lfcSE", *names])
        for index, unit in enumerate(holdout):
            writer.writerow([unit["id"], unit["variant"], unit["chr"],
                             int(clean[index]), f"{unit['y']:.6f}",
                             f"{unit['se']:.6f}",
                             *[f"{predictions[n][index]:.6f}" for n in names]])

    print("\n--- frozen holdout, chromosomes "
          + " and ".join(HOLDOUT_CHROMOSOMES) + " ---")
    print(f"{'model':13} {'Pearson':>8} {'95% CI':>15} {'Spearman':>9} "
          f"{'R2':>7} {'MAE':>6} {'CV Pearson':>14}")
    for name, entry in results.items():
        h, cv, ci = entry["holdout"], entry["cross_validation"], entry["holdout_pearson_ci95"]
        if h["pearson"] is None:
            print(f"{name:13} {'n/a':>8} {'':>15} {'n/a':>9} "
                  f"{h['r2']:>+7.3f} {h['mae']:>6.3f} {'n/a':>14}")
            continue
        print(f"{name:13} {h['pearson']:>8.3f} "
              f"{'[%.3f, %.3f]' % (ci['low'], ci['high']):>15} "
              f"{h['spearman']:>9.3f} {h['r2']:>+7.3f} {h['mae']:>6.3f} "
              f"{'%.3f +/- %.3f' % (cv['pearson_mean'], cv['pearson_sd']):>14}")
    print(f"noise ceiling: Pearson about {ceiling['holdout']['max_pearson']:.3f}, "
          f"R2 about {ceiling['holdout']['max_r2']:.3f}")
    print("wrote baseline_results.json and holdout_predictions.csv")


if __name__ == "__main__":
    main()
