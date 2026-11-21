"""Read a DNA window with the positional model trained in experiment 001.

What this is: the logistic-regression weights fitted on 106 UCI *E. coli*
sequences, applied to a 57-base window. It reports a score and shows which
positions drove it, which is the part worth reading.

What this is not, and the numbers are meaningless if it is used as such:

- not a human model. The training data is bacterial.
- not a measure of gene expression. The training labels were promoter versus
  non-promoter, not an amount of anything.
- not calibrated. `experiments/003` showed the headline accuracy moves by
  several points depending on an arbitrary fold assignment, so treat a score
  near the threshold as undecided.
- not evidence about disease, variants or editing.

The window must be 57 unambiguous bases aligned the way the UCI dataset is.
A window taken from somewhere else is outside the model's domain even when
the arithmetic succeeds.
"""
from pathlib import Path

import numpy as np

from .sequence import DNA_BASES, validate

COEFFICIENTS_PATH = (Path(__file__).resolve().parent.parent / "experiments" /
                     "001" / "genetic_ml_pilot" / "model_coefficients.npz")
WINDOW = 57
SCOPE = ("UCI E. coli 57-base promoter recognition only; "
         "not expression, not human, not calibrated")

_CACHE = {}


def load_model(path=COEFFICIENTS_PATH):
    """Load the stored weights as a (57, 4) array plus the intercept."""
    path = Path(path)
    key = str(path)
    if key not in _CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"model coefficients not found at {path}. They ship with "
                f"experiments/001; this module does not fit a new model.")
        data = np.load(path, allow_pickle=False)
        alphabet = "".join(str(b) for b in data["alphabet"])
        if alphabet != DNA_BASES:
            raise ValueError(
                f"stored alphabet {alphabet!r} is not {DNA_BASES!r}; "
                f"the one-hot column order would be wrong")
        length = int(data["input_length"][0])
        if length != WINDOW:
            raise ValueError(f"stored model expects {length} bases, not {WINDOW}")
        _CACHE[key] = {
            "weights": data["coef"].reshape(length, len(DNA_BASES)).astype(float),
            "intercept": float(data["intercept"][0]),
            "window": length,
            "alphabet": alphabet,
            "path": key,
        }
    return _CACHE[key]


def _one_hot(sequence):
    seq = validate(sequence, allow_ambiguous=False)
    if len(seq) != WINDOW:
        raise ValueError(
            f"this model reads exactly {WINDOW} bases, got {len(seq)}. "
            f"Use scan() to slide it along a longer sequence.")
    matrix = np.zeros((WINDOW, len(DNA_BASES)))
    for index, base in enumerate(seq):
        matrix[index, DNA_BASES.index(base)] = 1.0
    return seq, matrix


def score(sequence, model=None):
    """Score one 57-base window.

    `logit` is the raw linear score; positive favours the promoter class.
    `probability` is the logistic transform of it, reported because the model
    emits it, not because it is a calibrated likelihood.
    """
    model = model or load_model()
    seq, matrix = _one_hot(sequence)
    logit = float((matrix * model["weights"]).sum() + model["intercept"])
    return {
        "sequence": seq,
        "logit": logit,
        "probability": float(1.0 / (1.0 + np.exp(-np.clip(logit, -100, 100)))),
        "predicted_class": "promoter-like" if logit >= 0 else "not promoter-like",
        "scope": SCOPE,
    }


def position_contributions(sequence, model=None):
    """How much each position added to the score, given the base observed.

    This is the readable part of the model: a per-position number saying
    which bases pushed the window towards or away from the promoter class.
    Contributions sum to the logit minus the intercept.
    """
    model = model or load_model()
    seq, matrix = _one_hot(sequence)
    values = (matrix * model["weights"]).sum(axis=1)
    return [{"position": index + 1, "base": base, "contribution": float(value)}
            for index, (base, value) in enumerate(zip(seq, values))]


def top_positions(sequence, n=10, model=None):
    """The n positions that moved the score furthest from zero."""
    contributions = position_contributions(sequence, model)
    return sorted(contributions, key=lambda c: abs(c["contribution"]),
                  reverse=True)[:n]


def scan(sequence, step=1, both_strands=True, model=None, top=None):
    """Slide the 57-base window along a longer sequence.

    Returns one result per window, highest score first when `top` is set.
    Window coordinates are 0-based half-open on the strand named by
    `strand`; for the minus strand they are coordinates on the reverse
    complement, not on the input.

    Scores along a genome are not independent and are not corrected for the
    many windows tested, so a high score here is a place to look, not a
    finding.
    """
    from .sequence import reverse_complement

    model = model or load_model()
    forward = validate(sequence, allow_ambiguous=False)
    if len(forward) < WINDOW:
        raise ValueError(
            f"sequence is {len(forward)} bases, shorter than the "
            f"{WINDOW}-base window")
    if step < 1:
        raise ValueError("step must be at least 1")

    strands = [("+", forward)]
    if both_strands:
        strands.append(("-", reverse_complement(forward)))

    hits = []
    for strand, seq in strands:
        for start in range(0, len(seq) - WINDOW + 1, step):
            result = score(seq[start:start + WINDOW], model)
            result.update(strand=strand, start=start, end=start + WINDOW)
            hits.append(result)

    hits.sort(key=lambda hit: hit["logit"], reverse=True)
    return hits[:top] if top else hits
