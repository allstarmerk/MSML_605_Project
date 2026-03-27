import sys
import os
import numpy as np
import pytest

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.validation import (
    validate_pairs,
    validate_scores,
    validate_threshold,
    validate_metrics,
)


# ── validate_pairs ──────────────────
# validate_pairs checks that a pair array and its labels are well-formed before
# any scoring or evaluation runs. It guards against shape mismatches, bad split
# names, non binary labels, and empty inputs all of which would cause silent
# wrong results downstream if not caught here.

def test_validate_pairs_valid():
    # well-formed input: correct shape (N, 2), matching label count, binary
    # labels, and a valid split name should pass without raising
    pairs  = np.array([[0, 1], [2, 3]])
    labels = np.array([1, 0])
    validate_pairs(pairs, labels, "val")


def test_validate_pairs_invalid_split():
    # split name must be one of "train", "val", or "test" :  anything else
    # should raise so we catch typos before a whole evaluation run wastes time
    pairs  = np.array([[0, 1]])
    labels = np.array([1])
    with pytest.raises(AssertionError, match="Invalid split name"):
        validate_pairs(pairs, labels, "garbage")


def test_validate_pairs_mismatched_length():
    # every pair needs exactly one label : if counts differ the pair file is
    # corrupted or was loaded incorrectly, so this must be caught immediately
    pairs  = np.array([[0, 1], [2, 3]])
    labels = np.array([1])          # one fewer label than pairs
    with pytest.raises(AssertionError):
        validate_pairs(pairs, labels, "train")


def test_validate_pairs_wrong_shape():
    # pairs must be 2-D with shape (N, 2) :  a flat 1-D array means the .npy
    # file was generated incorrectly and index lookups would silently break
    pairs  = np.array([0, 1, 2])    # 1-D instead of (N, 2)
    labels = np.array([1, 0, 1])
    with pytest.raises(AssertionError):
        validate_pairs(pairs, labels, "test")


def test_validate_pairs_non_binary_labels():
    # labels must be 0 (different person) or 1 (same person)  any other value
    # means the pair file was created with the wrong schema
    pairs  = np.array([[0, 1], [2, 3]])
    labels = np.array([1, 2])       # 2 is not a valid label
    with pytest.raises(AssertionError):
        validate_pairs(pairs, labels, "val")


def test_validate_pairs_empty():
    # an empty pair array means ingestion failed or produced no pairs-
    # running evaluation on zero pairs would produce meaningless metrics
    pairs  = np.empty((0, 2), dtype=int)
    labels = np.array([])
    with pytest.raises(AssertionError):
        validate_pairs(pairs, labels, "val")


# ── validate_scores ───────────────────────────
# validate_scores checks the output of the scoring step before it is passed to
# the threshold and metric functions. It ensures one score exists per pair and
# that no math errors (division by zero, overflow) slipped through as NaN/Infinity.

def test_validate_scores_valid():
    # normal finite scores with one score per pair- should pass without raising
    scores = np.array([0.9, 0.5, 0.3])
    pairs  = np.array([[0, 1], [2, 3], [4, 5]])
    validate_scores(scores, pairs)


def test_validate_scores_count_mismatch():
    # if the scorer skipped a pair or returned an extra score the counts won't
    # match — catch this before threshold application produces wrong predictions
    scores = np.array([0.9, 0.5])           # only 2 scores for 3 pairs
    pairs  = np.array([[0, 1], [2, 3], [4, 5]])
    with pytest.raises(AssertionError):
        validate_scores(scores, pairs)


def test_validate_scores_nan():
    # NaN in scores usually means a zero-vector was passed to cosine similarity
    # (division by zero) must be caught so metrics aren't silently computed
    # on garbage values
    scores = np.array([0.9, float("nan"), 0.3])
    pairs  = np.array([[0, 1], [2, 3], [4, 5]])
    with pytest.raises(AssertionError, match="NaN or Inf"):
        validate_scores(scores, pairs)


def test_validate_scores_inf():
    # Inf in scores indicates a numerical overflow in the distance calculation
    # same risk as NaN: downstream metrics would be meaningless
    scores = np.array([0.9, float("inf"), 0.3])
    pairs  = np.array([[0, 1], [2, 3], [4, 5]])
    with pytest.raises(AssertionError, match="NaN or Inf"):
        validate_scores(scores, pairs)


# ── validate_threshold ──────────
# validate_threshold checks the threshold value loaded from eval.yaml before it
# is applied to scores. A bad threshold value (wrong type, NaN, Inf) would
# silently produce all-same or all-different predictions without any obvious error.

def test_validate_threshold_valid_float():
    # standard float threshold — the typical case loaded from eval.yaml
    validate_threshold(0.85)


def test_validate_threshold_valid_int():
    # integers are also acceptable since they are numeric and finite
    validate_threshold(1)


def test_validate_threshold_nan():
    # NaN would make every comparison False, classifying all pairs as different
    # person must be rejected before apply_threshold runs
    with pytest.raises(AssertionError):
        validate_threshold(float("nan"))


def test_validate_threshold_inf():
    # Inf would make every score fall below threshold, classifying all pairs as
    # different person  same silent failure mode as NaN
    with pytest.raises(AssertionError):
        validate_threshold(float("inf"))


def test_validate_threshold_string():
    # a string threshold usually means eval.yaml was edited incorrectly and the
    # value was not parsed as a number  catch this before a type error mid-run
    with pytest.raises(AssertionError):
        validate_threshold("0.5")


# ── validate_metrics ────────
# validate_metrics checks the dictionary returned by compute_metrics before it
# is logged or written to a report. It ensures all expected keys are present and
# that the total pair count is non-zero so logged values are meaningful.

def _good_metrics():
    # reusable fixture with all required keys and valid values
    return {
        "accuracy": 0.75, "balanced_accuracy": 0.76,
        "true_positive_rate": 0.80, "false_positive_rate": 0.28,
        "precision": 0.70, "f1_score": 0.75,
        "tp": 80, "fp": 20, "tn": 72, "fn": 28, "total": 200,
    }


def test_validate_metrics_valid():
    # complete metrics dict with a non-zero total  should pass without raising
    validate_metrics(_good_metrics())


def test_validate_metrics_missing_key():
    # if compute_metrics ever drops a key (e.g. during a refactor) the log file
    # and report would be silently incomplete catch it at the source
    m = _good_metrics()
    del m["f1_score"]
    with pytest.raises(AssertionError, match="Missing metric keys"):
        validate_metrics(m)


def test_validate_metrics_zero_total():
    # total = 0 means no pairs were evaluated, making every rate undefined 
    # this would log 0.0 for all metrics and look like a legitimate result
    m = _good_metrics()
    m["total"] = 0
    with pytest.raises(AssertionError):
        validate_metrics(m)
