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


# ── validate_pairs ───────────────────────────────────────────────────────────

def test_validate_pairs_valid():
    pairs  = np.array([[0, 1], [2, 3]])
    labels = np.array([1, 0])
    # should not raise
    validate_pairs(pairs, labels, "val")


def test_validate_pairs_invalid_split():
    pairs  = np.array([[0, 1]])
    labels = np.array([1])
    with pytest.raises(AssertionError, match="Invalid split name"):
        validate_pairs(pairs, labels, "garbage")


def test_validate_pairs_mismatched_length():
    pairs  = np.array([[0, 1], [2, 3]])
    labels = np.array([1])          # one fewer label
    with pytest.raises(AssertionError):
        validate_pairs(pairs, labels, "train")


def test_validate_pairs_wrong_shape():
    pairs  = np.array([0, 1, 2])    # 1-D instead of (N, 2)
    labels = np.array([1, 0, 1])
    with pytest.raises(AssertionError):
        validate_pairs(pairs, labels, "test")


def test_validate_pairs_non_binary_labels():
    pairs  = np.array([[0, 1], [2, 3]])
    labels = np.array([1, 2])       # 2 is not a valid label
    with pytest.raises(AssertionError):
        validate_pairs(pairs, labels, "val")


def test_validate_pairs_empty():
    pairs  = np.empty((0, 2), dtype=int)
    labels = np.array([])
    with pytest.raises(AssertionError):
        validate_pairs(pairs, labels, "val")


# ── validate_scores ──────────────────────────────────────────────────────────

def test_validate_scores_valid():
    scores = np.array([0.9, 0.5, 0.3])
    pairs  = np.array([[0, 1], [2, 3], [4, 5]])
    validate_scores(scores, pairs)     # should not raise


def test_validate_scores_count_mismatch():
    scores = np.array([0.9, 0.5])
    pairs  = np.array([[0, 1], [2, 3], [4, 5]])
    with pytest.raises(AssertionError):
        validate_scores(scores, pairs)


def test_validate_scores_nan():
    scores = np.array([0.9, float("nan"), 0.3])
    pairs  = np.array([[0, 1], [2, 3], [4, 5]])
    with pytest.raises(AssertionError, match="NaN or Inf"):
        validate_scores(scores, pairs)


def test_validate_scores_inf():
    scores = np.array([0.9, float("inf"), 0.3])
    pairs  = np.array([[0, 1], [2, 3], [4, 5]])
    with pytest.raises(AssertionError, match="NaN or Inf"):
        validate_scores(scores, pairs)


# ── validate_threshold ───────────────────────────────────────────────────────

def test_validate_threshold_valid_float():
    validate_threshold(0.85)     # should not raise


def test_validate_threshold_valid_int():
    validate_threshold(1)        # int is acceptable


def test_validate_threshold_nan():
    with pytest.raises(AssertionError):
        validate_threshold(float("nan"))


def test_validate_threshold_inf():
    with pytest.raises(AssertionError):
        validate_threshold(float("inf"))


def test_validate_threshold_string():
    with pytest.raises(AssertionError):
        validate_threshold("0.5")


# ── validate_metrics ─────────────────────────────────────────────────────────

def _good_metrics():
    return {
        "accuracy": 0.75, "balanced_accuracy": 0.76,
        "true_positive_rate": 0.80, "false_positive_rate": 0.28,
        "precision": 0.70, "f1_score": 0.75,
        "tp": 80, "fp": 20, "tn": 72, "fn": 28, "total": 200,
    }


def test_validate_metrics_valid():
    validate_metrics(_good_metrics())   # should not raise


def test_validate_metrics_missing_key():
    m = _good_metrics()
    del m["f1_score"]
    with pytest.raises(AssertionError, match="Missing metric keys"):
        validate_metrics(m)


def test_validate_metrics_zero_total():
    m = _good_metrics()
    m["total"] = 0
    with pytest.raises(AssertionError):
        validate_metrics(m)
