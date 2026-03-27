import sys
import os
import numpy as np
import pytest

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.evaluation import apply_threshold, compute_metrics


# ── apply_threshold ──────
# apply_threshold converts continuous similarity scores into binary same/different
# predictions by comparing each score to a cutoff value. It supports both cosine
# similarity (higher = more similar) and distance-based metrics (lower = more
# similar). These tests verify the boundary behavior and both scoring directions.

def test_apply_threshold_higher_is_similar_above():
    # cosine mode: scores above threshold predict same person (1), below predict
    # different person (0)  verifies the correct side of the boundary is labeled
    scores = np.array([0.95, 0.80, 0.70])
    preds = apply_threshold(scores, threshold=0.75, higher_is_similar=True)
    np.testing.assert_array_equal(preds, [1, 1, 0])



def test_apply_threshold_lower_is_similar():
    # distance mode (e.g. euclidean): a lower score means more similar, so scores
    # at or below the threshold predict same person: verifies the mode flag works
    scores = np.array([0.10, 0.50, 0.90])
    preds = apply_threshold(scores, threshold=0.50, higher_is_similar=False)
    np.testing.assert_array_equal(preds, [1, 1, 0])


# ── compute_metrics ────
# compute_metrics takes ground-truth labels and binary predictions and returns
# the full set of evaluation metrics used in the report and run logs. These tests
# use small toy inputs with known answers so we can verify exact values rather
# than relying on the function to check itself.

def test_compute_metrics_perfect():
    # all predictions match labels every metric should be at its best possible
    # value (accuracy=1, TPR=1, FPR=0) and confusion matrix counts must be exact
    labels      = np.array([1, 1, 0, 0])
    predictions = np.array([1, 1, 0, 0])
    m = compute_metrics(labels, predictions)

    assert m["accuracy"]           == 1.0
    assert m["balanced_accuracy"]  == 1.0
    assert m["true_positive_rate"] == 1.0
    assert m["false_positive_rate"] == 0.0
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["tn"] == 2
    assert m["fn"] == 0



def test_compute_metrics_required_keys():
    # the dict returned by compute_metrics must always contain every key that
    # validate_metrics and the run logger expect: a missing key would silently
    # produce an incomplete log entry or crash the report generation
    labels      = np.array([1, 0, 1, 0])
    predictions = np.array([1, 0, 0, 1])
    m = compute_metrics(labels, predictions)

    required = {
        "accuracy", "balanced_accuracy", "true_positive_rate",
        "false_positive_rate", "precision", "f1_score",
        "tp", "fp", "tn", "fn", "total",
    }
    assert required.issubset(set(m.keys()))



def test_compute_metrics_f1_no_positive_predictions():
    # when the model predicts all negative, precision and F1 are undefined (0/0)
    #  the function must return 0.0 for both rather than raising a ZeroDivisionError
    labels      = np.array([1, 1, 0, 0])
    predictions = np.array([0, 0, 0, 0])
    m = compute_metrics(labels, predictions)
    assert m["f1_score"]  == 0.0
    assert m["precision"] == 0.0
