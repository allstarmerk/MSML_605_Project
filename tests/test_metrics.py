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



