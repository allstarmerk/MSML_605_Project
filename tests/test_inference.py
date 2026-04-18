import sys
import os
import argparse
import numpy as np
import pytest
import torch

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.inference import compute_confidence, run_inference
from src.similarity import cosine_similarity



# This file tests the Milestone 3 inference pipeline.
#
#   1. compute_confidence  — tests the function that turns a raw score into a
#                            human-readable confidence value should between 0 to 1. Not to be confused with the input going into compute_confidence (cosine similarity ranges -1 to 1 raw score). Output is confidence it ranges from 0-1 and we need to test the diffrent test cases bellow


# ── 1. compute_confidence tests

# compute_confidence(score, threshold) returns a float in [0.0, 1.0].
# 0.0 = score is right at the boundary (maximum uncertainty).
# 1.0 = score is at the extreme end (maximum certainty).
# It works in both directions:
#   - "same" side:      score >= threshold, mapped toward 1.0
#   - "different" side: score <  threshold, mapped toward -1.0

arbitrary_threshold = 0.5

# Need tp write a test that checks confidence is exactly 0.0 when score == threshold
# can use the logic of --> compute_confidence(0.5, threshold=0.5) should return 0.0
confidence_value = compute_confidence(0.5, arbitrary_threshold)
print(confidence_value)

#need to  write a test that checks confidence is 1.0 when score == 1.0 (max same certainty)
# can use logic of --> compute_confidence(1.0, threshold=0.5) should return 1.0
confidence_value = compute_confidence(1.0, arbitrary_threshold)
print(confidence_value)

# write a test that checks confidence is 1.0 when score == -1.0 (max different certainty)
#  compute_confidence(-1.0, threshold=0.5) should return ~1.0
confidence_value = compute_confidence(-1.0, arbitrary_threshold)
print(confidence_value)

# write a test that checks confidence increases the farther score is from the threshold
#  compute_confidence(0.9, 0.5) should be greater than compute_confidence(0.6, 0.5)
confidence_increase = compute_confidence(0.9, arbitrary_threshold) > compute_confidence(0.6, arbitrary_threshold)
print("Comparing confidence values of scores 0.9 and 0.6 w/ threshold of 0.5")
print(f"score 0.9 confid_value: {compute_confidence(0.9, arbitrary_threshold)}")
print(f"score 0.6 confid_value: {compute_confidence(0.6, arbitrary_threshold)}")
print(f"Confidence value increases when score is farther away from threshold: {confidence_increase}")

#  write a test that checks confidence is always between 0.0 and 1.0
# Logic can be used--> loop over several score values like [-1.0, -0.5, 0.0, 0.3, 0.8, 1.0] and assert 0 <= c <= 1
test_scores = [-1.0, -0.5, 0.0, 0.3, 0.8, 1.0]

# tests still using arbitrary threshold of 0.5
cond = True
for score in test_scores:
    con_value = compute_confidence(score, arbitrary_threshold)
    if con_value < 0 or 1 < con_value:
        cond = False
        break

print(f"Confidence values all remain in range 0 <= con_value <= 1: {cond}")


# ── Formal pytest test functions (same logic as above, wrapped for pytest)
# The code above explores and prints the values manually.
# The functions below turn that same logic into actual pytest tests so the
# test suite can verify these conditions automatically on every run.

def test_confidence_at_boundary():
    assert compute_confidence(0.5, threshold=0.5) == 0.0


def test_confidence_same_at_max():
    assert compute_confidence(1.0, threshold=0.5) == 1.0


def test_confidence_different_at_max():
    assert compute_confidence(-1.0, threshold=0.5) == pytest.approx(1.0, abs=1e-6)


def test_confidence_increases_away_from_boundary():
    assert compute_confidence(0.9, 0.5) > compute_confidence(0.6, 0.5)


def test_confidence_always_in_range():
    for score in [-1.0, -0.5, 0.0, 0.3, 0.8, 1.0]:
        c = compute_confidence(score, threshold=0.5)
        assert 0.0 <= c <= 1.0, f"confidence out of range for score={score}: got {c}"


# cosine similarity on toy embeddings 

def test_cosine_identical_embeddings():
    # identical embeddings → cosine similarity = 1.0
    rng = np.random.default_rng(0)
    emb = rng.random((1, 512)).astype(np.float32)
    sim = cosine_similarity(emb, emb)
    assert np.allclose(sim, 1.0, atol=1e-5)


def test_cosine_orthogonal_embeddings():
    # orthogonal embeddings → cosine similarity = 0.0
    a = np.zeros((1, 4), dtype=np.float32); a[0, 0] = 1.0
    b = np.zeros((1, 4), dtype=np.float32); b[0, 1] = 1.0
    sim = cosine_similarity(a, b)
    assert np.allclose(sim, 0.0, atol=1e-5)


