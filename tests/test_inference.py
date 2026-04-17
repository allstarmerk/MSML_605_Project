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

# Need tp write a test that checks confidence is exactly 0.0 when score == threshold
# can use the logic of --> compute_confidence(0.5, threshold=0.5) should return 0.0

#need to  write a test that checks confidence is 1.0 when score == 1.0 (max same certainty)
# can use logic of --> compute_confidence(1.0, threshold=0.5) should return 1.0

# write a test that checks confidence is 1.0 when score == -1.0 (max different certainty)
#  compute_confidence(-1.0, threshold=0.5) should return ~1.0

# write a test that checks confidence increases the farther score is from the threshold
#  compute_confidence(0.9, 0.5) should be greater than compute_confidence(0.6, 0.5)

#  write a test that checks confidence is always between 0.0 and 1.0
# Logic can be used--> loop over several score values like [-1.0, -0.5, 0.0, 0.3, 0.8, 1.0] and assert 0 <= c <= 1


