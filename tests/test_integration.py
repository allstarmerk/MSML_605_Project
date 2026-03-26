import sys
import os
import numpy as np
import pytest

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.evaluation import score_pairs_cosine, apply_threshold, compute_metrics
from src.validation import validate_pairs, validate_scores, validate_threshold, validate_metrics

# ── Synthetic data helpers ────────────────────────────────────────────────────
# Build a tiny fake "dataset" without loading LFW so tests run fast and offline.
# Each "image" is a small flat array. Identical images are the same person.

def make_synthetic_dataset(seed=0):
    rng = np.random.default_rng(seed)
    # 10 identities, 3 images each — images within an identity are very similar
    n_identities = 10
    n_per_id     = 3
    images = []
    labels_per_image = []
    for identity in range(n_identities):
        base = rng.random(128 * 128 * 3).astype(np.float32)
        for _ in range(n_per_id):
            # add tiny noise so images are not identical but very close
            noise = rng.random(128 * 128 * 3).astype(np.float32) * 0.01
            images.append((base + noise).reshape(128, 128, 3))
            labels_per_image.append(identity)
    return np.array(images), np.array(labels_per_image)


