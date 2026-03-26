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


def make_synthetic_pairs(identity_labels, num_pairs=100, positive_ratio=0.5, seed=1):
    rng = np.random.default_rng(seed)
    n_positive = int(num_pairs * positive_ratio)
    n_negative = num_pairs - n_positive

    id_to_indices = {}
    for idx, identity in enumerate(identity_labels):
        id_to_indices.setdefault(identity, []).append(idx)

    identities = list(id_to_indices.keys())
    pairs, labels = [], []

    # positive pairs: two images from same identity
    for _ in range(n_positive):
        identity = rng.choice(identities)
        i, j = rng.choice(id_to_indices[identity], size=2, replace=False)
        pairs.append([i, j])
        labels.append(1)

    # negative pairs: two images from different identities
    for _ in range(n_negative):
        id_a, id_b = rng.choice(identities, size=2, replace=False)
        i = rng.choice(id_to_indices[id_a])
        j = rng.choice(id_to_indices[id_b])
        pairs.append([i, j])
        labels.append(0)

    return np.array(pairs), np.array(labels)


# ── Integration tests ─────────────────────────────────────────────────────────

class TestEndToEndPipeline:

    def setup_method(self):
        self.images, self.id_labels = make_synthetic_dataset(seed=42)
        self.pairs, self.labels = make_synthetic_pairs(
            self.id_labels, num_pairs=100, positive_ratio=0.5, seed=99
        )

    def test_cosine_same_person_higher_than_different(self):
        # same-person pairs should have higher cosine similarity on average
        scores = score_pairs_cosine(self.images, self.pairs)
        same_mean  = scores[self.labels == 1].mean()
        diff_mean  = scores[self.labels == 0].mean()
        assert same_mean > diff_mean, (
            f"Same-person mean ({same_mean:.4f}) should exceed "
            f"different-person mean ({diff_mean:.4f})"
        )

    def test_full_pipeline_smoke(self):
        # run the complete pipeline without any assertion errors end-to-end
        validate_pairs(self.pairs, self.labels, "test")
        scores = score_pairs_cosine(self.images, self.pairs)
        validate_scores(scores, self.pairs)

        threshold = 0.9990
        validate_threshold(threshold)

        predictions = apply_threshold(scores, threshold, higher_is_similar=True)
        metrics     = compute_metrics(self.labels, predictions)
        validate_metrics(metrics)

        # sanity check: counts must add up to total
        assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == metrics["total"]
