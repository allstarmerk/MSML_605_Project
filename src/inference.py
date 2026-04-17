import time
import numpy as np
from src.embeddings import preprocess_image, embed_single
from src.similarity import cosine_similarity


def compute_confidence(score, threshold):
    # Normalized distance from the decision boundary.
   # -1 to 0 to 1
    if score >= threshold:
        denom = max(1.0 - threshold, 1e-10)
        return float(min(1.0, (score - threshold) / denom))
    else:
        denom = max(threshold + 1.0, 1e-10)
        return float(min(1.0, (threshold - score) / denom))


