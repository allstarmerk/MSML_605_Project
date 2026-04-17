# What this file does-
#   This is the core of the Milestone 3 system. It takes two face images and runs
#   them through the full verification pipeline, returning a same/different decision
#   along with a score, and confidence value.

#   compute_confidence(score, threshold)
#       Takes the raw cosine similarity score and the threshold and
#       returns a confidence value between 0.0 and 1.0. 0.0 means the score is
#       right at the decision boundary (uncertain). 1.0 means maximum certainty.
#       Works in both directions same and different decisions.

#   run_inference(img1, img2, model, threshold, device)
#       Runs the full 5-stage pipeline on a pair of images and returns a dict 

# Pipeline stages :
#  1  resize + normalize each image to a (3,160,160) tensor
#  2 pass each tensor through FaceNet → 512-D vector
#  3 cosine similarity between the two 512-D vectors
#  4 score >= threshold → "same", else "different"
#  5  compute_confidence(score, threshold)

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


def run_inference(img1, img2, model, threshold, device="cpu"):
    
    t_total_start = time.perf_counter()
    #steps needed
    # Stage 1: Preprocessing
    t0 = time.perf_counter()
    tensor1 = preprocess_image(img1)
    tensor2 = preprocess_image(img2)
    t_preprocess_ms = (time.perf_counter() - t0) * 1000

    # Stage 2: Embedding generation
    t0 = time.perf_counter()
    emb1 = embed_single(tensor1, model, device)   # (512,)
    emb2 = embed_single(tensor2, model, device)   # (512,)
    t_embed_ms = (time.perf_counter() - t0) * 1000

    # Stage 3: Similarity scoring
    t0 = time.perf_counter()
    score = float(cosine_similarity(emb1[None], emb2[None])[0])
    t_score_ms = (time.perf_counter() - t0) * 1000

    # Stage 4: Threshold decision
    decision = "same" if score >= threshold else "different"

    # Stage 5: Confidence computation
    confidence = compute_confidence(score, threshold)

    t_total_ms = (time.perf_counter() - t_total_start) * 1000

    return {
        "score":      round(score,      6),
        "threshold":  threshold,
        "decision":   decision,
        "confidence": round(confidence, 6),
        "latency_ms": {
            "preprocess": round(t_preprocess_ms, 3),
            "embed":      round(t_embed_ms,      3),
            "score":      round(t_score_ms,      3),
            "total":      round(t_total_ms,      3),
        },
    }
