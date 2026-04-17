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

# Stage 1: Preprocessing
 # Stage 2: Embedding generation
 # Stage 3: Similarity scoring
  # Stage 4: Threshold decision
  # Stage 5: Confidence computation
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
