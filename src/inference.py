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
   

    # Stage 3: Similarity scoring
    

    # Stage 4: Threshold decision
    
    # Stage 5: Confidence computation
    
