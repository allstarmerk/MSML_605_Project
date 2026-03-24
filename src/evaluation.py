# will take pairs, images, and threshold and produce metrics
import numpy as np
from src.similarity import cosine_similarity, flatten_embedding

def score_pairs(images, pairs):
    #compute the cosine similarity score for every pair
    a = flatten_embedding(images[pairs])
    b = flatten_embedding(images[pairs[:, 1] ])
    return cosine_similarity(a, b) #higher = more similar, lower = less similar?