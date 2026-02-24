import sys
import os
import time
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.similarity import (
    cosine_similarity,
    cosine_similarity_loop,
    euclidean_distance,
    euclidean_distance_loop
)

def timeit(fn, *args, reps=5):
    times = []
    for _ in range(reps): #using _ to show we not using index 
        t0 = time.perf_counter
        fn(*args)
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 100 # in milisec


