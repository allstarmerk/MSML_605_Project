import sys
import os
import time
import numpy as np
#Going to be for comparing the python loop doing the cosin and euclidean vs vectorized way of doing it with numpy
# compares the timing and corectness 

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


def check(a, b, label):
    ok = np.allclose(a, b, atol=1e-4)
    sym = "Pass" if ok else "Fail"
    print(f" {sym} correctness [{label}] max diff: {np.max(np.abs(a-b)): .2e}")


def main():
    rng = np.random.default_rng(42)

    print("similarity benchmark: using Loop vs Numpy")
    print("=" * 55)

    for n, d in [(100, 512), (1000, 512), (5000, 512)]:
        a = rng.random((n,d)).astype(np.float32)
        b = rng.random((n, d)).astype(np.float32)

        print(f"\n[N={n}, D={d}]")

        t_cos_loop = timeit(cosine_similarity_loop, a, b)
        t_cos_vec = timeit(cosine_similarity, a, b)
        print(f" Cosine loop: {t_cos_loop:7.2f} MS")
        print(f" Cosine Numpy: {t_cos_vec:7.2f} MS  to {t_cos_loop / t_cos_vec:.1f} speedup")
        check(cosine_similarity_loop(a, b), cosine_similarity(a, b), "Cosine")

        t_euc_loop = timeit(euclidean_distance_loop, a, b)
        t_euc_vec = timeit(euclidean_distance, a, b)
        print(f" Euclid loop: {t_euc_loop:7.2f} MS")
        print(f" Euclid Numpy: {t_euc_vec:7.2f} MS  to {t_euc_loop / t_euc_vec:.1f} speedup")
        check(euclidean_distance_loop(a, b), euclidean_distance(a, b), "Euclidean")

    print("\n additional checks")
    x =  rng.random((10, 64)).astype(np.float32)
    sim = cosine_similarity(x, x)
    print(f" {'Pass' if np.allclose(sim, 1.0, atol=1e-5) else 'Fail'} identical vectors to cosine = 1")

    dist = euclidean_distance(x, x)
    print(f"") #line 71




