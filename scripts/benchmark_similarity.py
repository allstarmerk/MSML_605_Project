import sys
import os
import time
import numpy as np
#Going to be for comparing the python loop doing the cosin and euclidean vs vectorized way of doing it with numpy
# compares the timing and corectness 

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, root_path)

from src.similarity import (
    cosine_similarity,
    cosine_similarity_loop,
    euclidean_distance,
    euclidean_distance_loop
)

def timeit(fn, *args, reps=5):
    times = []
    for _ in range(reps): #using _ to show we not using index 
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000 # in milisec


def check(a, b, label):
    ok = np.allclose(a, b, atol=1e-4)
    sym = "PASS" if ok else "FAIL"
    print(f" Small Difference Tolerance Check [{label}]: {sym}\n Max Difference: {np.max(np.abs(a-b)): .2e}")


def main():
    rng = np.random.default_rng(42)

    print("Similarity Benchmark: using Loop vs Numpy")
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
        print(f" \nEuclid loop: {t_euc_loop:7.2f} MS")
        print(f" Euclid Numpy: {t_euc_vec:7.2f} MS  to {t_euc_loop / t_euc_vec:.1f} speedup")
        check(euclidean_distance_loop(a, b), euclidean_distance(a, b), "Euclidean")

    print("\nAdditional Checks: unit tests for the math for each")
    x =  rng.random((10, 64)).astype(np.float32)
    sim = cosine_similarity(x, x)
    print(f" Identical Vector Check (Cosine = 1): {'PASS' if np.allclose(sim, 1.0, atol=1e-5) else 'FAIL'}")

    dist = euclidean_distance(x, x)
    print(f" Identical Vector Check (Euclidean = 0): {'PASS' if  np.allclose(dist, 0.0, atol=1e-5) else 'FAIL' }"  ) 

    e1 = np.array([[1., 0., 0.]], dtype=np.float32)
    e2 = np.array([[0., 1., 0.]], dtype=np.float32)
    print(f" Orthogonal Vector Check (Cosine = 0): {'PASS' if np.allclose(cosine_similarity(e1, e2), 0., atol=1e-5) else 'FAIL'}")


    big_a = rng.standard_normal((500, 256)).astype(np.float32)
    big_b = rng.standard_normal((500, 256)).astype(np.float32)
    sims = cosine_similarity(big_a, big_b)
    print(f" Cosine Range in [-1, 1]: {'PASS' if np.all(sims >= -1-1e-5) and np.all(sims <= 1+1e-5) else 'FAIL'}")

    print("\nSimilarity Benchmark Complete")

if __name__ == "__main__":
    main()
