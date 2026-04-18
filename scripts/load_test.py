import sys
import os
import time
import json
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

import yaml
from src.embeddings import load_model
from src.inference import run_inference


def load_lfw_pairs(n_pairs, seed=42):
    from src.data_ingestion import load_config, load_lfw
    config = load_config()
    pairs_dir = Path(config["paths"]["pairs_dir"])
    pairs  = np.load(pairs_dir / "val_pairs.npy")
    labels = np.load(pairs_dir / "val_labels.npy")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=min(n_pairs, len(pairs)), replace=False)
    splits, _ = load_lfw(config)
    val_images = splits["val"][0]
    selected = pairs[idx]
    tasks = [(val_images[selected[i, 0]], val_images[selected[i, 1]])
             for i in range(len(selected))]
    return tasks


def make_synthetic_pairs(n_pairs, seed=42):
    # generates random (H, W, C) uint8 arrays — no dataset required
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n_pairs):
        img1 = (rng.random((62, 47, 3)) * 255).astype(np.uint8)
        img2 = (rng.random((62, 47, 3)) * 255).astype(np.uint8)
        tasks.append((img1, img2))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Concurrency load test for face verification inference."
    )
    parser.add_argument("--n-requests", type=int, default=50,
                        help="Number of inference requests to run (default: 50)")
    parser.add_argument("--workers",    type=int, default=4,
                        help="Number of concurrent worker threads (default: 4)")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Use random synthetic images instead of loading LFW (Docker-friendly)")
    parser.add_argument("--output",     type=str, default="outputs/load_test_results.json",
                        help="Path to save JSON results (default: outputs/load_test_results.json)")
    args = parser.parse_args()

    eval_cfg_path = Path(root_path) / "configs" / "eval.yaml"
    with open(eval_cfg_path) as f:
        eval_cfg = yaml.safe_load(f)
    threshold = eval_cfg["selected_threshold"]

    print("Loading FaceNet model...")
    model = load_model()

    if args.synthetic:
        print(f"Generating {args.n_requests} synthetic pairs...")
        tasks = make_synthetic_pairs(args.n_requests, seed=42)
    else:
        print(f"Loading {args.n_requests} LFW val pairs...")
        tasks = load_lfw_pairs(args.n_requests, seed=42)

    print(f"\nLoad test: {args.n_requests} requests | {args.workers} workers | threshold={threshold:.4f}\n")

    latencies = []
    errors    = 0
    t_wall_start = time.perf_counter()

    def run_one(task):
        img1, img2 = task
        try:
            result = run_inference(img1, img2, model, threshold)
            return result["latency_ms"]["total"], None
        except Exception as e:
            return None, str(e)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            latency, err = future.result()
            if err:
                errors += 1
            else:
                latencies.append(latency)

    t_wall = time.perf_counter() - t_wall_start
    latencies_sorted = sorted(latencies)
    n_success   = len(latencies_sorted)
    throughput  = n_success / t_wall if t_wall > 0 else 0.0
    mean_lat    = float(np.mean(latencies_sorted))  if latencies_sorted else 0.0
    p50         = float(np.percentile(latencies_sorted, 50))  if latencies_sorted else 0.0
    p95         = float(np.percentile(latencies_sorted, 95))  if latencies_sorted else 0.0
    p99         = float(np.percentile(latencies_sorted, 99))  if latencies_sorted else 0.0

    print(f"-- Load Test Results -----------------------------")
    print(f"  Total requests:  {args.n_requests}")
    print(f"  Successful:      {n_success}")
    print(f"  Errors:          {errors}")
    print(f"  Wall-clock time: {t_wall:.2f} s")
    print(f"  Throughput:      {throughput:.2f} requests/sec")
    print(f"  Latency (ms):")
    print(f"    Mean:          {mean_lat:.1f}")
    print(f"    p50:           {p50:.1f}")
    print(f"    p95:           {p95:.1f}")
    print(f"    p99:           {p99:.1f}")

    results = {
        "config": {
            "n_requests": args.n_requests,
            "workers":    args.workers,
            "threshold":  threshold,
            "synthetic":  args.synthetic,
        },
        "results": {
            "total_requests":  args.n_requests,
            "successful":      n_success,
            "errors":          errors,
            "wall_clock_s":    round(t_wall,      3),
            "throughput_rps":  round(throughput,  3),
            "latency_ms": {
                "mean": round(mean_lat, 2),
                "p50":  round(p50,      2),
                "p95":  round(p95,      2),
                "p99":  round(p99,      2),
            },
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
