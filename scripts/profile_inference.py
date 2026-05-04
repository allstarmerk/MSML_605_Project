import sys
import os
import time
import json
import platform
import numpy as np
from pathlib import Path

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

import torch
from src.embeddings import load_model, embed_images
from src.inference import run_inference


def get_hardware_info():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": "cpu",
    }


def make_synthetic_pair(rng):
    img1 = (rng.random((62, 47, 3)) * 255).astype(np.uint8)
    img2 = (rng.random((62, 47, 3)) * 255).astype(np.uint8)
    return img1, img2


def profile_single_pair_latency(model, n_warmup=10, n_runs=50):
    rng = np.random.default_rng(42)

    print(f"  Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        img1, img2 = make_synthetic_pair(rng)
        run_inference(img1, img2, model, threshold=0.397)

    print(f"  Measuring ({n_runs} runs)...")
    records = []
    for _ in range(n_runs):
        img1, img2 = make_synthetic_pair(rng)
        result = run_inference(img1, img2, model, threshold=0.397)
        records.append(result["latency_ms"])

    stages = ["preprocess", "embed", "score", "total"]
    summary = {}
    for stage in stages:
        vals = [r[stage] for r in records]
        summary[stage] = {
            "mean_ms":  round(float(np.mean(vals)),             3),
            "std_ms":   round(float(np.std(vals)),              3),
            "min_ms":   round(float(np.min(vals)),              3),
            "p50_ms":   round(float(np.percentile(vals, 50)),   3),
            "p95_ms":   round(float(np.percentile(vals, 95)),   3),
            "max_ms":   round(float(np.max(vals)),              3),
        }
    return summary


def profile_batch_sensitivity(model, batch_sizes=(1, 4, 8, 16, 32, 64),
                               n_images=128, n_repeats=3):
    rng = np.random.default_rng(0)
    images = (rng.random((n_images, 62, 47, 3)) * 255).astype(np.uint8)

    results = {}
    for bs in batch_sizes:
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            embed_images(images, model, batch_size=bs, device="cpu")
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times))
        results[str(bs)] = {
            "batch_size":       bs,
            "total_ms_mean":    round(mean_ms,                      2),
            "per_image_ms":     round(mean_ms / n_images,           3),
            "images_per_sec":   round(n_images / (mean_ms / 1000),  1),
        }
    return results


def print_stage_table(stage_latency):
    print(f"\n{'Stage':<15} {'Mean':>8} {'Std':>8} {'p50':>8} {'p95':>8}  (ms)")
    print("-" * 52)
    for stage, s in stage_latency.items():
        print(f"  {stage:<13} {s['mean_ms']:>8.2f} {s['std_ms']:>8.2f} "
              f"{s['p50_ms']:>8.2f} {s['p95_ms']:>8.2f}")


def print_batch_table(batch_results):
    print(f"\n{'Batch':>6}  {'Total (ms)':>11}  {'Per image (ms)':>15}  {'img/sec':>8}")
    print("-" * 46)
    for _, s in batch_results.items():
        print(f"  {s['batch_size']:>4}  {s['total_ms_mean']:>11.1f}  "
              f"{s['per_image_ms']:>15.3f}  {s['images_per_sec']:>8.1f}")


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    print("Loading FaceNet model (cpu)...")
    model = load_model(device="cpu")

    print("\nHardware info:")
    hw = get_hardware_info()
    for k, v in hw.items():
        print(f"  {k}: {v}")

    print("\n-- Per-stage single-pair latency --")
    stage_latency = profile_single_pair_latency(model, n_warmup=10, n_runs=50)
    print_stage_table(stage_latency)

    print("\n-- Batch-size sensitivity (128 synthetic images, 3 repeats) --")
    batch_results = profile_batch_sensitivity(model)
    print_batch_table(batch_results)

    output = {
        "hardware":              hw,
        "per_stage_latency_ms":  stage_latency,
        "batch_size_sensitivity": batch_results,
        "methodology": {
            "timing":           "time.perf_counter()",
            "n_warmup":         10,
            "n_measurement":    50,
            "input":            "synthetic random uint8 images (62x47x3)",
            "threshold":        0.397,
            "device":           "cpu",
            "batch_n_images":   128,
            "batch_repeats":    3,
        },
    }

    out_path = out_dir / "profiling_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
