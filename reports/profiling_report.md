# Profiling Report — Face Verification System

**MSML/MSAI 605 | Milestone 4**
**Authors:** Johnathan Sheikh, Renae Ricketts

---

## 1. Methodology

All measurements were taken on CPU using `time.perf_counter()` for high-resolution wall-clock timing. The profiling script is located at `scripts/profile_inference.py`. Results were saved to `outputs/profiling_results.json`.

**Hardware and environment:**

| Field | Value |
|---|---|
| Platform | Windows 11 (10.0.26200) |
| Processor | AMD64 Family 25 Model 97 (AuthenticAMD) |
| Python version | 3.14.3 |
| PyTorch version | 2.10.0+cpu |
| CUDA available | No |
| Device used | CPU |

**Per-stage latency methodology:**
- Input: synthetic random uint8 images (62x47x3), generated with a fixed seed (42)
- Warmup: 10 runs discarded before measurement
- Measurement: 50 timed runs
- Each run calls `run_inference(img1, img2, model, threshold=0.397)`, which instruments each stage internally with `time.perf_counter()`

**Batch-size sensitivity methodology:**
- Input: 128 synthetic random uint8 images (62x47x3), fixed seed (0)
- Batch sizes tested: 1, 4, 8, 16, 32, 64
- Each batch size repeated 3 times; mean is reported
- Calls `embed_images(images, model, batch_size=bs, device="cpu")` directly to isolate embedding throughput

---

## 2. Per-Stage Single-Pair Latency

Each inference call is decomposed into three stages plus the total end-to-end time:

1. **Preprocess** — resize to 160x160, normalize to [-1, 1], convert to tensor
2. **Embed** — forward pass through FaceNet (InceptionResnetV1) for both images
3. **Score** — cosine similarity between the two 512-D embedding vectors

| Stage | Mean (ms) | Std (ms) | Min (ms) | p50 (ms) | p95 (ms) | Max (ms) |
|---|---|---|---|---|---|---|
| preprocess | 0.722 | 0.061 | 0.658 | 0.686 | 0.808 | 0.832 |
| embed | 56.925 | 4.337 | 49.292 | 56.675 | 62.595 | 74.763 |
| score | 0.074 | 0.006 | 0.065 | 0.071 | 0.084 | 0.088 |
| **total** | **57.724** | **4.354** | **50.149** | **57.483** | **63.437** | **75.631** |

**Key finding:** Embedding generation dominates inference time, accounting for 98.6% of total latency (56.9 ms out of 57.7 ms mean). Preprocessing and cosine similarity scoring together contribute less than 0.8 ms per pair. Any meaningful latency optimization must target the FaceNet forward pass (e.g., batching, quantization, or GPU acceleration).

---

## 3. Batch-Size Sensitivity

128 synthetic images were embedded at varying batch sizes to characterize throughput scaling on CPU.

| Batch Size | Total Time (ms) | Per Image (ms) | Images/sec |
|---|---|---|---|
| 1 | 3540.8 | 27.662 | 36.2 |
| 4 | 1356.3 | 10.596 | 94.4 |
| 8 | 1045.7 | 8.170 | 122.4 |
| 16 | 910.7 | 7.115 | 140.5 |
| 32 | 877.6 | 6.856 | 145.8 |
| 64 | 867.4 | 6.777 | 147.6 |

**Key finding:** Throughput improves substantially from batch size 1 to 16 (36 img/sec to 140 img/sec — a 3.9x speedup), driven by better CPU parallelism across threads. Gains flatten between batch size 16 and 64, with only a 5% additional improvement from 16 to 64. For bulk embedding workloads on CPU, a batch size of 16–32 offers the best balance of throughput and memory use. The diminishing returns above batch size 16 indicate CPU memory bandwidth and thread saturation rather than compute limits.

---

## 4. Summary

The inference pipeline is dominated by the FaceNet forward pass (~57 ms per pair, ~99% of latency). All other stages (preprocessing and cosine scoring) are negligible. On CPU, batching images together accelerates throughput meaningfully up to batch size 16, beyond which gains plateau. The system runs comfortably on CPU-only hardware for research use; GPU acceleration would reduce the embedding stage by an order of magnitude for production-scale workloads.

Raw results are stored in `outputs/profiling_results.json`.
