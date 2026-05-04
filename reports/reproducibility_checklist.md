# Reproducibility Checklist — Face Verification System

**MSML/MSAI 605 | Milestone 4**
**Authors:** Johnathan Sheikh, Renae Ricketts
**Final Release Tag:** v1.0-final

---

All commands below should be run from the repository root after cloning. No manual file editing is required unless a step says otherwise.

---

## Step 1 — Clone and set up environment

```bash
git clone <repo-url>
cd MSML_605_Project
git checkout v1.0-final

# Create virtual environment and install dependencies
python -m venv .venv
# Windows:
.venv\Scripts\Activate
# macOS/Linux:
source .venv/bin/activate

# Install CPU-only PyTorch first (required for facenet-pytorch on CPU-only machines)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Expected: no import errors. `python -c "import torch; import facenet_pytorch; print('OK')"` should print `OK`.

---

## Step 2 — Download LFW dataset and generate pairs

```bash
python scripts/lfw_ingestion_script.py
```

Expected output:
- Downloads LFW dataset (~170 MB) to `data/`
- Creates `data/manifest.json`
- Creates `data/pairs/train_pairs.npy`, `data/pairs/train_labels.npy`
- Creates `data/pairs/val_pairs.npy`, `data/pairs/val_labels.npy`
- Creates `data/pairs/test_pairs.npy`, `data/pairs/test_labels.npy`

Verify pair counts:
```bash
python -c "
import numpy as np
for split in ['train', 'val', 'test']:
    labels = np.load(f'data/pairs/{split}_labels.npy')
    print(f'{split}: {len(labels)} pairs, {labels.sum()} positive, {(labels==0).sum()} negative')
"
```

Expected:
```
train: 3000 pairs, 1500 positive, 1500 negative
val:   1000 pairs, 500 positive, 500 negative
test:  1000 pairs, 500 positive, 500 negative
```

---

## Step 3 — Run all tests

```bash
pytest tests/ -v
```

Expected: all tests pass. Tests do not require the LFW dataset — the integration test uses a synthetic fixture and runs fully offline.

---

## Step 4 — Run threshold sweep on val split

```bash
python scripts/threshold_sweep.py --run-id run_sweep_val --note "Threshold sweep on val split"
```

Expected: prints the optimal threshold. The threshold 0.3970 should be selected (maximizes balanced accuracy on val split). This run is appended to `outputs/runs.jsonl`.

Confirm `configs/eval.yaml` has:
```yaml
selected_threshold: 0.3970
```

If not, update it to match the printed value before Step 5.

---

## Step 5 — Evaluate on val and test splits

```bash
python scripts/run_evaluation.py \
  --val-run-id run_val_final \
  --test-run-id run_test_final \
  --note "Final evaluation at threshold 0.3970"
```

Expected test-split metrics (Run 11 reference values):

| Metric | Expected Value |
|---|---|
| Balanced Accuracy | 0.9790 |
| True Positive Rate (TPR) | 0.9740 |
| False Positive Rate (FPR) | 0.0160 |
| F1 Score | 0.9789 |
| True Positives | 487 |
| False Positives | 8 |
| True Negatives | 492 |
| False Negatives | 13 |

Outputs written:
- `outputs/confusion_matrix_val.png`
- `outputs/confusion_matrix_test.png`
- Entry appended to `outputs/runs.jsonl`

---

## Step 6 — Run CLI inference (single pair)

```bash
python scripts/infer.py --img1 <path/to/face1.jpg> --img2 <path/to/face2.jpg>
```

Expected: JSON output with `decision`, `score`, and `confidence` fields. Example:
```json
{
  "decision": "same person",
  "score": 0.712,
  "confidence": 0.582
}
```

---

## Step 7 — Run load test

```bash
python scripts/load_test.py --n-requests 50 --workers 4 --synthetic
```

Expected: prints latency statistics (mean ~57 ms per pair on a modern CPU without Docker overhead), throughput ~17 pairs/sec with 4 workers on synthetic images. The `--synthetic` flag avoids needing the LFW dataset for this step.

Results saved to `outputs/load_test_results.json`.

---

## Step 8 — Run profiling script

```bash
python scripts/profile_inference.py
```

Expected output: per-stage latency table and batch-size sensitivity table. Reference values:

| Stage | Mean (ms) |
|---|---|
| preprocess | ~0.7 |
| embed | ~57 |
| score | ~0.07 |
| total | ~58 |

| Batch Size | Images/sec |
|---|---|
| 1 | ~36 |
| 16 | ~140 |
| 64 | ~148 |

Results saved to `outputs/profiling_results.json`.

---

## Step 9 — Docker build and test

```bash
# Build the Docker image (downloads FaceNet weights during build)
docker build -t face-verifier .

# Run tests inside Docker
docker run face-verifier pytest tests/ -v

# Run synthetic load test inside Docker
docker run face-verifier python scripts/load_test.py --n-requests 10 --workers 2 --synthetic

# Confirm CLI help works
docker run face-verifier python scripts/infer.py --help
```

Expected: all tests pass, load test completes, help text is printed. The Docker image runs fully offline after build (weights are baked in).

---

## Artifact Locations

| Artifact | Path |
|---|---|
| System Card | `reports/system_card.md` |
| Profiling Report | `reports/profiling_report.md` |
| Reproducibility Checklist | `reports/reproducibility_checklist.md` (this file) |
| Run log | `outputs/runs.jsonl` |
| Profiling results | `outputs/profiling_results.json` |
| Load test results | `outputs/load_test_results.json` |
| Confusion matrix (val) | `outputs/confusion_matrix_val.png` |
| Confusion matrix (test) | `outputs/confusion_matrix_test.png` |
| Dataset config | `configs/dataset.yaml` |
| Eval config (threshold) | `configs/eval.yaml` |
| Dockerfile | `Dockerfile` |
