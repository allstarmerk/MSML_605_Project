# MSML_605_Project
MSML/MSAI project. It's a facial recognition system similar to faceId. Given two face images, the system produces a similarity score and makes a same-person vs. different-person decision using a calibrated threshold. Built on the LFW (Labeled Faces in the Wild) dataset.

- Group: Johnathan Sheikh, Renae Ricketts

- Main branch is for after bugs are fixed in working branch. Then we merge the working branch into the main branch. This allows for the main branch to be bug free and always in working condition. Then the end product will be the main branch. Working branch is essentially the development branch and main being the production branch.

- After you have work completed on the working branch you create a pull request. If u wanna use the UI of github you go to pull request. Click on create new pull request. Then you select the working branch and the main branch. Then you click on create pull request. Then you can add a description of the changes you made. Then you can click on create pull request again. Then the pull request will be created and you can see it in the pull request tab. Then you can merge the pull request if there are no conflicts. If there are conflicts then you have to resolve them before merging.

---

## Repository Structure

```
MSML_605_Project-1/
├── configs/          # dataset.yaml (splits, pairs, seeds) and eval.yaml (threshold settings)
├── data/             # LFW images and generated pair files (gitignored — generated locally)
├── outputs/          # Generated plots, confusion matrices, and run logs
├── reports/          # Milestone 2 evaluation report PDF and figures
├── scripts/          # Entry-point scripts for ingestion, sweeps, evaluation, and benchmarks
├── src/              # Core importable modules (evaluation, validation, similarity, ingestion)
├── tests/            # Unit tests and integration test
└── requirements.txt
```

---

How to run:

1. To set up the environment, run the following commands:
```
git clone <repo url>
cd MSML_605_Project
python -m venv .venv   (or for Windows you may have to use -->  py -m venv .venv)
.venv\Scripts\Activate
pip install -r requirements.txt
```
<br/>

2. Ingest LFW dataset and generate pairs, run the following:
```
python scripts/lfw_ingestion_script.py
```
- This should download lfw data set (170MB) and saves/creates data/manifest.json & data/pairs/
- Generates train (3000 pairs), val (1000 pairs), and test (1000 pairs) splits — all 50/50 positive/negative
- Uses seed 42 for both the train/val/test identity split and pair generation, ensuring full reproducibility
<br/>


3. To verify the pair splits, run the following:
```
python -c "import numpy as np; pairs = np.load('data/pairs/train_pairs.npy'); labels = np.load('data/pairs/train_labels.npy'); print('pairs shape:', pairs.shape); print('positive pairs:', labels.sum()); print('negative pairs:', (labels==0).sum())"
```

Expected Output:
  - pairs shape: (3000, 2)
  - positive pairs: 1500
  - negative pairs: 1500
<br/>


4. To verify determinism, run ingestion script a 2nd time:
```
python scripts/lfw_ingestion_script.py
```

- Then compare new data/manifest.json with the 1st run's manifest summary printed on the command terminal and check if they have identical counts to confirm determinism
<br/>


5. To run the Similarity Benchmark test, run the following:
```
python scripts/benchmark_similarity.py
```

- This will provide an output showing the comparison of using the loop vs Numpy for both cosine and euclidean calculations. It also has a unit test at the end checking the math of the math functions.
<br/>

6. To run all tests (unit tests + integration test):
```
pytest tests/
```
- Tests do not require the LFW dataset — the integration test uses a small synthetic fixture and runs fully offline
- Expected: all tests pass in a few seconds
<br/>

Design Choices:
  - We use a hard coded seed in config/ to meet the deterministic requirement. Seed is fixed for numpy, tensorflow, and python random.

  - Saved pair indices are saved as .npy files so evaluation is always accurate and reproducible. Pairs store image indices, not file paths, so the same pair file works regardless of where the dataset is stored locally.

  - Most of the settings needed are in /configs/dataset.yaml (splits, seeds, pair counts, image size, data-centric cap) and /configs/eval.yaml (threshold sweep range, selected threshold, scoring direction).

  - Cosine similarity is used as the scoring function (higher score = more similar). The threshold is applied as: score >= threshold → same person. This direction is documented in eval.yaml under `score_direction`.

  - The pipeline includes validation checks at every major step (pair file existence, pair shape and label validity, score count and finiteness, threshold numeric validity, and metric key completeness) so errors surface early with clear messages rather than silently producing wrong results.

---

## Milestone 2 — Evaluation Loop

> **Note:** Complete all Milestone 1 steps first (environment setup, LFW ingestion, and pair generation — steps 1-4 above) before running any Milestone 2 commands.

Builds a reproducible evaluation loop on top of the Milestone 1 pipeline. Includes threshold calibration on the validation split, experiment tracking, error analysis, and tests.

**Threshold selection rule:** Maximize balanced accuracy on the validation split. Balanced accuracy averages TPR and TNR, making it robust to class imbalance. The selected threshold is stored in `configs/eval.yaml` under `selected_threshold`. It is chosen on val and then applied to test without further tuning.

**Data-centric improvement:** Capped overrepresented identities at 4 images each before generating pairs. Identity 1871 had 530 images and dominated the pair distribution — without the cap, a disproportionate share of pairs involved that one identity, skewing both training and evaluation. Controlled via `max_images_per_identity` in configs/dataset.yaml — set to 999 for baseline (no cap), 4 for the improved version.

### Milestone 2 — How to run

**Baseline (runs 1-3):**
```
# Set max_images_per_identity: 999 in configs/dataset.yaml first
python scripts/lfw_ingestion_script.py
python scripts/threshold_sweep.py --run-id run1_sweep_val_baseline --note "Baseline threshold sweep on val split"
# Update selected_threshold in configs/eval.yaml with the value printed above
python scripts/run_evaluation.py --val-run-id run2_val_selected_threshold --test-run-id run3_test_final_baseline --note "Baseline evaluation"
```

**After data-centric improvement (runs 4-5):**

> **Threshold selection rule (same as baseline):** Maximize balanced accuracy on the validation split. Update `selected_threshold` in configs/eval.yaml with the value printed by the sweep before running evaluation.

```
# Set max_images_per_identity: 4 in configs/dataset.yaml first
python scripts/lfw_ingestion_script.py
python scripts/threshold_sweep.py --run-id run4_sweep_val_post_cap --note "Sweep after capping identities at 4 images"
# Update selected_threshold in configs/eval.yaml with the value printed above
python scripts/run_evaluation.py --val-run-id run5_val_post_cap --test-run-id run5_test_post_cap --note "Run 5 - Evaluation after capping identities at 4 images per identity"
```

**What each script produces:**
- `threshold_sweep.py` → `outputs/roc_curve.png`, `outputs/sweep_results.json`, and a logged entry in `outputs/runs.jsonl`
- `run_evaluation.py` → `outputs/confusion_matrix_val.png`, `outputs/confusion_matrix_test.png`, and two logged entries in `outputs/runs.jsonl`

All 5 runs are appended to `outputs/runs.jsonl`. Each entry records: run ID, timestamp, commit hash, split, threshold, all metrics, and the note passed via `--note`.

### Milestone 2 — Error Analysis

After `run_evaluation.py` completes, side-by-side composite images are saved for the first 10 false positive and false negative pairs from each split. These let you visually inspect which pairs the model got wrong and spot patterns in failures.

**Output structure:**
```
outputs/
  false_positives/{run_id}/
    composites/         ← side-by-side images (model said same, actually different)
    pairs_list.jsonl    ← full list of all FP pair indices
  false_negatives/{run_id}/
    composites/         ← side-by-side images (model said different, actually same)
    pairs_list.jsonl    ← full list of all FN pair indices
```

**To save more or fewer composite images**, pass `max_pairs` to `log_errors` in `scripts/run_evaluation.py`:
```python
# save first 25 pairs instead of the default 10
log_errors(..., error_type="FP", max_pairs=25)

# save all error pairs (can be 100s of images — use clean script after)
log_errors(..., error_type="FP", max_pairs=None)
```

**To delete the composite images after analysis** (keeps `pairs_list.jsonl` logs):
```bash
# preview what would be deleted
python scripts/clean_error_outputs.py --dry-run

# delete all composite images across all runs
python scripts/clean_error_outputs.py
```

**What to look for in the error pairs:**
- **False positives** (model said same person): look for similar lighting, pose, or background between two different people — these fool cosine similarity at the pixel level
- **False negatives** (model said different person): look for the same person with very different lighting, angle, expression, or age — large appearance variation pushes cosine similarity below the threshold

The `pairs_list.jsonl` keeps the full index list even after images are deleted, so you can always regenerate specific pairs by re-running evaluation.

### Milestone 2 — Artifacts

| Artifact | Location |
|---|---|
| Tracked run log (all 5 runs) | `outputs/runs.jsonl` |
| ROC curve (val sweep) | `outputs/roc_curve.png` |
| Confusion matrix — val split | `outputs/confusion_matrix_val.png` |
| Confusion matrix — test split | `outputs/confusion_matrix_test.png` |
| FP composite images | `outputs/false_positives/{run_id}/composites/` (gitignored) |
| FN composite images | `outputs/false_negatives/{run_id}/composites/` (gitignored) |
| FP/FN pair index logs | `outputs/false_positives/{run_id}/pairs_list.jsonl` (gitignored) |
| Evaluation report (PDF) | `reports/milestone2_report.pdf` |
| Threshold sweep raw data | `outputs/sweep_results.json` (gitignored — regenerated by running sweep) |

---

## Milestone 3 — Embedding-Based Inference System

Upgrades the verifier from pixel-vector similarity to **FaceNet (InceptionResnetV1, pretrained on VGGFace2) embeddings**, packages it in Docker, exposes a CLI inference interface, and measures throughput under concurrent use.

**Embedding model:** FaceNet / InceptionResnetV1 pretrained on VGGFace2. Each face image is resized to 160×160, normalized to [-1, 1], and passed through the network to produce a 512-dimensional L2-normalized embedding. Cosine similarity between the two embeddings is the verification score.

**Confidence rule:** Normalized distance from the decision boundary, in [0.0, 1.0].
- For a "same" decision (score ≥ threshold): `confidence = (score − threshold) / (1.0 − threshold)`
- For a "different" decision (score < threshold): `confidence = (threshold − score) / (threshold + 1.0)`
- 0.0 means the score is exactly at the boundary; 1.0 means maximum certainty.

**Threshold selection:** Same rule as Milestone 2 — maximize balanced accuracy on the val split. Re-run the sweep after switching to embeddings since the score distribution changes.

**Inference stages (kept separate for Milestone 4 profiling):**
1. Preprocessing — resize + normalize to (3, 160, 160) tensor
2. Embedding generation — InceptionResnetV1 forward pass → 512-D vector
3. Similarity scoring — cosine similarity between the two embeddings
4. Threshold decision — score ≥ threshold → "same"
5. Confidence computation — normalized boundary distance

---

### Milestone 3 — How to Run

#### Option A: Local environment

```
# Install dependencies (CPU-only torch)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Re-run threshold sweep with FaceNet embeddings (run 6):**
```
python scripts/lfw_ingestion_script.py
python scripts/threshold_sweep.py --run-id run6_sweep_val_facenet --note "Threshold sweep with FaceNet embeddings"
# Update selected_threshold in configs/eval.yaml with the printed value
```

**Evaluate val + test at locked threshold (runs 7–8):**
```
python scripts/run_evaluation.py --val-run-id run7_val_facenet --test-run-id run8_test_facenet --note "FaceNet embedding evaluation"
```

**Single-pair CLI inference:**
```
python scripts/infer.py --img1 path/to/face1.jpg --img2 path/to/face2.jpg
```

**Batch inference (CSV file with img1,img2 per row):**
```
python scripts/infer.py --batch path/to/pairs.csv
```

**Load test (LFW val pairs):**
```
python scripts/load_test.py --n-requests 50 --workers 4
```

**Load test (synthetic images — no dataset required):**
```
python scripts/load_test.py --n-requests 50 --workers 4 --synthetic
```

**Run all tests:**
```
pytest tests/
```

---

#### Option B: Docker

```
# Build the image (downloads FaceNet weights during build)
docker build -t face-verifier .

# Single-pair inference (mount a folder containing your images)
docker run -v /path/to/your/images:/images face-verifier \
  python scripts/infer.py --img1 /images/face1.jpg --img2 /images/face2.jpg

# Load test with synthetic images (no dataset needed)
docker run face-verifier \
  python scripts/load_test.py --n-requests 50 --workers 4 --synthetic

# Run tests
docker run face-verifier pytest tests/
```

---

### Milestone 3 — Artifacts

| Artifact | Location |
|---|---|
| Embedding module | `src/embeddings.py` |
| Inference pipeline | `src/inference.py` |
| CLI entrypoint | `scripts/infer.py` |
| Load test script | `scripts/load_test.py` |
| Dockerfile | `Dockerfile` |
| Inference unit + smoke tests | `tests/test_inference.py` |
| Embedding + inference config | `configs/eval.yaml` (embedding section) |
| Load test results | `outputs/load_test_results.json` |
| ROC curve (FaceNet sweep) | `outputs/roc_curve.png` |
| Confusion matrices | `outputs/confusion_matrix_val.png`, `outputs/confusion_matrix_test.png` |
| Run log (all runs) | `outputs/runs.jsonl` |
