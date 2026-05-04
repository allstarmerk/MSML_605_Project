# System Card — Face Verification System

**MSML/MSAI 605 | Milestone 4**
**Authors:** Johnathan Sheikh, Renae Ricketts
**Final Release Tag:** v1.0-final
**Operating Threshold:** 0.3970

---

## Section 1 — System Overview & Pipeline Summary

This system is a face verification system built on a pretrained deep face embedding model. Given two images, it
determines whether both images show the same person and returns a binary decision (same / different), a similarity
score, and a calibrated confidence value.

**Pipeline (5 stages):**

1. **Preprocessing** — Each input image is loaded as RGB, resized to 160×160 pixels, and normalized to the range [−1, 1]. The result is a (3, 160, 160) floating-point tensor.
2. **Embedding generation** — Each tensor is passed through FaceNet (InceptionResnetV1, pretrained on VGGFace2). This produces a 512-dimensional L2-normalized embedding vector per image.
3. **Similarity scoring** — Cosine similarity is computed between the two 512-D embedding vectors. Scores range from −1 (completely dissimilar) to 1 (identical).
4. **Threshold decision** — If the cosine similarity score is ≥ 0.3970, the system outputs "same person." Otherwise it outputs "different person."
5. **Confidence computation** — A confidence value in [0.0, 1.0] is computed as the normalized distance from the decision boundary. A confidence of 0.0 means the score landed exactly at the threshold (maximum uncertainty). A confidence of 1.0 means maximum certainty in either direction.

---

## Section 2 — Intended Use & Explicitly Out-of-Scope Uses

**Intended use:**
This system is intended for academic and research use in face verification — specifically, answering the question
"do these two images show the same person?" on pairs of pre-cropped, roughly frontal face photographs. It was
developed and evaluated on the Labeled Faces in the Wild (LFW) dataset and is appropriate for use in similar
controlled research settings.

**Explicitly out-of-scope uses:**
- Processing non-frontal, heavily occluded, or very low resolution face images
- Any medical, legal, or high-stakes decision-making where errors have serious consequences
- Any deployment context where a false positive (two different people flagged as the same person) or false negative carries meaningful risk to individuals

---

## Section 3 — Data Summary & Important Data Limitations

**Evaluation dataset — LFW (Labeled Faces in the Wild):**
The system was evaluated on LFW, a publicly available benchmark dataset of approximately 13,233 face images
spanning 5,749 identities. Images were collected by scraping web photos of public figures and celebrities. The
dataset was split 70/15/15 into train, validation, and test sets using a fixed seed (42) for reproducibility. Pair
generation produced 3,000 training pairs, 1,000 validation pairs, and 1,000 test pairs, each with a 50/50
positive-to-negative ratio.

**Pretraining data — VGGFace2:**
The FaceNet embedding model (InceptionResnetV1) used in this system was not trained from scratch. It was
pretrained on VGGFace2, a large-scale face recognition dataset containing approximately 3.31 million images of
around 9,131 identities, also sourced from web-scraped photographs of public figures.

**Important data limitations:**
Both LFW and VGGFace2 are heavily skewed toward adult faces of people who appear frequently in Western media. Male
subjects and subjects of European descent are over-represented in both datasets. Children, elderly individuals,
non-Western faces, and darker skin tones are underrepresented or absent. No demographic metadata is available for
LFW, so per-group performance has not been measured. Any deployment on populations that differ meaningfully from
these datasets may result in degraded and unmeasured performance.

---

## Section 4 — Operating Threshold & Key Metrics

The operating threshold of **0.3970** was selected by running a threshold sweep on the validation split and choosing
the value that maximized balanced accuracy. The threshold was selected on the validation split only and then
applied to the test split without further tuning.

**Key metrics at threshold = 0.3970 (test split, Run 11):**

| Metric | Value |
|---|---|
| Balanced Accuracy | 0.9790 |
| True Positive Rate (TPR) | 0.9740 |
| False Positive Rate (FPR) | 0.0160 |
| F1 Score | 0.9789 |
| True Positives | 487 |
| False Positives | 8 |
| True Negatives | 492 |
| False Negatives | 13 |

A false positive rate of 1.6% means approximately 1 in 62 pairs of different people will be incorrectly classified
as the same person at this threshold.

---

## Section 5 — Observed Failure Modes & Limitations

From observing incorrectly marked Negatives (False Negatives) and incorrectly marked Positives (False Positives),
it has been observed that the model seems to have trouble recognizing two images as the same person when the images
have significantly different lighting. Also observed: if two different people have the same or similar skin tones or
the same or similar creases in the face in two different images, then the model will mistakenly label the images as
of the same person. The model also has difficulty recognizing the same person in two different images when one image
displays the subject wearing different headwear or different makeup than the other image, or when one image has an
obstruction blocking the subject's face and the other one doesn't.

These failures could possibly stem from imperfections during the image preprocessing since the resolution of the
images lowers and the cropping of the images such that the subject's hairline is cropped out.

Visual examples of failure cases are shown in the Appendix below.

---

## Section 6 — Fairness-Related Risks & Misuse Concerns

**Fairness risks:**
Because both LFW and VGGFace2 over-represent Western adult faces — particularly male subjects — the system's
performance on underrepresented groups such as darker-skinned individuals, non-Western faces, children, and
elderly people has not been measured and may be lower. No demographic metadata was available to perform subgroup
analysis, so no per-group accuracy claims are made here. Users should not assume the reported metrics apply
equally across all demographic groups. This system should not be deployed in any context where equal treatment
across demographic groups is required.

Additionally, the observed failure patterns — the model struggles when two images of the same person differ
significantly in lighting, expression, or accessories — may affect some populations more than others. For example,
images collected in lower-light environments or with less standardized photography conditions could
systematically increase error rates for certain groups.

**Misuse concerns:**
This system is capable of matching two images of the same person with 97.4% recall at the operating threshold.
This capability creates real misuse risks if the system is deployed outside its intended academic context:
- It could be used to track or de-anonymize individuals from photographs without their consent.
- It could be integrated into surveillance pipelines to identify people in public spaces.
- At a 1.6% false positive rate, roughly 1 in 62 comparisons of two different people will be incorrectly matched, which is unacceptable in any high-stakes identification setting.

This system was built for research and should not be used for any identification, surveillance, or access-control
purpose.

---

## Section 7 — Operational Constraints

**Input format:**
- Accepts any RGB image file (JPEG, PNG)
- Any resolution is accepted — images are internally resized to 160×160 pixels
- Images are assumed to already be face crops. There is no built-in face detector. If an input image is not a cropped face region, the output will be meaningless without error
- Faces should be roughly frontal. Extreme profile angles, heavy occlusions, or non-face inputs are not handled and will produce unreliable scores

**Latency (CPU, measured on LFW validation pairs with 4 concurrent workers on johnathans system):**
- Mean latency: 309.7 ms per pair
- Median (p50): 257.8 ms per pair
- p95: 1,022 ms per pair
- p99: 1,033 ms per pair
- Throughput: approximately 12.7 pairs/second with 4 workers

**Hardware requirements:**
- Tested CPU-only on Python 3.11 with PyTorch 2.2.2
- Requires approximately 1–2 GB of RAM to load the FaceNet model
- GPU is not required but would significantly reduce embedding latency
- On first run without Docker, requires internet access to download FaceNet model weights (~89 MB). The Docker image bakes weights in at build time and runs fully offline

**Software requirements:**
- Python 3.11, facenet-pytorch, torch 2.2.2, torchvision 0.17.2 (see `requirements.txt`)
- Runs in Docker using `python:3.11-slim` base image (see `Dockerfile`)

---

## Section 8 — Reproducibility

Full instructions to reproduce all results from a clean clone are in `README.md`. The threshold (0.3970) is stored
in `configs/eval.yaml` under `selected_threshold`. The dataset split and pair generation seed (42) are stored in
`configs/dataset.yaml`. All evaluation runs are logged to `outputs/runs.jsonl`.

The final release is tagged as **v1.0-final** in the Git repository. Checking out this tag and following the README
will reproduce the evaluation results, CLI behavior, and load test results documented in this System Card.

**Key artifact locations:**
- System Card: `reports/system_card.pdf`
- Profiling report: `reports/profiling_report.pdf`
- Reproducibility checklist: `reports/reproducibility_checklist.md`
- Run log: `outputs/runs.jsonl`
- Confusion matrices: `outputs/confusion_matrix_val.png`, `outputs/confusion_matrix_test.png`
- Load test results: `outputs/load_test_results.json`

---

## Appendix — Failure Case Examples (Test Split)

All examples are from the test split at threshold 0.3970. Each image shows a side-by-side pair: left image and right image as fed to the model.

### False Positives — Different people classified as the same person (8 total)

These pairs scored ≥ 0.3970 but are actually two different people. Common patterns: similar skin tone, similar facial structure, or similar lighting/photography conditions between the two subjects.

![FP pair 0](../outputs/false_positives/run_test_final/composites/pair0.jpg)
![FP pair 1](../outputs/false_positives/run_test_final/composites/pair1.jpg)
![FP pair 2](../outputs/false_positives/run_test_final/composites/pair2.jpg)
![FP pair 3](../outputs/false_positives/run_test_final/composites/pair3.jpg)
![FP pair 4](../outputs/false_positives/run_test_final/composites/pair4.jpg)
![FP pair 5](../outputs/false_positives/run_test_final/composites/pair5.jpg)
![FP pair 6](../outputs/false_positives/run_test_final/composites/pair6.jpg)
![FP pair 7](../outputs/false_positives/run_test_final/composites/pair7.jpg)

### False Negatives — Same person classified as different people (13 total, 10 shown)

These pairs scored < 0.3970 but are actually the same person. Common patterns: significantly different lighting, headwear or accessories, facial obstructions, or large changes in expression or age between the two images.

![FN pair 0](../outputs/false_negatives/run_test_final/composites/pair0.jpg)
![FN pair 1](../outputs/false_negatives/run_test_final/composites/pair1.jpg)
![FN pair 2](../outputs/false_negatives/run_test_final/composites/pair2.jpg)
![FN pair 3](../outputs/false_negatives/run_test_final/composites/pair3.jpg)
![FN pair 4](../outputs/false_negatives/run_test_final/composites/pair4.jpg)
![FN pair 5](../outputs/false_negatives/run_test_final/composites/pair5.jpg)
![FN pair 6](../outputs/false_negatives/run_test_final/composites/pair6.jpg)
![FN pair 7](../outputs/false_negatives/run_test_final/composites/pair7.jpg)
![FN pair 8](../outputs/false_negatives/run_test_final/composites/pair8.jpg)
![FN pair 9](../outputs/false_negatives/run_test_final/composites/pair9.jpg)
