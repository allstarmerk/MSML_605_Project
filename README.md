# MSML_605_Project
MSML/MSAI project. It's a facial recognition system similar to faceId

- Group: Johnathan Sheikh, Renae Ricketts

- Main branch is for after bugs are fixed in working branch. Then we merge the working branch into the main branch. This allows for the main branch to be bug free and always in working condition. Then the end product will be the main branch. Working branch is essentialy the development branch and main being the production branch. 

- After you have work completed on the working branch you create a pull request. If u wanna use the UI of github you go to pull request. Click on create new pull request. Then you select the working branch and the main branch. Then you click on create pull request. Then you can add a description of the changes you made. Then you can click on create pull request again. Then the pull request will be created and you can see it in the pull request tab. Then you can merge the pull request if there are no conflicts. If there are conflicts then you have to resolve them before merging.

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
- This should download lfw data set(170MG) and saves/creates data/manifest.json & data/pairs/
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
  
- Then compare new data/manifest.json with the 1st run's manifest summary printed on the command terminal and check if they have identical counts. to comfirm determinism
<br/> 


5. To run the Similarity Benchmark test, run the following:
```
python scripts/benchmark_similarity.py
```
  
- This will provide a output showing the comparision of using the loop vs Numpy for both cosine and euclidean calculations. It also has a unit test at the end checking the math of the math functions.
<br/>

Design Choices:
  - We use a hard coded seed in config/ to meet the determinisc requirment. Seed is fixed numpy, tensorflow, and python random.

  - Saved pair indices are saved as a .npy so evaluation is always accurate and reproducable

  - Most of the settings needed are hard coded in /configs/dataset.yaml

---

## Milestone 2 — Evaluation Loop

> **Note:** Complete all Milestone 1 steps first (environment setup, LFW ingestion, and pair generation — steps 1-4 above) before running any Milestone 2 commands.

Builds a reproducible evaluation loop on top of the Milestone 1 pipeline. Includes threshold calibration on the validation split, experiment tracking, error analysis, and tests.

**Threshold selection rule:** Maximize balanced accuracy on the validation split. The selected threshold is stored in configs/eval.yaml.

**Data-centric improvement:** Capped overrepresented identities at 20 images each before generating pairs. Identity 1871 had 530 images and dominated the pair distribution. Controlled via `max_images_per_identity` in configs/dataset.yaml — set to 999 for baseline (no cap), 20 for the improved version.

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
```
# Set max_images_per_identity: 20 in configs/dataset.yaml first
python scripts/lfw_ingestion_script.py
python scripts/threshold_sweep.py --run-id run4_sweep_val_post_cap --note "Sweep after capping identities at 20 images"
# Update selected_threshold in configs/eval.yaml with the value printed above
python scripts/run_evaluation.py --val-run-id run5_val_post_cap --test-run-id run5_test_post_cap --note "Run 5 - Evaluation after capping identities at 20 images per identity"
```






