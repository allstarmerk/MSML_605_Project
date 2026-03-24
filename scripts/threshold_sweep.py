import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Sweeps thresholds on the val split, and will tell you what to set the
# "selected_threshold" that we will set in eval.yaml to

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.data_ingestion import load_config, load_lfw
from src.evaluation import (
    score_pairs_cosine,
    apply_threshold,
    compute_metrics,
    log_run,
)
from src.validation import (
    validate_pair_files,
    validate_pairs,
    validate_scores,
)


def load_pairs(pairs_dir, split_name):
    # load the pair indices and labels from the .npy files created in milestone 1
    pairs  = np.load(Path(pairs_dir) / f"{split_name}_pairs.npy")
    labels = np.load(Path(pairs_dir) / f"{split_name}_labels.npy")
    return pairs, labels


def run_sweep(scores, labels, thresholds):
    # hold scores fixed and vary only the threshold
    # this isolates the effect of the threshold itself
    results = []
    for t in thresholds:
        preds = apply_threshold(scores, t, higher_is_similar=True)
        m = compute_metrics(labels, preds)
        m["threshold"] = float(t)
        results.append(m)
    return results



