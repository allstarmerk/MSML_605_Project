import sys
import os
import numpy as np
from pathlib import Path

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


