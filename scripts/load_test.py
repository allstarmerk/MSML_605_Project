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


