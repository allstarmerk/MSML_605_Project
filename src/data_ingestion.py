import os 
import json
import random
from pathlib import Path
import numpy as np
import yaml

def load_config(config_path=None):
    if config_path is None:  
        project_root = Path (__file__).resolve().parent.parent
        config_path = project_root / "configs" / "dataset.yaml" 
    with open(config_path) as f:
        return yaml.safe_load(f)
    

def set_seeds(seed): #fixes "randomness" with pre set seed value in config/
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    

def load_lfw(config):
  seed =  config["dataset"]["seed"]  
  split_policy = config["dataset"]["split_policy"]

  set_seeds(seed)
  from sklearn.datasets import fetch_lfw_people
  print("loading dataset LFW via sklearn...")
  lfw = fetch_lfw_people(min_faces_per_person=1, color=True, resize=0.5)

  images = (lfw.images * 255).astype(np.uint8)
  labels = lfw.target.astype(np.int32)
  order = np.argsort(labels, kind="stable") # sort by label for deterministic ordering before spliting
  images = images[order]       # kind="stable" keeps relative order of images within same identity
  labels = labels[order]
  total = len(labels)

  rng = np.random.default_rng(seed)
  indices = rng.permutation(total)

  n_train = int(total * split_policy["train"])
  n_val = int(total * split_policy["val"])

  train_idx = indices[:n_train]
  val_idx =  indices[n_train : n_train + n_val]
  test_idx = indices[n_train + n_val :]

  splits = {
      "train": (images[train_idx], labels[train_idx]),
      "val": (images[val_idx], labels[val_idx]),  
      "test": (images[test_idx], labels[test_idx]),
  }
  manifest = {
      "seed": seed,
      "total_examples": total,
      "num_identites": int(np.unique(labels).size),
      "split policy": split_policy,
      "data_source": {
          "name": "lfw",
          "loader": "sklearn.datasets.fetch_lfw_people",
          "cache_dir": str(Path.home() / "scikit_learn_data"),
      },
      "split_counts": {
          "train": len(train_idx),
          "val": len(val_idx),
          "test": len(test_idx),
      },
    

  }
  return splits,  manifest

def save_mainifest(manifest, path):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
      json.dump(manifest, f, indent=2)
  print(f"Manifest saved to : {path}")

