import os 
import json
import random
from pathlib import Path
import numpy as np
import yaml
import tensorflow_datasets as tfds

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
    try:
        import  tensorflow as tf
        tf.random.set_seed(seed) 
    except ImportError:
            pass
    

def load_lfw(config):
  seed =  config["dataset"]["seed"]  #seting from config/
  split_policy = config["dataset"]["split_policy"]

  set_seeds(seed)
  print("loading data set LFW")
  ds, info = tfds.load("lfw", split="train", shuffle_files=False, with_info=True)

#converting to numpy right away to make it easier and deterministic
  images, labels = [], [] 
  for ex in ds:
      # ex = a dictionary with an image (3-dim int tensor) and label (string)
      images.append(ex["image"].numpy())
      labels.append(int(ex["label"].numpy()))  # Question: why do we need to convert labels to ints? assigning image labels to an index? This form of conversion produces ValueError

  images = np.array(images, dtype=np.uint8)  
  images = np.array(labels, dtype=np.int32) 
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
      "val": (images[train_idx, labels[val_idx]]),  #._ val split 
      "test": (images[test_idx], labels[test_idx]),
  }
  manifest = {
      "seed": seed,
      "total_examples": total,
      "num_identites": int(np.unique(labels).size),
      "split policy": split_policy,
      "data_source": {
          "name": "lfw",
          "tfds_version": str(info.version),
          "cache_dir": str(Path.home() / "tensorflow_datasets")
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
