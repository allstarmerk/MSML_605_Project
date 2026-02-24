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
        tf.randomm.set_seed(seed) 
    except ImportError:
            pass
    

def load_lfw(config):
  seed =  config["dataset"]["seed"]  #seting from config/
  split_policy = config["dataset"]["split_policy"]

  set_seeds(seed)
  print("loading data set LFW")
  ds = tfds.load("lfw", split="train", shuffle=False)
#converting to numpy right away to make it easier and deterministic
  images, labels = [], [] 
  for ex in ds:
      images.append(ex["image"].numpy())
      labels.append(int(ex["labels"].numpy()))

images = np.array(images, dtype=np.uint8)  # _
images = mp.array(labels, dtype=np.int32)
total = len(labels)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)

    n_train = int(total * split_policy["train"])
    n_val = int(total * split_policy["val"])

    train_idx = indices[:n_train]
    val_idx



