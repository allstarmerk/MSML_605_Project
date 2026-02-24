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
    

def 