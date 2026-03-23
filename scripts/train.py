#module should set up the tracking and logging of runs (put this functionality in a utilities module eventually?)
#module should train and evaluate a face-verification model when executed and track training runs

import numpy as np
import sys
import random
import os
from typing import Dict, List

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.similarity import cosine_similarity

#loading Training pairs and corresponding labels
pairs  = np.load('data/pairs/train_pairs.npy')
labels = np.load('data/pairs/train_labels.npy')

# generate ID or run for logging
def generate_run_id() -> str:
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{random.randint(0, 9999):04d}"
    return f"{ts}_{suffix}"

# write evaluation metrics
def eval_metrics(model: Dict, pair_set, labels_set) -> Dict[str, float]:
    # pair_set: a np.array containing ALL image pairs
    # labels_set: a np.array containing ALL labels; label indices correspond with pair indices

    tp = fp = tn = fn = 0     #True/False Negatives and Positives
    thr = float(model["threshold"])
    
    img_a = pair_set[:,0]   #set of 1st imgs in pair
    img_b = pair_set[:,1]   #set of 2nd imgs in pair

    sim_scores = cosine_similarity(img_a,img_b)
    
    yhats = 1 if sim_scores >= thr else 0

    if yhats == 1 and labels_set == 1:
        tp += 1
    elif yhats == 1 and labels_set == 0:
        fp += 1
    elif yhats == 0 and yhats == 0:
        tn += 1
    else:
        fn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "n_valid": total}

