import numpy as np
from pathlib import Path

#the purpose is to check inputs and outputs for common errors before steps in pipline. without we can get silent errors where it runs fine but produces wrong results no errors.
# can use assert statments for checks. If condition=False it  makes a error with discription of why  it failed
VALID_SPLITS = {"train", "val", "test"}

def validate_pairs(pairs, labels, split_name):  #checks if the pairs and labels are valid for the given split
   
    #check if split name is valid 
    assert split_name in VALID_SPLITS, \
    f"Invalid split name: '{split_name}'. Must be one of {VALID_SPLITS}."

    # check the types
    assert isinstance(pairs, np.ndarray), "Pairs should be a numpy array"
    assert isinstance(labels, np.ndarray), "Labels should be a numpy array"

    #check the shape
    assert pairs.ndim == 2 and pairs.shape[1] == 2, \
        f"Pairs should have shape (N, 2), but got {pairs.shape}"
    
    # check the len matches
    assert len(pairs) == len(labels), \
        f"Number of pairs ({len(pairs)}) does not match number of labels ({len(labels)})"
    
    #check labels are binary
    unique = set(np.unique(labels).tolist())
    assert unique.issubset({0, 1}), \
        f"Labels should be binary (0 or 1), but got unique values: {unique}"
                 
    #check not empty
    assert len(pairs) > 0, "Pairs array is empty"

    print(f"[validate_pairs] OK - {len(pairs)} pairs, split='{split_name}'")


def validate_scores(scores, pairs):
    #verify score count matches pair count
    assert len(scores) == len(pairs), \
        f"Score count ({len(scores)}) does not match pair count ({len(pairs)})"

    #check for non NAN values or infinite values from numpy math errors
    assert np.all(np.isfinite(scores)), \
        "Scores contain NaN or Inf values from a math error"

    print(f"[validate_scores] OK - {len(scores)} scores, "
          f"range [{scores.min():.4f}, {scores.max():.4f}]")


def validate_threshold(threshold):
    #check if threshold is a number
    assert isinstance(threshold, (int, float)), \
        f"Threshold must be a number, got {type(threshold)}"

    #check if threshold is in valid range for cosine
    assert np.isfinite(threshold), \
        f"Threshold must be a finite number, got {threshold}"

    print(f"[validate_threshold] OK - threshold={threshold}")


def validate_metrics(metrics):
    # check all expected metric keys are present in the returned dictionary
    required_keys = {
        "accuracy", "balanced_accuracy", "true_positive_rate",
        "false_positive_rate", "precision", "f1_score",
        "tp", "fp", "tn", "fn", "total"
    }
    missing = required_keys - set(metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"

    
    assert metrics["total"] > 0, "Total pairs is 0, cannot compute meaningful metrics"

    print(f"[validate_metrics] OK - "
          f"accuracy={metrics['accuracy']:.4f}, "
          f"bal_acc={metrics['balanced_accuracy']:.4f}")


def validate_pair_files(pairs_dir, split_name):
    #check if the pair files exist for the given split
    pairs_path  = Path(pairs_dir) / f"{split_name}_pairs.npy"
    labels_path = Path(pairs_dir) / f"{split_name}_labels.npy"

    assert pairs_path.exists(),  f"Missing pairs file:  {pairs_path}"
    assert labels_path.exists(), f"Missing labels file: {labels_path}"

    print(f"[validate_pair_files] OK - "
          f"found {pairs_path.name} and {labels_path.name}")
