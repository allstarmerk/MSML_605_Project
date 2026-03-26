import json
import datetime
import subprocess
import numpy as np
from pathlib import Path
from src.similarity import cosine_similarity, euclidean_distance, flatten_embedding


def score_pairs_cosine(images, pairs):
    #compute the cosine similarity score for every pair
    #higher = more similar, lower = less similar
    a = flatten_embedding(images[pairs[:, 0]])
    b = flatten_embedding(images[pairs[:, 1]])
    return cosine_similarity(a, b)  # returns (N,) array in range [-1, 1]


def score_pairs_euclidean(images, pairs):
    # compute euclidean distance for every pair
    # lower = more similar, higher = less similar
    a = flatten_embedding(images[pairs[:, 0]])
    b = flatten_embedding(images[pairs[:, 1]])
    return euclidean_distance(a, b)  # returns (N,) array in range [0, inf)


def apply_threshold(scores, threshold, higher_is_similar=True):  #to use euclidean distance, set higher_is_similar to False since lower scores mean more similar. Also must change values in eval.yaml for min/max range
    #convert continuous scores to binary decisions
    # higher_is_similar=True for cosine (score >= threshold means same person)
    # higher_is_similar=False for euclidean (score <= threshold means same person)
    if higher_is_similar:
        return (scores >= threshold).astype(np.int32)
    else:
        return (scores <= threshold).astype(np.int32)


def compute_metrics(labels, predictions):
    #return accuracy, and true positive rate (same and correct match), false positive rate(predict same person but different), true negative rate ("diff person prediction" and it is different people),
    # false negative (predicted different people but actually same person)
    #TPR (true positive rate) = How good it is at catching matches.
    #FPR (false positive rate) = How often it incorrectly matches different people.
    #accuracy = how many pairs it got right overall.
    #TNR (true negative rate) = How good it is at correctly identifying different people
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    total    = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    #High TPR = good at catching real matches
    #Low FPR = good at rejecting impostors
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    #precision = out of all the pairs it predicted as matches, how many were actually matches.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    #the balance between true positive rate and precision
    f1_score = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    #average of true positive rate and true negative rate (which is 1 - false positive rate)
    #used to pick threshold because its balanced between both despite the dataset being imbalanced
    balanced_accuracy = (tpr + (1 - fpr)) / 2

    return {
        "accuracy":          round(accuracy,          4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "true_positive_rate": round(tpr,              4),
        "false_positive_rate": round(fpr,             4),
        "precision":         round(precision,         4),
        "f1_score":          round(f1_score,          4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
    }


def log_run(run_id, split, metric_name, threshold, metrics, note,
            runs_path="outputs/runs.jsonl"):
    # append one tracked run record to the runs log file
    # each run records: id, timestamp, commit hash, split, threshold, metrics, and a note
    Path(runs_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"

    record = {
        "run_id":      run_id,
        "timestamp":   datetime.datetime.now().isoformat(),
        "commit":      commit,
        "split":       split,
        "metric_name": metric_name,
        "threshold":   round(float(threshold), 4),
        "metrics":     metrics,
        "note":        note,
    }

    with open(runs_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"Run '{run_id}' logged to {runs_path}")
    return record


def log_errors(run_id, split, error_slice, errors_path="outputs/false_positives.jsonl"):
    Path(errors_path).parent.mkdir(parents=True, exist_ok=True)
    """try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    """
    with open(errors_path, "a") as f:
        f.write(json.dumps(error_slice.tolist()))

    print(f"Errors from '{run_id} logged to {errors_path}")