import json
import datetime
import subprocess
import numpy as np
from pathlib import Path
from matplotlib import image
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


# returns 2 arrays containing pairs of False Positives and False Negatives
def error_slices(pairs, labels, predictions):
    fp_slice = pairs[(predictions == 1) & (labels == 0)]
    fn_slice = pairs[(predictions == 0) & (labels == 1)]

    return fp_slice, fn_slice


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


def log_errors(run_id, split_name, error_images, error_pairs, errors_path,
               error_type="error", max_pairs=10):
    # save FP or FN pair images to disk and write the pair index list to pairs_list.jsonl
    # error_images: shape (N, 2, H, W, C) — both images for each error pair
    # error_pairs:  shape (N, 2)           — image indices for each error pair (for filenames)
    # errors_path:  directory to write into (e.g. outputs/false_positives/{run_id}/)
    # error_type:   label shown in composite title — "FP" or "FN"
    # max_pairs:    how many pairs to save images for (None = all)
    import matplotlib.pyplot as plt
    Path(errors_path).mkdir(parents=True, exist_ok=True)

    n_total = error_pairs.shape[0]
    n_save  = n_total if max_pairs is None else min(max_pairs, n_total)

    composites_path = errors_path + "composites/"
    Path(composites_path).mkdir(parents=True, exist_ok=True)

    for pair_id in range(n_save):
        img1, img2 = error_images[pair_id, 0], error_images[pair_id, 1]
        img1_id, img2_id = error_pairs[pair_id, 0], error_pairs[pair_id, 1]

        # side-by-side composite with labels so both images are visible at once
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(img1)
        axes[0].set_title(f"img {img1_id}")
        axes[0].axis("off")
        axes[1].imshow(img2)
        axes[1].set_title(f"img {img2_id}")
        axes[1].axis("off")
        fig.suptitle(f"{error_type} | pair {pair_id} | {split_name}", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{composites_path}pair{pair_id}.jpg", dpi=100)
        plt.close()

    # always write the full pair index list regardless of max_pairs
    record = {
        "run_id":          run_id,
        "split":           split_name,
        "total_errors":    n_total,
        "images_saved":    n_save,
        "mislabeled pairs": error_pairs.tolist(),
    }
    with open(errors_path + "pairs_list.jsonl", "a") as f:
        f.write(json.dumps(record) + '\n')

    print(f"  {error_type}: {n_total} pairs total, {n_save} composites saved to {composites_path}")
