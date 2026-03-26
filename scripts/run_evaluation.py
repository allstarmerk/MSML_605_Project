import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Loads the selected threshold from eval.yaml and evaluates on both the val
# and test splits. Saves confusion matrices and logs runs 2 and 3.
# we will run this after threshold_sweep.py and after updating selected_threshold in eval.yaml

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.data_ingestion import load_config, load_lfw
from src.evaluation import (
    score_pairs_cosine,
    apply_threshold,
    compute_metrics,
    log_run,
    error_slices,
    log_errors,
)
from src.validation import (
    validate_pair_files,
    validate_pairs,
    validate_scores,
    validate_threshold,
    validate_metrics,
)


def load_pairs(pairs_dir, split_name):
    # load the pair indices and labels from the .npy files created in milestone 1
    pairs  = np.load(Path(pairs_dir) / f"{split_name}_pairs.npy")
    labels = np.load(Path(pairs_dir) / f"{split_name}_labels.npy")
    return pairs, labels


def save_confusion_matrix(metrics, split_name, threshold, out_dir):
    # build the 2x2 confusion matrix array
    # rows = actual, columns = predicted
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Different", "Pred: Same"])
    ax.set_yticklabels(["Actual: Different", "Actual: Same"])

    # write the count number inside each cell
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    ax.set_title(
        f"Confusion Matrix — {split_name} split\n"
        f"Threshold = {threshold:.4f}"
    )
    plt.tight_layout()

    out_path = Path(out_dir) / f"confusion_matrix_{split_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")


def evaluate_split(split_name, images, pairs, labels,
                   threshold, out_dir, run_id, note):
    print(f"\n── Evaluating {split_name} split ──────────────────")

    # validate all inputs before doing any computation
    validate_pairs(pairs, labels, split_name)
    validate_threshold(threshold)

    # score every pair using cosine similarity
    scores = score_pairs_cosine(images, pairs)
    validate_scores(scores, pairs)

    # convert scores to binary same/different decisions using the threshold
    predictions = apply_threshold(scores, threshold, higher_is_similar=True)

    # extract pair IDs and corresponding images of FPs and FNs for error analysis
    fp_slice, fn_slice = error_slices(pairs, labels, predictions)
    fp_images, fn_images = images[fp_slice], images[fn_slice]

    # compute all metrics and validate the output
    metrics = compute_metrics(labels, predictions)
    validate_metrics(metrics)


    save_confusion_matrix(metrics, split_name, threshold, out_dir)


    log_run(
        run_id=run_id,
        split=split_name,
        metric_name="cosine",
        threshold=threshold,
        metrics=metrics,
        note=note,
    )

    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  TPR:               {metrics['true_positive_rate']:.4f}")
    print(f"  FPR:               {metrics['false_positive_rate']:.4f}")
    print(f"  F1:                {metrics['f1_score']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")

    log_errors(run_id, split_name, fp_images, fp_slice, errors_path=f"outputs/false_positives/{run_id}/", error_type="FP")
    log_errors(run_id, split_name, fn_images, fn_slice, errors_path=f"outputs/false_negatives/{run_id}/", error_type="FN")

    return metrics


def main():
    # pass run IDs via --val-run-id and --test-run-id arguments, for example:
    # Baseline:  python scripts/run_evaluation.py --val-run-id run2_val_selected_threshold --test-run-id run3_test_final_baseline
    # Post-cap:  python scripts/run_evaluation.py --val-run-id run5_val_post_cap --test-run-id run5_test_post_cap
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-run-id",  type=str, default="eval_val",
                        help="Run ID for the val split evaluation")
    parser.add_argument("--test-run-id", type=str, default="eval_test",
                        help="Run ID for the test split evaluation")
    parser.add_argument("--note", type=str, default="Evaluation at selected threshold.",
                        help="Short note describing this run")
    args = parser.parse_args()

    config = load_config()

    eval_config_path = Path(root_path) / "configs" / "eval.yaml"
    with open(eval_config_path) as f:
        eval_cfg = yaml.safe_load(f)

    pairs_dir = config["paths"]["pairs_dir"]
    threshold = eval_cfg["selected_threshold"]
    out_dir   = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    print(f"Selected threshold from eval.yaml: {threshold}")
    print("Loading LFW dataset...")
    splits, _ = load_lfw(config)

    # evaluate val split at selected threshold and record confusion matrix
    validate_pair_files(pairs_dir, "val")
    val_pairs, val_labels = load_pairs(pairs_dir, "val")
    evaluate_split(
        split_name="val",
        images=splits["val"][0],
        pairs=val_pairs,
        labels=val_labels,
        threshold=threshold,
        out_dir=out_dir,
        run_id=args.val_run_id,
        note=args.note + f" Val split. Threshold={threshold:.4f}.",
    )

    # evaluate test split at the locked threshold
    # threshold is NOT tuned on test — it is locked from the val sweep
    validate_pair_files(pairs_dir, "test")
    test_pairs, test_labels = load_pairs(pairs_dir, "test")
    evaluate_split(
        split_name="test",
        images=splits["test"][0],
        pairs=test_pairs,
        labels=test_labels,
        threshold=threshold,
        out_dir=out_dir,
        run_id=args.test_run_id,
        note=args.note + f" Test split. Threshold was NOT tuned on test.",
    )

    print("\nEvaluation complete.")
    print(f"Runs logged to outputs/runs.jsonl")
    print("Confusion matrices saved to outputs/")


if __name__ == "__main__":
    main()
