import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Sweeps thresholds on the val split, and will tell you what to set the
# "selected_threshold" that we will set in eval.yaml to

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.data_ingestion import load_config, load_lfw
from src.evaluation import (
    score_pairs_cosine,
    apply_threshold,
    compute_metrics,
    log_run,
)
from src.validation import (
    validate_pair_files,
    validate_pairs,
    validate_scores,
)


def load_pairs(pairs_dir, split_name):
    # load the pair indices and labels from the .npy files created in milestone 1
    pairs  = np.load(Path(pairs_dir) / f"{split_name}_pairs.npy")
    labels = np.load(Path(pairs_dir) / f"{split_name}_labels.npy")
    return pairs, labels


def run_sweep(scores, labels, thresholds):
    # hold scores fixed and vary only the threshold
    # this isolates the effect of the threshold itself
    results = []
    for t in thresholds:
        preds = apply_threshold(scores, t, higher_is_similar=True)
        m = compute_metrics(labels, preds)
        m["threshold"] = float(t)
        results.append(m)
    return results


def save_roc_plot(results, out_dir):
    fprs = [r["false_positive_rate"] for r in results]
    tprs = [r["true_positive_rate"]  for r in results]

    # find the point with the best balanced accuracy to highlight on the plot
    best = max(results, key=lambda r: r["balanced_accuracy"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fprs, tprs, color="#2563eb", linewidth=2, label="Threshold sweep")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.scatter(
        [best["false_positive_rate"]], [best["true_positive_rate"]],
        color="red", zorder=5,
        label=f"Best threshold = {best['threshold']:.3f}\n"
              f"Bal. Accuracy  = {best['balanced_accuracy']:.4f}"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Validation Threshold Sweep")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path(out_dir) / "roc_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {out_path}")


def save_sweep_results(results, out_dir):
    # save every threshold and its metrics so we can inspect later
    out_path = Path(out_dir) / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Sweep results saved to {out_path}")


def main():
    config = load_config()

    eval_config_path = Path(root_path) / "configs" / "eval.yaml"
    with open(eval_config_path) as f:
        eval_cfg = yaml.safe_load(f)

    pairs_dir = config["paths"]["pairs_dir"]
    out_dir   = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # build the range of thresholds to sweep from eval.yaml settings
    thresholds = np.linspace(
        eval_cfg["threshold_sweep"]["min"],
        eval_cfg["threshold_sweep"]["max"],
        eval_cfg["threshold_sweep"]["steps"],
    )

    print("Loading LFW dataset...")
    splits, _ = load_lfw(config)
    images = splits["val"][0]

    # validate and load val pairs
    validate_pair_files(pairs_dir, "val")
    pairs, labels = load_pairs(pairs_dir, "val")
    validate_pairs(pairs, labels, "val")

    # score all pairs once — scores stay fixed for the entire sweep
    print("Scoring pairs...")
    scores = score_pairs_cosine(images, pairs)
    validate_scores(scores, pairs)

    # run the sweep across all threshold values
    print(f"Running threshold sweep ({len(thresholds)} steps)...")
    results = run_sweep(scores, labels, thresholds)

    # pick best threshold by maximizing balanced accuracy on val split
    best = max(results, key=lambda r: r["balanced_accuracy"])

    print(f"\n── Sweep Results ──────────────────────────────")
    print(f"  Best threshold:    {best['threshold']:.4f}")
    print(f"  Balanced Accuracy: {best['balanced_accuracy']:.4f}")
    print(f"  TPR:               {best['true_positive_rate']:.4f}")
    print(f"  FPR:               {best['false_positive_rate']:.4f}")
    print(f"  F1:                {best['f1_score']:.4f}")
    print(f"  TP={best['tp']}  FP={best['fp']}  TN={best['tn']}  FN={best['fn']}")

    # save artifacts
    save_roc_plot(results, out_dir)
    save_sweep_results(results, out_dir)

    # log this as Run 1
    log_run(
        run_id="run1_sweep_val_baseline",
        split="val",
        metric_name="cosine",
        threshold=best["threshold"],
        metrics=best,
        note="Run 1 - Baseline threshold sweep on val split. "
             "Best threshold selected by max balanced accuracy.",
    )

    print(f"\n>>> ACTION REQUIRED: Open configs/eval.yaml and set:")
    print(f"    selected_threshold: {best['threshold']:.4f}")
    print("\nThreshold sweep complete.")


if __name__ == "__main__":
    main()
