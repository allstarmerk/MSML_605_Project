import sys
import os
import argparse
import csv
import yaml
from pathlib import Path

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.embeddings import load_model
from src.inference import run_inference


def print_result(img1, img2, result):
    print(f"\nPair:       {img1}  vs  {img2}")
    print(f"  Score:      {result['score']:.6f}")
    print(f"  Threshold:  {result['threshold']:.4f}")
    print(f"  Decision:   {result['decision'].upper()}")
    print(f"  Confidence: {result['confidence']:.6f}")
    print(f"  Latency:    {result['latency_ms']['total']:.1f} ms  "
          f"(preprocess={result['latency_ms']['preprocess']:.1f} ms, "
          f"embed={result['latency_ms']['embed']:.1f} ms, "
          f"score={result['latency_ms']['score']:.1f} ms)")


def main():
    parser = argparse.ArgumentParser(
        description="Face verification CLI given two images, returns same/different decision."
    )
    parser.add_argument("--img1",      type=str, help="Path to first image")
    parser.add_argument("--img2",      type=str, help="Path to second image")
    parser.add_argument("--batch",     type=str, help="CSV file with one img1,img2 path pair per row")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the operating threshold (default: value in configs/eval.yaml)")
    args = parser.parse_args()

    if not args.img1 and not args.batch:
        parser.error("Provide --img1 and --img2 for a single pair, or --batch for a batch file.")
    if args.img1 and not args.img2:
        parser.error("--img1 requires --img2.")

    eval_cfg_path = Path(root_path) / "configs" / "eval.yaml"
    with open(eval_cfg_path) as f:
        eval_cfg = yaml.safe_load(f)
    threshold = args.threshold if args.threshold is not None else eval_cfg["selected_threshold"]

    print("Loading FaceNet model (vggface2 pretrained)...")
    model = load_model()
    print(f"Threshold:  {threshold:.4f}\n")

    if args.batch:
        with open(args.batch, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                img1, img2 = row[0].strip(), row[1].strip()
                result = run_inference(img1, img2, model, threshold)
                print_result(img1, img2, result)
    else:
        result = run_inference(args.img1, args.img2, model, threshold)
        print_result(args.img1, args.img2, result)


if __name__ == "__main__":
    main()
