import sys
import os
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


