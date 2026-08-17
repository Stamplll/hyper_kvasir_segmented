"""รัน: python scripts/evaluate.py --weights runs/segment/polyp_detection/weights/best.pt"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config, resolve_path
from common.evaluate import evaluate_on_test


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate polyp segmentation model on test set")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
    dataset_yaml = resolve_path(cfg["paths"]["processed_dir"]) / "dataset.yaml"
    evaluate_on_test(cfg, args.weights, str(dataset_yaml))
