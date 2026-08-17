"""รัน: python scripts/train.py [--config config.yaml] [--device 0|cpu|auto]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.hardware import print_hardware_summary
from common.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: project config.yaml)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device override, for example 0, 1, cpu, or auto",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print_hardware_summary()
    run_training(config_path=args.config, device=args.device)
