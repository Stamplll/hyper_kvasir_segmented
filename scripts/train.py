"""รัน: python scripts/train.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.hardware import print_hardware_summary
from common.train import run_training

if __name__ == "__main__":
    print_hardware_summary()
    run_training()
