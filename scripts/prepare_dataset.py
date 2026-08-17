"""รัน: python scripts/prepare_dataset.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config, set_seed
from common.data import prepare_dataset

if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg["seed"])
    prepare_dataset(cfg)
