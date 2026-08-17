"""รัน: python scripts/predict.py --source path/to/image_or_video --weights weights/polyp_yolo11n_seg_best.pt --save"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config
from common.predictor import run_inference


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Run YOLO11 segmentation inference for polyp detection."
    )
    parser.add_argument("--source", type=str, required=True,
                         help="Path to an image, video, folder, or stream source.")
    parser.add_argument("--weights", type=str,
                         default=f"{cfg['paths']['weights_dir']}/polyp_yolo11n_seg_best.pt",
                         help="Path to trained model weights (.pt).")
    parser.add_argument("--conf", type=float, default=cfg["inference"]["conf_threshold"],
                         help="Confidence threshold for predictions.")
    parser.add_argument("--imgsz", type=int, default=cfg["inference"]["img_size"],
                         help="Inference image size.")
    parser.add_argument("--save", action="store_true",
                         help="Save prediction visualizations to runs/predict.")
    parser.add_argument("--show", action="store_true",
                         help="Display prediction window during inference.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
    run_inference(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show,
        runs_dir=cfg["paths"]["runs_dir"],
    )
