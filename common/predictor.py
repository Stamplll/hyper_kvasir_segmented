"""
common/predictor.py
ครอบการเรียก model.predict() ของ ultralytics ไว้ที่เดียว
(ย้าย logic หลักมาจาก src/predict.py เดิม เหลือ scripts/predict.py ไว้แค่ parse args)
"""

from pathlib import Path

from ultralytics import YOLO


def run_inference(
    source: str,
    weights: str,
    conf: float,
    imgsz: int,
    save: bool = False,
    show: bool = False,
    runs_dir: str = "runs",
):
    source_path = Path(source)
    if not source_path.exists() and not source.startswith(("rtsp://", "http://", "https://")):
        raise FileNotFoundError(f"Source not found: {source_path}")

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    results = model.predict(
        source=str(source_path) if source_path.exists() else source,
        conf=conf,
        imgsz=imgsz,
        save=save,
        show=show,
        project=f"{runs_dir}/predict",
        name="polyp_inference",
        exist_ok=True,
        verbose=True,
    )

    print(f"Done. Total predictions: {len(results)}")
    return results
