"""
common/yolo_trainer.py
ครอบการเรียก ultralytics YOLO train() ไว้ที่เดียว เพื่อให้ scripts/train.py
เหลือแค่ thin entry point (เหมือนแนวทางของ repo classification ที่มี
common/yolo_trainer.py แยกจาก models/<model>/train.py)
"""

from pathlib import Path

from ultralytics import YOLO

from common.config import resolve_path
from common.hardware import resolve_device


def build_model(weights: str) -> YOLO:
    """โหลด YOLO segmentation model จาก pretrained weights (เช่น yolo11n-seg.pt)"""
    return YOLO(weights)


def train_yolo_seg(cfg: dict, dataset_yaml_path: str):
    """
    เทรน YOLO segmentation model ตามค่าใน config.yaml
    คืนค่า results object ของ ultralytics (มี metrics, save_dir ฯลฯ)
    """
    if not Path(dataset_yaml_path).exists():
        raise FileNotFoundError(
            f"❌ ไม่พบไฟล์ {dataset_yaml_path} กรุณารัน scripts/prepare_dataset.py ก่อน"
        )

    model = build_model(cfg["model"]["weights"])
    device = resolve_device(cfg["training"]["device"])

    # ใช้ absolute path เสมอ — ถ้าใช้ relative path ("runs/segment") ultralytics จะเอาไป
    # join กับ settings.runs_dir ของมันเองอีกชั้น (เจอบั๊กจริงตอนทดสอบ: ได้ path ซ้อนกัน
    # เป็น "runs/segment/runs/segment/polyp_detection")
    project_dir = resolve_path(cfg["paths"]["runs_dir"]) / "segment"

    print("⏳ กำลังเทรนโมเดล โปรดรอสักครู่...")
    results = model.train(
        data=dataset_yaml_path,
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["model"]["img_size"],
        batch=cfg["training"]["batch_size"],
        patience=cfg["training"]["patience"],
        device=device,
        project=str(project_dir),
        name="polyp_detection",
        exist_ok=True,
        seed=cfg["seed"],
    )

    print("✅ การเทรนเสร็จสมบูรณ์!")
    print(f"ผลลัพธ์และกราฟต่างๆ ถูกบันทึกไว้ที่: {results.save_dir}")
    return results


def get_best_weights_path(results) -> Path:
    """หา path ของ best.pt จากผลลัพธ์การเทรน"""
    return Path(results.save_dir) / "weights" / "best.pt"
