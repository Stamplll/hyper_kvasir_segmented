"""
common/train.py
จุดรวมสำหรับเทรน — โหลด config, เตรียม dataset ถ้ายังไม่มี, แล้วเรียก yolo_trainer
เหมือนแนวคิด common/train.py ของ repo classification (thin, เรียก trainer จริงที่แยกไว้)
"""

from pathlib import Path

from common.config import load_config, resolve_path, set_seed
from common.data import prepare_dataset
from common.timing import timed_stage
from common.yolo_trainer import get_best_weights_path, train_yolo_seg


def run_training(config_path: str | None = None, device: str | None = None):
    cfg = load_config(config_path)
    if device is not None:
        cfg["training"]["device"] = device
    set_seed(cfg["seed"])

    dataset_yaml = resolve_path(cfg["paths"]["processed_dir"]) / "dataset.yaml"
    if not dataset_yaml.exists():
        print("ℹ️  ยังไม่มี dataset ที่เตรียมไว้ กำลังรัน prepare_dataset ก่อน...")
        with timed_stage("prepare_dataset"):
            dataset_yaml = prepare_dataset(cfg)

    with timed_stage("training"):
        results = train_yolo_seg(cfg, str(dataset_yaml))

    best_weights = get_best_weights_path(results)
    print(f"📦 best weights อยู่ที่: {best_weights}")
    print(
        "   คัดลอกไปไว้ที่ weights/ ด้วยตัวเองถ้าพอใจกับผลลัพธ์แล้ว "
        "(เช่น cp <best_weights> weights/polyp_yolo11n_seg_best.pt)"
    )
    return results


if __name__ == "__main__":
    run_training()
