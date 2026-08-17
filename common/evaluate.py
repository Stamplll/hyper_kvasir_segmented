"""
common/evaluate.py
ประเมินผลบน test set แยกจาก train.py ชัดเจน
(ของเดิม repo นี้ยังไม่มีขั้นตอน evaluate แยก เพิ่มเข้ามาให้ครบ pipeline)
"""

from pathlib import Path

from ultralytics import YOLO

from common.metrics import extract_seg_metrics, print_metrics_summary, save_metrics


def evaluate_on_test(cfg: dict, weights_path: str, dataset_yaml_path: str) -> dict:
    """รันโมเดลบน test split แล้วคืน metrics dict พร้อมบันทึกลงไฟล์"""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ weights: {weights_path}")

    model = YOLO(str(weights_path))

    print("⏳ กำลังประเมินผลบน test set...")
    results = model.val(
        data=dataset_yaml_path,
        split="test",
        conf=cfg["evaluation"]["conf_threshold"],
        iou=cfg["evaluation"]["iou_threshold"],
    )

    metrics = extract_seg_metrics(results)
    print_metrics_summary(metrics)

    out_path = Path(cfg["paths"]["runs_dir"]) / "evaluation" / "metrics.json"
    save_metrics(metrics, str(out_path))

    return metrics
