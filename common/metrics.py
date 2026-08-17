"""
common/metrics.py
แปลงผลลัพธ์การ evaluate ของ ultralytics ให้เป็น dict/summary ที่อ่านง่าย
และบันทึกเป็น json — จุดเดียวที่ metrics ถูกดึงออกมา กัน evaluate.py
กับ generate_report เข้าใจตัวเลขไม่ตรงกัน
"""

import json
from pathlib import Path


def extract_seg_metrics(ultra_metrics) -> dict:
    """
    ดึงตัวเลขสำคัญจาก ultralytics SegmentMetrics object:
    box mAP (จากการ detect กรอบ) และ mask mAP (จากตัว segmentation เอง)
    """
    box = ultra_metrics.box
    seg = ultra_metrics.seg

    return {
        "box": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
        },
        "mask": {
            "precision": float(seg.mp),
            "recall": float(seg.mr),
            "mAP50": float(seg.map50),
            "mAP50-95": float(seg.map),
        },
    }


def save_metrics(metrics: dict, out_path: str) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"✅ บันทึก metrics ไว้ที่: {out_file}")


def print_metrics_summary(metrics: dict) -> None:
    print("=== Evaluation Summary (test set) ===")
    print(f"[Mask]  Precision: {metrics['mask']['precision']:.3f}  "
          f"Recall: {metrics['mask']['recall']:.3f}  "
          f"mAP50: {metrics['mask']['mAP50']:.3f}  "
          f"mAP50-95: {metrics['mask']['mAP50-95']:.3f}")
    print(f"[Box]   Precision: {metrics['box']['precision']:.3f}  "
          f"Recall: {metrics['box']['recall']:.3f}  "
          f"mAP50: {metrics['box']['mAP50']:.3f}  "
          f"mAP50-95: {metrics['box']['mAP50-95']:.3f}")
