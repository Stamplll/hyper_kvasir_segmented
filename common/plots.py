"""
common/plots.py
ultralytics สร้างกราฟ (results.png, confusion_matrix.png ฯลฯ) ให้อัตโนมัติอยู่แล้วใน
runs/segment/polyp_detection/ ไฟล์นี้เก็บฟังก์ชันเสริมสำหรับกราฟที่ ultralytics ไม่ได้ทำให้
เช่น เทียบ metrics ระหว่างหลายรอบการเทรน (ถ้ามีการรันซ้ำหลาย seed/config ในอนาคต)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric_comparison(run_metrics: dict, out_path: str) -> None:
    """
    run_metrics: {"run_name": {"mask": {...}, "box": {...}}, ...}
    วาดกราฟแท่งเทียบ mask mAP50-95 ระหว่างแต่ละรอบการรัน
    """
    names = list(run_metrics.keys())
    values = [m["mask"]["mAP50-95"] for m in run_metrics.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, values, color="#4C72B0")
    ax.set_ylabel("Mask mAP50-95")
    ax.set_title("เปรียบเทียบผลการเทรนแต่ละรอบ")
    ax.set_ylim(0, 1)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"✅ บันทึกกราฟไว้ที่: {out_file}")


def load_and_plot_from_json(metrics_json_paths: dict, out_path: str) -> None:
    """โหลด metrics.json หลายไฟล์ (จากหลาย run) แล้วเรียก plot_metric_comparison"""
    run_metrics = {}
    for name, path in metrics_json_paths.items():
        with open(path) as f:
            run_metrics[name] = json.load(f)
    plot_metric_comparison(run_metrics, out_path)
