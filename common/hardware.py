"""
common/hardware.py
ตรวจสอบ GPU/CPU ที่มี และเลือก device ให้ ultralytics ใช้
"""

import platform
import subprocess


def check_gpu() -> dict:
    """คืนข้อมูล hardware ที่ตรวจเจอ: มี GPU ไหม, ชื่ออะไร, VRAM เท่าไหร่"""
    info = {
        "has_cuda": False,
        "device_name": None,
        "vram_gb": None,
        "os": platform.system(),
        "python_version": platform.python_version(),
    }

    try:
        import torch

        info["has_cuda"] = torch.cuda.is_available()
        if info["has_cuda"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
    except ImportError:
        pass

    return info


def resolve_device(requested: str = "auto") -> str:
    """
    แปลงค่า device จาก config ("auto") เป็นค่าที่ ultralytics เข้าใจ
    ถ้าขอ auto แต่ไม่มี GPU จะ fallback ไป cpu พร้อม print คำเตือน
    """
    if requested != "auto":
        return requested

    info = check_gpu()
    if info["has_cuda"]:
        return "0"

    print("⚠️  ไม่พบ GPU (CUDA) — จะรันบน CPU ซึ่งช้ากว่ามาก")
    return "cpu"


def print_hardware_summary() -> None:
    info = check_gpu()
    print("=== Hardware Summary ===")
    print(f"OS: {info['os']}")
    print(f"Python: {info['python_version']}")
    if info["has_cuda"]:
        print(f"GPU: {info['device_name']} ({info['vram_gb']} GB VRAM)")
    else:
        print("GPU: ไม่พบ (จะใช้ CPU)")
