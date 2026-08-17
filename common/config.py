"""
common/config.py
โหลด config.yaml ครั้งเดียว ใช้ร่วมกันทุกสคริปต์ + ตั้งค่า random seed ให้ reproducible
"""

import os
import random
from pathlib import Path

import numpy as np
import yaml

_CONFIG_CACHE = None


def get_project_root() -> Path:
    """หา root ของ project จากตำแหน่งไฟล์นี้ (common/config.py -> ../)"""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    """
    โหลด config.yaml (cache ไว้ ไม่ต้องอ่านไฟล์ซ้ำ)
    override ด้วย environment variable HK_CONFIG_PATH ได้ถ้าต้องการ
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = os.environ.get(
            "HK_CONFIG_PATH", str(get_project_root() / "config.yaml")
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _CONFIG_CACHE = cfg
    return cfg


def resolve_path(relative_path: str) -> Path:
    """แปลง path สัมพัทธ์ใน config.yaml ให้เป็น absolute path จาก project root"""
    return get_project_root() / relative_path


def set_seed(seed: int) -> None:
    """ตั้งค่า seed ให้ random / numpy สำหรับการแบ่ง dataset แบบ reproducible"""
    random.seed(seed)
    np.random.seed(seed)


def get_seed() -> int:
    cfg = load_config()
    return cfg.get("seed", 42)
