"""
common/timing.py
วัดเวลาที่ใช้ในแต่ละขั้นตอน (เตรียมข้อมูล, เทรน, ประเมินผล) แล้วบันทึกลงไฟล์
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timed_stage(stage_name: str, log_path: str | None = None):
    """
    ใช้แบบ:
        with timed_stage("training", "results/timing.json"):
            model.train(...)
    จะ print เวลาที่ใช้ และถ้าระบุ log_path จะบันทึกสะสมลงไฟล์ json ด้วย
    """
    start = time.perf_counter()
    print(f"⏳ เริ่ม stage: {stage_name}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"✅ {stage_name} เสร็จใน {elapsed:.1f} วินาที ({elapsed / 60:.2f} นาที)")

        if log_path:
            log_file = Path(log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if log_file.exists():
                data = json.loads(log_file.read_text())
            data[stage_name] = round(elapsed, 2)
            log_file.write_text(json.dumps(data, indent=2))
