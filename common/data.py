"""
common/data.py
ทุกอย่างที่เกี่ยวกับการเตรียมข้อมูล:
  - แตกไฟล์ zip ต้นฉบับ
  - หาโฟลเดอร์ images/masks อัตโนมัติ
  - แปลง binary mask -> YOLO polygon label
  - แบ่ง train/val/test
  - สร้าง dataset.yaml ให้ ultralytics ใช้

เดิมทุกอย่างนี้อยู่รวมกันใน src/data_prep.py ไฟล์เดียว ย้ายมาไว้ที่นี่เพื่อให้
scripts/prepare_dataset.py เหลือแค่เป็น thin entry point
"""

import os
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png")


def extract_dataset(zip_path: str, extract_dir: str) -> None:
    """แตกไฟล์ .zip ไปยังโฟลเดอร์เป้าหมาย (ข้ามถ้าแตกไว้แล้ว)"""
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"ℹ️  {extract_dir} มีข้อมูลอยู่แล้ว ข้ามการแตกไฟล์")
        return

    os.makedirs(extract_dir, exist_ok=True)
    print(f"กำลังแตกไฟล์ {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("✅ แตกไฟล์เสร็จสมบูรณ์")


def find_dir(root: str, target_name: str) -> str | None:
    """
    ค้นหาโฟลเดอร์ที่ชื่อตรงกับ target_name (หรือ target_name + 's') ภายใน root

    หมายเหตุ (แก้บั๊กจาก data_prep.py ต้นฉบับ): เดิมใช้ `target_name in d.lower()`
    ซึ่งจะ match ผิดพลาดกับโฟลเดอร์แม่อย่าง "segmented-images" ตั้งแต่ชั้นแรก (เพราะคำว่า
    "image" เป็น substring ของ "images" ที่อยู่ในชื่อนั้นด้วย) แล้วหยุดค้นหาทันที ไม่มีทาง
    ลงไปเจอโฟลเดอร์ "images/" ตัวจริงที่อยู่ข้างในเลย ที่นี่เปลี่ยนไปเช็ค "ชื่อโฟลเดอร์ตรงกัน
    แบบ exact" ก่อน (รองรับทั้งเอกพจน์/พหูพจน์) แล้วค่อย fallback ไปใช้ substring ถ้าหา
    exact match ไม่เจอเลย
    """
    target = target_name.lower()
    candidates_exact = []
    candidates_substring = []

    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            name = d.lower()
            if name == target or name == target + "s":
                candidates_exact.append(os.path.join(dirpath, d))
            elif target in name:
                candidates_substring.append(os.path.join(dirpath, d))

    if candidates_exact:
        # ถ้ามีหลายอัน เลือกอันที่ path สั้นที่สุด (ตื้นที่สุด) ก่อน
        return min(candidates_exact, key=len)
    if candidates_substring:
        return min(candidates_substring, key=len)
    return None


def mask_to_yolo_polygons(
    mask_path: str,
    img_w: int,
    img_h: int,
    class_id: int = 0,
    min_area: int = 20,
    epsilon_ratio: float = 0.001,
) -> list[str]:
    """
    แปลง binary mask (ขาว-ดำ) เป็น polygon รูปแบบ YOLO segmentation
    คืนค่า list ของบรรทัด "class_id x1 y1 x2 y2 ... xn yn" (พิกัด normalize 0-1)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        coords = []
        for point in approx:
            x, y = point[0]
            coords.append(x / img_w)
            coords.append(y / img_h)

        line = f"{class_id} " + " ".join(f"{c:.6f}" for c in coords)
        lines.append(line)

    return lines


def split_dataset(image_files: list, ratios: dict, seed: int) -> dict:
    """สุ่มแบ่งรายชื่อไฟล์เป็น train/val/test ตามสัดส่วนที่กำหนด"""
    files = image_files.copy()
    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def find_matching_mask(masks_dir: str, stem: str) -> str | None:
    """หาไฟล์ mask ที่ชื่อ (stem) ตรงกับภาพ โดยไม่สนนามสกุล"""
    for ext in IMG_EXTS:
        candidate = os.path.join(masks_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def process_split(
    split_name: str,
    filenames: list,
    images_dir: str,
    masks_dir: str,
    output_root: str,
    min_area: int,
    epsilon_ratio: float,
) -> dict:
    """คัดลอกภาพ + สร้างไฟล์ label .txt สำหรับ split หนึ่งๆ (train/val/test)
    คืนค่าสถิติ (จำนวนที่สำเร็จ/ข้าม) เพื่อให้ scripts/prepare_dataset.py เอาไป log ได้
    """
    img_out_dir = Path(output_root) / split_name / "images"
    lbl_out_dir = Path(output_root) / split_name / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for filename in filenames:
        stem = Path(filename).stem
        src_img_path = os.path.join(images_dir, filename)

        mask_path = find_matching_mask(masks_dir, stem)
        if mask_path is None:
            print(f"⚠️  ไม่พบ mask สำหรับ {filename} ข้ามไฟล์นี้")
            skipped += 1
            continue

        img = cv2.imread(src_img_path)
        if img is None:
            print(f"⚠️  อ่านภาพ {filename} ไม่ได้ ข้ามไฟล์นี้")
            skipped += 1
            continue
        h, w = img.shape[:2]

        shutil.copy2(src_img_path, img_out_dir / filename)

        polygon_lines = mask_to_yolo_polygons(
            mask_path, w, h, min_area=min_area, epsilon_ratio=epsilon_ratio
        )
        label_path = lbl_out_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(polygon_lines))

    done = len(filenames) - skipped
    print(f"✅ {split_name}: {done}/{len(filenames)} ภาพ -> {img_out_dir}")
    return {"split": split_name, "done": done, "skipped": skipped}


def write_dataset_yaml(output_root: str, class_names: list) -> Path:
    """สร้างไฟล์ dataset.yaml สำหรับให้ train.py เรียกใช้"""
    data = {
        "path": os.path.abspath(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    yaml_path = Path(output_root) / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"✅ สร้างไฟล์ config: {yaml_path}")
    return yaml_path


def prepare_dataset(cfg: dict) -> Path:
    """
    รันขั้นตอนเตรียมข้อมูลทั้งหมดตามค่าใน config.yaml:
    แตกไฟล์ -> หา images/masks -> แบ่ง split -> แปลง label -> สร้าง dataset.yaml
    คืนค่า path ของ dataset.yaml ที่สร้างเสร็จ
    """
    from common.config import resolve_path  # lazy import กัน circular import

    zip_path = resolve_path(cfg["paths"]["raw_zip"])
    raw_dir = resolve_path(cfg["paths"]["raw_extract_dir"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])

    if not zip_path.exists():
        raise FileNotFoundError(
            f"❌ ไม่พบไฟล์ {zip_path} กรุณาวาง hyper-kvasir-segmented-images.zip "
            f"ไว้ที่ {cfg['paths']['raw_zip']}"
        )

    extract_dataset(str(zip_path), str(raw_dir))

    images_dir = find_dir(str(raw_dir), "image")
    masks_dir = find_dir(str(raw_dir), "mask")
    if not (images_dir and masks_dir):
        raise FileNotFoundError("❌ ไม่พบโฟลเดอร์ images หรือ masks ในข้อมูลที่แตกออกมา")

    print(f"📂 พบโฟลเดอร์ images ที่: {images_dir}")
    print(f"📂 พบโฟลเดอร์ masks ที่: {masks_dir}")

    image_files = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)
    )
    print(f"🖼️  จำนวนภาพทั้งหมด: {len(image_files)}")
    if not image_files:
        raise RuntimeError("❌ ไม่พบไฟล์ภาพในโฟลเดอร์ images")

    splits = split_dataset(image_files, cfg["dataset"]["splits"], seed=cfg["seed"])
    print(
        f"📊 แบ่งข้อมูล -> train: {len(splits['train'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])}"
    )

    for split_name, filenames in splits.items():
        process_split(
            split_name,
            filenames,
            images_dir,
            masks_dir,
            str(processed_dir),
            min_area=cfg["dataset"]["min_contour_area"],
            epsilon_ratio=cfg["dataset"]["polygon_epsilon_ratio"],
        )

    yaml_path = write_dataset_yaml(str(processed_dir), cfg["dataset"]["class_names"])
    print("\n🎉 เตรียมข้อมูลเสร็จสมบูรณ์! พร้อมสำหรับรัน scripts/train.py")
    return yaml_path
