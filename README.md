# Hyper-Kvasir Segmented — Polyp Detection (YOLO11-Seg)

โปรเจกต์เทรนโมเดล **YOLO11 Instance Segmentation** สำหรับตรวจจับและตีกรอบติ่งเนื้อ (polyp)
ในภาพส่องกล้องทางเดินอาหาร

โครงสร้างนี้แยกจากเวอร์ชันต้นฉบับ ([Stamplll/hyper-kvasir-segmented](https://github.com/Stamplll/hyper-kvasir-segmented))
ที่รวมทุกอย่างไว้ใน `src/` 3 ไฟล์ ให้เป็นสไตล์ `common/` + `scripts/` (thin entry points)
เหมือนโปรเจกต์ classification benchmark เพื่อ:
- แยก logic ที่ใช้ซ้ำ (data prep, training, evaluation) ออกจากสคริปต์ที่รันจริง
- เพิ่ม `config.yaml` เป็นจุดศูนย์กลาง ไม่ต้องแก้ตัวเลขกระจายในหลายไฟล์
- เพิ่มขั้นตอน **evaluate** ที่เวอร์ชันเดิมไม่มี (มีแค่ train/predict)

## โครงสร้างโปรเจกต์

```
hyper-kvasir-segmented-modular/
├── config.yaml              ← ค่าคอนฟิกทั้งหมด แก้ที่นี่ที่เดียว
├── requirements.txt
│
├── common/                  ← logic หลัก ใช้ร่วมกันทุกสคริปต์
│   ├── __init__.py
│   ├── config.py            โหลด config.yaml, resolve path, ตั้ง seed
│   ├── hardware.py          ตรวจ GPU/CPU, เลือก device
│   ├── data.py               แตก zip → หา images/masks → mask→polygon → split → dataset.yaml
│   ├── yolo_trainer.py       ครอบ ultralytics model.train()
│   ├── evaluate.py           ครอบ model.val() บน test set (ของใหม่ — เดิมไม่มี)
│   ├── metrics.py            ดึง/บันทึก/แสดงผล metrics แบบเดียวกันทุกที่
│   ├── predictor.py          ครอบ model.predict() (ย้ายมาจาก src/predict.py)
│   ├── timing.py             จับเวลาแต่ละ stage
│   ├── plots.py               กราฟเปรียบเทียบผล (เผื่อรันหลายรอบ/config ในอนาคต)
│   └── train.py               entry point กลาง: prepare (ถ้ายังไม่มี) → train
│
├── scripts/                 ← ตัวที่ผู้ใช้เรียกจริงจาก command line (ไฟล์บางๆ)
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── data/
│   ├── raw/                 วาง hyper-kvasir-segmented-images.zip ไว้ที่นี่
│   └── processed/           สร้างอัตโนมัติ (train/val/test + dataset.yaml)
├── weights/                 เก็บ best.pt ที่เทรนเสร็จ
└── runs/                    ผลลัพธ์การเทรน/evaluate/predict ของ ultralytics
```

## การติดตั้ง

### 1. สร้าง virtual environment และติดตั้ง dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. ติดตั้ง PyTorch แบบ CUDA (จำเป็นถ้าจะเทรนบน GPU)
> **⚠️ ขั้นตอนนี้จำเป็น ไม่ใช่ทางเลือก (โดยเฉพาะบน Windows)**
> `pip install -r requirements.txt` ในขั้นตอนที่ 1 มักจะดึง PyTorch เวอร์ชัน **CPU-only**
> (`+cpu`) มาลงโดยอัตโนมัติ ถ้าข้ามขั้นตอนนี้ไป โค้ดจะยังรันได้ปกติแต่ **จะไม่ใช้ GPU เลย**
> แม้เครื่องจะมีการ์ดจอ — การเทรนจะช้าลงมาก (หลักชั่วโมง → หลักนาทีต่อ epoch ต่างกันได้หลายเท่า)
> โดยไม่มี error หรือคำเตือนใดๆ ให้สังเกต

```bash
python -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```
คำสั่งนี้จะบังคับถอนแล้วติดตั้ง `torch`/`torchvision` build ที่คอมไพล์มาพร้อม CUDA 12.1 ทับตัว
`+cpu` ที่อาจติดมาจากขั้นตอนที่ 1 (ต้องมี NVIDIA GPU + driver ที่รองรับ CUDA 12.1 ขึ้นไป)

### 3. ตรวจสอบว่า PyTorch มองเห็น GPU แล้ว
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
ถ้าได้ผลลัพธ์ `True` แปลว่าพร้อมเทรนบน GPU แล้ว ถ้าได้ `False` ให้กลับไปรันคำสั่งในขั้นตอนที่ 2
ใหม่ และตรวจสอบว่าเวอร์ชัน CUDA driver ในเครื่องตรงกับ `cu121` หรือไม่ (เช็คได้ด้วย `nvidia-smi`)

## วิธีใช้งาน

### 1. เตรียมข้อมูล
วาง `hyper-kvasir-segmented-images.zip` ไว้ที่ `data/raw/` แล้วรัน:
```bash
python scripts/prepare_dataset.py
```

### 2. เทรนโมเดล
```bash
python scripts/train.py
```
(ถ้าต้องการบังคับใช้ GPU ตัวแรก ใช้ `python scripts/train.py --device 0`)
(ถ้าจะบังคับ CPU ใช้ `python scripts/train.py --device cpu`)
(ถ้ายังไม่เคยรัน prepare_dataset จะรันให้อัตโนมัติก่อนเทรน)

ผลลัพธ์ (weights, กราฟ loss, confusion matrix) บันทึกที่ `runs/segment/polyp_detection`

### 3. ประเมินผลบน test set (ของใหม่)
```bash
python scripts/evaluate.py --weights runs/segment/polyp_detection/weights/best.pt
```
บันทึก mask/box precision, recall, mAP50, mAP50-95 ไว้ที่ `runs/evaluation/metrics.json`

### 4. รัน Inference
```bash
python scripts/predict.py --source path/to/image_or_video --weights weights/polyp_yolo11n_seg_best.pt --save
```

## สิ่งที่แก้ไข/เพิ่มเติมจากต้นฉบับ

1. **แก้บั๊กใน `find_dir()`** — ต้นฉบับ match โฟลเดอร์ `images` ผิดพลาดกับโฟลเดอร์แม่
   `segmented-images` (เพราะ "image" เป็น substring ของชื่อนั้นด้วย) ทำให้ได้ path ผิด
   ตั้งแต่ก้าวแรก เวอร์ชันนี้เช็ค exact-name match ก่อน แล้วค่อย fallback เป็น substring
   — **ทดสอบรันจริงกับ zip ที่อัปโหลดแล้ว** ได้ 700/150/150 ภาพครบทุกไฟล์ ไม่มีข้ามเลย
2. **เพิ่ม `patience` (early stopping)** ใน config — ต้นฉบับเทรน 100 epochs แบบไม่มี
   early stopping เลย ซึ่งเสี่ยง overfit เพราะมีข้อมูลแค่ ~1,000 ภาพ
3. **เพิ่ม `common/evaluate.py`** — ต้นฉบับมีแค่ train/predict ไม่มีขั้นตอนวัดผลบน test
   set แยกต่างหาก
4. **แยก conf threshold ของ evaluation กับ inference ออกจากกัน** — ต้นฉบับใช้ 0.79
   เป็นค่าเดียวทุกที่ ซึ่งสูงเกินไปสำหรับตอน evaluate (ควรใช้ค่ามาตรฐาน 0.25 เพื่อวัดผล
   ให้เทียบกับงานวิจัยอื่นได้ ส่วน 0.79 ที่สูงเหมาะกับตอน inference จริงที่อยากได้ค่าที่มั่นใจสูง)

## หมายเหตุ
- `data/raw/`, `data/processed/`, `weights/`, `runs/` ถูก `.gitignore` ไว้ (เก็บเฉพาะ `.gitkeep`)
- โมเดลพื้นฐาน `yolo11n-seg.pt` (Nano) — เปลี่ยนได้ที่ `config.yaml` → `model.weights`

**Research and educational use only. ไม่ใช้เพื่อการวินิจฉัยทางคลินิก**