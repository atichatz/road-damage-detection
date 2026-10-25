# Road Damage Detection Source Code & Results

**Course:** ITCS352 Introduction to Deep Learning | ICT Mahidol 2026  
**Students:** 6688012 (Perawis Buranasing) · 6688193 Atichat Kangsamut

**Task:** Road Damage Detection 
**Dataset:** RDD2022 Japan subset — 4 classes: D00 / D10 / D20 / D40  
**Final model:** YOLOv8s Run B (mAP@0.5 = 0.562, imgsz=832)

---

## Dataset (Google Drive)

The docker, presentation-slide-gr6.pdf, outputs folder and RDD2022 dataset (~10 GB) is NOT included in this folder. because it too large to zip and upload to MyCourses

You can download from: **[Google Drive Click Here!!](https://drive.google.com/drive/folders/1YQY92dxvJJrmRwA8P0NwHahJMPYNYzYU?usp=sharing)**


---

## Folder Structure

```
road-damage-det/
├── README.md                        ← this file
├── report-gr6.pdf                   ← final report
├── presentation-slide-gr6.pdf       ← presentation slides [DOWNLOAD FROM Google Drive]
├── data                             ← RDD2022 dataset [DOWNLOAD FROM Google Drive]
│
├── config/
│   └── paths.py                     ← central path config (edit DATA_PROC here)
│
├── src/
│   ├── detection_utils.py           ← shared: train loop, evaluate, mAP, save predictions
│   ├── fix_data_configs.py          ← one-off: fix paths in dataset.yaml / manifest.json
│   ├── phase1_inspect.py            ← Phase 1: dataset inspection
│   ├── phase3_prepare_data.py       ← Phase 3: XML → YOLO/torchvision convert + split
│   ├── phase4_train_ssd.py          ← Exp 1: SSD MobileNetV3 baseline training
│   ├── phase5_ssd_analysis.py       ← Phase 5: SSD analysis (learning curves, confusion matrix)
│   ├── phase6_train_fasterrcnn.py   ← Exp 2: Faster R-CNN ResNet-50 FPN training
│   ├── phase7_train_yolov8s.py      ← Exp 3: YOLOv8s Run A (imgsz=640) training
│   ├── phase7b_yolov8s_runB.py      ← Ablation: YOLOv8s Run B (imgsz=832, final model)
│   ├── phase7_postprocess.py        ← post-process YOLOv8s Run A results
│   ├── phase8_figures.py            ← generate all 5 comparison figures
│   └── phase8/
│       ├── demo.py                  ← Gradio demo app (image / video / webcam)
│       └── test_img_vdo/            ← sample road images + video for demo testing
│
├── notebooks/
│   └── phase2_eda.ipynb             ← Phase 2: EDA (class distribution, bbox analysis)
│
├── outputs/                         ← [DOWNLOAD FROM Google Drive]
│   ├── figures/                     ← all figures used in the report
│   │   ├── phase2_*.png             ← EDA figures (6 files)
│   │   ├── phase5_*.png             ← SSD analysis figures (4 files)
│   │   └── phase8_comparison/       ← 3-model comparison figures (fig1–fig5 + FRCNN curves)
│   └── runs/                        ← training outputs for all 4 experiments
│       ├── trainer_SSD_*/           ← Exp 1: model_best.pth, sample_predictions, tensorboard
│       ├── trainer_FasterRCNN_*/    ← Exp 2: model_best.pth, sample_predictions, tensorboard
│       ├── trainer_YOLOv8s_lr0.01_bs16_*/    ← Exp 3 Run A: weights, val results, tensorboard
│       └── trainer_YOLOv8s_imgsz832_*/       ← Exp 3 Run B (FINAL): weights, val results, tensorboard
│
└── docker/                          ← [DOWNLOAD FROM Google Drive]
    ├── README.md                    ← Docker instructions
    ├── Dockerfile
    ├── docker-compose.yml
    ├── best.pt                      ← YOLOv8s Run B weights (22 MB)
    ├── config/
    └── src_demo/demo.py
```

---

## Deliverables Summary

| File | Description |
|---|---|
| `report-gr6.pdf` | Final written report |
| `presentation-slide-gr6.pdf` | Presentation slides |
| `src/` | All training and analysis scripts |
| `notebooks/phase2_eda.ipynb` | EDA notebook |
| `outputs/figures/` | All figures in the report |
| `outputs/runs/` | Training results (weights, predictions, TensorBoard logs) |
| `docker/` | Self-contained Docker demo |

---

## Experiment Summary

| Experiment | Model | mAP@0.5 | mAP@0.5:0.95 | Speed |
|---|---|---|---|---|
| Exp 1 (Baseline) | SSD MobileNetV3 | 0.210 | 0.084 | ~12 ms/img |
| Exp 2 | Faster R-CNN ResNet-50 FPN | 0.561 | 0.244 | ~175 ms/img |
| Exp 3 Run A | YOLOv8s (imgsz=640) | 0.556 | 0.258 | ~7.7 ms/img |
| **Exp 3 Run B (Final)** | **YOLOv8s (imgsz=832)** | **0.562** | **0.265** | **~11 ms/img** |

Iterative story: SSD underfits (weak backbone, 320px) → Faster R-CNN improves backbone (+167% mAP) but slow → YOLOv8s matches accuracy + 23× faster → Run B ablation confirms imgsz=832 helps thin-crack detection (D10 AP +0.023).

---

## How to View Results (No Training Required)

### Option A — View figures directly
Open any `.png` in `outputs/figures/` — all comparison charts are ready.

### Option B — View TensorBoard logs
```bash
pip install tensorboard
tensorboard --logdir outputs/runs
# open http://localhost:6006
```
All 4 experiments are logged: loss, mAP, per-class AP, precision, recall.

### Option C — Run the Gradio demo (Docker, easiest)
No Python environment needed. See `docker/README.md` for full instructions.

```bash
cd docker
docker compose up --build
# open http://localhost:7860
```

### Option D — Run the Gradio demo (conda)
```bash
conda activate road-damage-det
python src/phase8/demo.py
# open http://localhost:7860
```

---

## How to Re-run Training Scripts

### 1. Setup environment
```bash
conda create -n road-damage-det python=3.10 -y
conda activate road-damage-det
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy pandas matplotlib seaborn scikit-learn tensorboard tqdm torchmetrics pyyaml pillow gradio
```

### 2. Download dataset from Google Drive
Download `japan_only/` from the Drive link above. Place it at:
- **WSL:** `~/data/japan_only/`
- **Windows:** update `DATA_PROC` in `config/paths.py` to your actual path

### 3. Configure paths
Edit `config/paths.py` — only one line needs changing:
```python
DATA_PROC = Path("/your/path/to/japan_only")   # change this to match your machine
```
`PROJECT_ROOT` is auto-detected from the file's location — no need to edit.

### 4. Run scripts in order
```bash
# (Phase 1 and 3 required only if re-building the processed dataset)
python src/phase1_inspect.py
python src/phase3_prepare_data.py

# Training experiments
python src/phase4_train_ssd.py          # Exp 1: SSD (~2h on RTX 4060)
python src/phase5_ssd_analysis.py       # SSD analysis figures
python src/phase6_train_fasterrcnn.py   # Exp 2: Faster R-CNN (~4h)
python src/phase7_train_yolov8s.py      # Exp 3 Run A: YOLOv8s 640px (~1h)
python src/phase7b_yolov8s_runB.py      # Exp 3 Run B: YOLOv8s 832px (~1.5h)
python src/phase8_figures.py            # Generate comparison figures
```

---

## Notes

- All scripts were developed and tested on **WSL Ubuntu 22.04** with **RTX 4060 8GB VRAM**
- Scripts use paths from `config/paths.py` — no hardcoded paths in src/
- TensorBoard logs are in `outputs/runs/<run_name>/tensorboard/`
- The final model weights are at:  
  `outputs/runs/trainer_YOLOv8s_imgsz832_lr0.01_bs8_ep50_20260403-140823/yolo_output/train/weights/best.pt`
