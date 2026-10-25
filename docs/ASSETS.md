# Dataset and large training artifacts

## Dataset

The dataset is not present in either supplied project directory and is not included
in this repository. The supplied `Dataset_Link.docx` points to:

[RDD2022 on Figshare](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)

Download and extract the raw dataset under `data/raw/RDD2022/`, then configure
`DATA_RAW` and `DATA_PROC` in `config/paths.py` and run the preparation scripts.
The dataset reference warns that the full dataset was too large for the team's Drive.

## Original project assets

The original submission README links to the following folder for Docker, slides,
and training output assets:

[Original project assets on Google Drive](https://drive.google.com/drive/folders/1YQY92dxvJJrmRwA8P0NwHahJMPYNYzYU?usp=sharing)

Availability and access permissions of this external folder have not been verified
during packaging. Local originals are preserved in the two supplied project folders.

## Files omitted from Git

The final YOLOv8s `best.pt` and the Run A best weights are included. Large Faster
R-CNN checkpoints remain external, and last-epoch YOLO weights are omitted in favor
of the saved best checkpoints. Restore large checkpoints under their matching
`outputs/runs/` paths if needed for evaluation or resumed training.

| Original relative path | MiB | Reason |
| --- | ---: | --- |
| `config/__pycache__/__init__.cpython-310.pyc` | 0.00 | Generated Python cache |
| `config/__pycache__/paths.cpython-310.pyc` | 0.00 | Generated Python cache |
| `outputs/outputs/runs/trainer_FasterRCNN_lr0.005_bs4_ep20_20260402-141232/checkpoint_latest.pth` | 314.89 | Large training checkpoint; download from original project assets |
| `outputs/outputs/runs/trainer_FasterRCNN_lr0.005_bs4_ep20_20260402-141232/model_best.pth` | 158.11 | Large training checkpoint; download from original project assets |
| `outputs/outputs/runs/trainer_YOLOv8s_imgsz832_lr0.01_bs8_ep50_20260403-140823/yolo_output/train/weights/last.pt` | 21.51 | Last-epoch checkpoint; best.pt is included instead |
| `outputs/outputs/runs/trainer_YOLOv8s_lr0.01_bs16_ep50_20260402-232703/yolo_output/train/weights/last.pt` | 21.48 | Last-epoch checkpoint; best.pt is included instead |

The demo sample image/video directory mentioned in the original README was absent
from the supplied files. Use your own road images or videos.
