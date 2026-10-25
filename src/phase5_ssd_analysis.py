###########################
# Import Python Packages
###########################
import os
import sys
import json
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.paths import DATA_PROC, MANIFEST_PATH, RUNS_DIR, OUTPUT_DIR, FIGURES_DIR

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
import torchvision.transforms.functional as TF
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


####################
# Configuration
####################
RUN_NAME      = "trainer_SSD_lr0.01_bs16_ep30_20260401-210026"
SCORE_THRESH  = 0.3 
IOU_THRESH    = 0.5
N_SAMPLES     = 8
RANDOM_SEED   = 42

# Class setup (torchvision 1-indexed; 0 = background)
CLASS_NAMES  = ["__background__", "D00", "D10", "D20", "D40"]
DAMAGE_NAMES = ["D00", "D10", "D20", "D40"]   # 4 damage classes (no background)
NUM_CLASSES  = 5
CLASS_COLORS = {1: "#E74C3C", 2: "#3498DB", 3: "#2ECC71", 4: "#F39C12"}


####################
# Paths
####################
RUN_DIR      = RUNS_DIR / RUN_NAME
CKPT_PATH    = RUN_DIR / "model_best.pth"
TB_DIR       = RUN_DIR / "tensorboard"
PRED_DIR     = RUN_DIR / "sample_predictions"
OUT_FIGURES  = FIGURES_DIR
os.makedirs(OUT_FIGURES, exist_ok=True)

print("=" * 60)
print("PHASE 5: SSD BASELINE ANALYSIS")
print("=" * 60)
print(f"Run : {RUN_NAME}")
print(f"Checkpoint: {CKPT_PATH}")
print(f"TBEvents : {TB_DIR}")
print()

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
print()


####################
# Dataset
####################
class RDDDataset(Dataset):
    """
    Road Damage Detection val dataset — same as Phase 4.
    Reads from manifest.json created in Phase 3.
    """
    def __init__(self, manifest_path, split="val", data_root=None):
        with open(manifest_path, "r") as f:
            all_records = json.load(f)
        self.records  = [r for r in all_records if r["split"] == split]
        self.data_root = Path(data_root) if data_root else DATA_PROC

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        img   = Image.open(self.data_root / rec["image_path"]).convert("RGB")
        img_t = TF.to_tensor(img)

        boxes  = torch.as_tensor(rec["boxes"],  dtype=torch.float32)
        labels = torch.as_tensor(rec["labels"], dtype=torch.int64)

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        return img_t, {"boxes": boxes, "labels": labels}


def detection_collate(batch):
    """Custom collate_fn for detection — keeps variable-length targets as a list."""
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


print("Loading val dataset ...")
val_dataset = RDDDataset(MANIFEST_PATH, split="val")
val_loader  = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    collate_fn=detection_collate
)
print(f"Val images: {len(val_dataset)}")
print()


####################
# Model
####################
print("Loading model from checkpoint ...")
model = ssdlite320_mobilenet_v3_large(
    weights=None,
    weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,  # must match Phase 4 training
    num_classes=NUM_CLASSES,
)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt)
model.to(device)
model.eval()
print("Model loaded successfully.")
print()


###########################
# Section 1 — Learning Curves
# Read TensorBoard event files and plot all tracked scalars
###########################
print("[1/4] Plotting learning curves from TensorBoard ...")

def read_tb_scalars(event_dir, tag):
    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    if tag not in available:
        print(f"WARNING: tag '{tag}' not found. Available: {available}")
        return [], []
    events = ea.Scalars(tag)
    steps  = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

tb_subdirs = list(TB_DIR.iterdir()) if TB_DIR.exists() else []
print(f"  TensorBoard sub-dirs: {[d.name for d in tb_subdirs]}")

# Collect train loss
loss_steps, loss_vals = [], []
map50_steps, map50_vals = [], []
map95_steps, map95_vals = [], []

for subdir in tb_subdirs:
    if not subdir.is_dir():
        continue
    name = subdir.name

    if "loss" in name.lower() and "train" in name.lower():
        s, v = read_tb_scalars(subdir, "Train vs. Valid/loss")
        if s:
            loss_steps, loss_vals = s, v
            print(f"Loaded train loss : {len(s)} points from '{name}'")

    elif "mAP_mAP50" in name and "mAP5095" not in name:
        s, v = read_tb_scalars(subdir, "Performance/mAP")
        if s:
            map50_steps, map50_vals = s, v
            print(f"Loaded mAP@0.5: {len(s)} points from '{name}'")

    elif "mAP_mAP5095" in name:
        s, v = read_tb_scalars(subdir, "Performance/mAP")
        if s:
            map95_steps, map95_vals = s, v
            print(f"Loaded mAP@0.5:0.95: {len(s)} points from '{name}'")

# Plot: learning curves (loss + mAP in two panels, same x-axis)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Exp 1 — SSD Baseline | Learning Curves", fontsize=14, fontweight="bold")

# Panel 1: Train Loss
ax1 = axes[0]
if loss_steps:
    ax1.plot(loss_steps, loss_vals, color="#E74C3C", linewidth=2, label="Train Loss")
    # Mark the LR decay points
    for decay_ep in [10, 20]:
        ax1.axvline(x=decay_ep, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax1.text(decay_ep + 0.3, max(loss_vals) * 0.95,
                 f"LR÷10\n(ep{decay_ep})", fontsize=8, color="grey")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
else:
    ax1.text(0.5, 0.5, "No loss data found", ha="center", va="center",
             transform=ax1.transAxes, fontsize=12, color="red")
    ax1.set_title("Training Loss (data unavailable)")

# Panel 2: mAP@0.5 and mAP@0.5:0.95
ax2 = axes[1]
if map50_steps:
    ax2.plot(map50_steps, map50_vals,
             color="#3498DB", linewidth=2, marker="o", markersize=3, label="mAP@0.5")
if map95_steps:
    ax2.plot(map95_steps, map95_vals,
             color="#F39C12", linewidth=2, marker="s", markersize=3, label="mAP@0.5:0.95")
if map50_steps or map95_steps:
    for decay_ep in [10, 20]:
        ax2.axvline(x=decay_ep, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.set_title("Validation mAP")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
else:
    ax2.text(0.5, 0.5, "No mAP data found", ha="center", va="center",
             transform=ax2.transAxes, fontsize=12, color="red")
    ax2.set_title("Validation mAP (data unavailable)")

plt.tight_layout()
curve_path = OUT_FIGURES / "phase5_learning_curves.png"
plt.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {curve_path.name}")
print()


###########################
# Section 2 — Per-Class AP
# Load best model and re-evaluate on val set to get per-class AP
###########################
print("[2/4] Computing per-class AP on val set ...")
metric = MeanAveragePrecision(
    iou_type="bbox",
    box_format="xyxy",
    max_detection_thresholds=[1, 10, 300],
)
metric.warn_on_many_detections = False

with torch.no_grad():
    for images, targets in val_loader:
        images_dev  = [img.to(device) for img in images]
        preds       = model(images_dev)
        preds_filtered = []
        for p in preds:
            keep = p["scores"].cpu() >= SCORE_THRESH
            preds_filtered.append({
                "boxes":  p["boxes"].cpu()[keep],
                "scores": p["scores"].cpu()[keep],
                "labels": p["labels"].cpu()[keep],
            })

        targets_cpu = [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
                       for t in targets]
        metric.update(preds_filtered, targets_cpu)

result = metric.compute()
map50_val = result["map_50"].item()
map95_val = result["map"].item()
_raw = result.get("map_per_class", torch.zeros(4))
if isinstance(_raw, torch.Tensor):
    _raw = _raw.cpu()
    per_cls_ap = _raw.tolist() if _raw.dim() > 0 else [_raw.item()]
else:
    per_cls_ap = [float(_raw)]

while len(per_cls_ap) < 4:
    per_cls_ap.append(-1.0)

print(f"mAP@0.5 : {map50_val:.4f}")
print(f"mAP@0.5:0.95: {map95_val:.4f}")
print(f"Per-class AP@0.5:")
for i, name in enumerate(DAMAGE_NAMES):
    ap_val = per_cls_ap[i] if i < len(per_cls_ap) else -1.0
    print(f"    {name}: {ap_val:.4f}" if ap_val >= 0 else f"    {name}: (not detected)")
print()

# Plot per-class AP bar chart
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
bar_vals = [max(v, 0) for v in per_cls_ap[:4]]   # clip negatives to 0 for display
bars = ax.bar(DAMAGE_NAMES, bar_vals, color=colors, edgecolor="black", linewidth=0.7)
ax.axhline(y=map50_val, color="navy", linestyle="--", linewidth=1.5,
           label=f"Overall mAP@0.5 = {map50_val:.3f}")

for bar, val in zip(bars, per_cls_ap[:4]):
    label = f"{val:.3f}" if val >= 0 else "N/A"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            label, ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xlabel("Damage Class")
ax.set_ylabel("AP @ IoU=0.5")
ax.set_title("Exp 1 — SSD Baseline | Per-Class AP on Val Set\n"
             "(Not logged in Phase 4 — computed post-hoc for analysis)")
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
ap_path = OUT_FIGURES / "phase5_per_class_ap.png"
plt.savefig(ap_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {ap_path.name}")
print()


###########################
# Section 3 — Confusion Matrix
###########################
print("[3/4] Building confusion matrix ...")

def box_iou(box1, box2):
    """Compute IoU between a single box [x1,y1,x2,y2] and an array of boxes."""
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-6)

# Matrix dimensions: rows = actual class, cols = predicted class
# We use 5 categories: D00, D10, D20, D40, Background
# Index: 0=D00, 1=D10, 2=D20, 3=D40, 4=BG
N_CONF = 5
conf_matrix = np.zeros((N_CONF, N_CONF), dtype=int)

# Map torchvision label (1-indexed) -> confusion matrix index (0-indexed)
label_to_idx = {1: 0, 2: 1, 3: 2, 4: 3} # 1=D00->0, 2=D10->1, 3=D20->2, 4=D40->3
BG_IDX = 4

with torch.no_grad():
    for images, targets in val_loader:
        images_dev = [img.to(device) for img in images]
        preds = model(images_dev)

        for pred, target in zip(preds, targets):
            # Filter predictions by score threshold
            keep = pred["scores"] > SCORE_THRESH
            pred_boxes = pred["boxes"][keep].cpu().numpy()
            pred_labels = pred["labels"][keep].cpu().numpy()

            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()

            # Track which preds are matched
            pred_matched = np.zeros(len(pred_boxes), dtype=bool)

            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                gt_idx = label_to_idx.get(int(gt_label), BG_IDX)

                if len(pred_boxes) == 0:
                    conf_matrix[gt_idx][BG_IDX] += 1
                    continue

                ious = box_iou(gt_box, pred_boxes)
                best_i = int(np.argmax(ious))
                best_iou = ious[best_i]

                if best_iou >= IOU_THRESH:
                    pred_idx = label_to_idx.get(int(pred_labels[best_i]), BG_IDX)
                    conf_matrix[gt_idx][pred_idx] += 1
                    pred_matched[best_i] = True
                else:
                    conf_matrix[gt_idx][BG_IDX] += 1

            for i, (box, label) in enumerate(zip(pred_boxes, pred_labels)):
                if not pred_matched[i]:
                    pred_idx = label_to_idx.get(int(label), BG_IDX)
                    conf_matrix[BG_IDX][pred_idx] += 1

print("  Confusion matrix (rows=GT, cols=Pred):")
conf_labels = DAMAGE_NAMES + ["BG"]
for i, row_name in enumerate(conf_labels):
    row_str = "  ".join(f"{conf_matrix[i][j]:5d}" for j in range(N_CONF))
    print(f"{row_name:>4s} | {row_str}")
print(f"| " + "  ".join(f"{n:>5s}" for n in conf_labels))
print()

# Plot confusion matrix heatmap
fig, ax = plt.subplots(figsize=(8, 6))
# Normalise row-wise (recall view): how often each GT class is correctly predicted
conf_norm = conf_matrix.astype(float)
row_sums = conf_norm.sum(axis=1, keepdims=True)
conf_norm = np.divide(conf_norm, row_sums, where=(row_sums > 0))

sns.heatmap(
    conf_norm,
    annot=conf_matrix,# show raw counts
    fmt="d",
    cmap="Blues",
    xticklabels=conf_labels,
    yticklabels=conf_labels,
    linewidths=0.5,
    ax=ax,
    vmin=0, vmax=1
)
ax.set_xlabel("Predicted Class", fontsize=12)
ax.set_ylabel("Actual (GT) Class", fontsize=12)
ax.set_title(
    f"Exp 1 — SSD Baseline | Detection Confusion Matrix\n"
    f"(score > {SCORE_THRESH}, IoU ≥ {IOU_THRESH} to match; cell = count, color = row-normalised recall)",
    fontsize=11
)
plt.tight_layout()
cm_path = OUT_FIGURES / "phase5_confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {cm_path.name}")
print()

###########################
# Section 4 — Qualitative Prediction Review
# Plot sample predictions with overlaid GT boxes
###########################
print("[4/4] Generating qualitative prediction samples ...")

indices = random.sample(range(len(val_dataset)), min(N_SAMPLES, len(val_dataset)))

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle(
    "Exp 1 — SSD Baseline | Best-Epoch Sample Predictions (Val Set)\n"
    "Solid box = prediction | Dashed white = ground truth",
    fontsize=12, fontweight="bold"
)
axes = axes.flatten()

with torch.no_grad():
    for plot_idx, sample_idx in enumerate(indices):
        img_tensor, target = val_dataset[sample_idx]
        pred = model([img_tensor.to(device)])[0]

        keep = pred["scores"] > SCORE_THRESH
        pred_boxes = pred["boxes"][keep].cpu()
        pred_labels = pred["labels"][keep].cpu()
        pred_scores = pred["scores"][keep].cpu()

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        ax = axes[plot_idx]
        ax.imshow(img_np)

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            xmin, ymin, xmax, ymax = box.tolist()
            cls_name = CLASS_NAMES[label.item()] if label.item() < len(CLASS_NAMES) else "?"
            color = CLASS_COLORS.get(label.item(), "#FFFFFF")
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(xmin, max(ymin - 5, 0), f"{cls_name} {score:.2f}",
                    color=color, fontsize=7, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

        for box, label in zip(target["boxes"], target["labels"]):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)

        n_pred = keep.sum().item()
        n_gt = len(target["boxes"])
        ax.set_title(f"Pred: {n_pred}  |  GT: {n_gt}", fontsize=9)
        ax.axis("off")

plt.tight_layout()
qual_path = OUT_FIGURES / "phase5_sample_predictions_best.png"
plt.savefig(qual_path, dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {qual_path.name}")
print()