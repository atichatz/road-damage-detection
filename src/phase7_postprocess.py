import sys
import csv
import random
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.paths import YOLO_YAML, RUNS_DIR

from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from PIL import Image
import yaml


####################
# Target Run
####################
RUN_NAME = "trainer_YOLOv8s_lr0.01_bs16_ep50_20260402-232703"

# Hyperparameters used during training (needed for labels/display only — no re-training)
learning_rate = 0.01
batch_size    = 16
epochs        = 50
imgsz         = 640

CLASS_NAMES  = ["D00", "D10", "D20", "D40"]
CLASS_COLORS = {"D00": "#E74C3C", "D10": "#3498DB", "D20": "#2ECC71", "D40": "#F39C12"}

device = "0" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*60}")
print(f"PHASE 7 POST-PROCESSING (no re-training)")
print(f"{'='*60}")
print(f"Run : {RUN_NAME}")
print(f"YOLO config: {YOLO_YAML}")
print()


####################
# Paths
####################
run_dir         = RUNS_DIR / RUN_NAME
tb_dir          = run_dir / "tensorboard"
pred_dir        = run_dir / "sample_predictions"
yolo_output_dir = run_dir / "yolo_output"
results_csv     = yolo_output_dir / "train" / "results.csv"
best_pt         = yolo_output_dir / "train" / "weights" / "best.pt"

if not run_dir.exists():
    raise FileNotFoundError(
        f"Run folder not found: {run_dir}\n"
        f"Make sure RUN_NAME matches the folder name in outputs/runs/."
    )

if not best_pt.exists():
    print(f"WARNING: best.pt not found at {best_pt}. Falling back to last.pt.")
    best_pt = yolo_output_dir / "train" / "weights" / "last.pt"
    if not best_pt.exists():
        raise FileNotFoundError("Neither best.pt nor last.pt found. Cannot continue.")

print(f"best.pt : {best_pt}")

writer = SummaryWriter(log_dir=str(tb_dir))

print(f"\n[1/4] Logging metrics to TensorBoard from {results_csv} ...")
epoch_data = []
if results_csv.exists():
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_data.append({k.strip(): v.strip() for k, v in row.items()})

    for i, row in enumerate(epoch_data):
        epoch = i

        def safe_float(key, default=0.0):
            """Parse a CSV field to float; return default on failure."""
            try:
                return float(row.get(key, default))
            except (ValueError, TypeError):
                return default

        train_box_loss = safe_float("train/box_loss")
        train_cls_loss = safe_float("train/cls_loss")
        train_dfl_loss = safe_float("train/dfl_loss")
        train_total = train_box_loss + train_cls_loss + train_dfl_loss

        val_box_loss = safe_float("val/box_loss")
        val_cls_loss = safe_float("val/cls_loss")
        val_dfl_loss = safe_float("val/dfl_loss")

        map50 = safe_float("metrics/mAP50(B)")
        map5095 = safe_float("metrics/mAP50-95(B)")
        precision = safe_float("metrics/precision(B)")
        recall = safe_float("metrics/recall(B)")
        lr_pg0 = safe_float("lr/pg0")

        ratio = (map5095 / map50) if map50 > 0 else 0.0

        # Lab-standard TensorBoard format (same as SSD + Faster R-CNN runs)
        writer.add_scalars("Train vs. Valid/loss", {
            "train": train_total,
            "val": val_box_loss + val_cls_loss + val_dfl_loss
        }, epoch)
        writer.add_scalars("Performance/mAP", {
            "mAP50": map50,
            "mAP5095": map5095
        }, epoch)
        writer.add_scalars("Performance/Precision_Recall", {
            "Precision": precision,
            "Recall": recall
        }, epoch)
        writer.add_scalar("Performance/localization_ratio", ratio, epoch)
        writer.add_scalar("Training/learning_rate", lr_pg0, epoch)
        writer.add_scalars("YOLOv8/train_loss_components", {
            "box": train_box_loss,
            "cls": train_cls_loss,
            "dfl": train_dfl_loss,
        }, epoch)

    print(f"Logged {len(epoch_data)} epochs to TensorBoard.")

    # Find best epoch by mAP@0.5
    best_epoch_idx = max(
        range(len(epoch_data)),
        key=lambda i: float(epoch_data[i].get("metrics/mAP50(B)", 0) or 0)
    )
    best_row = epoch_data[best_epoch_idx]
    best_map50_csv = float(best_row.get("metrics/mAP50(B)", 0))
    print(f"Best epoch (from CSV): {best_epoch_idx + 1}  |  mAP@0.5 = {best_map50_csv:.4f}")

else:
    print(f"WARNING: results.csv not found at {results_csv}. TensorBoard logging skipped.")
    epoch_data = []
    best_epoch_idx = -1
    best_map50_csv = 0.0


print(f"\n[2/4] Running final validation on {best_pt.name} ...")

best_model = YOLO(str(best_pt))

val_results = best_model.val(
    data = str(YOLO_YAML),
    imgsz = imgsz,
    batch = batch_size,
    device = device,
    verbose = True,
    plots = True,
    save_json = False,
    project= str(yolo_output_dir),
    name = "val_best",
    exist_ok = True,
)

final_map50 = float(val_results.box.map50)
final_map5095 = float(val_results.box.map)
final_precision = float(val_results.box.mp)
final_recall = float(val_results.box.mr)
final_ratio = (final_map5095 / final_map50) if final_map50 > 0 else 0.0

# Per-class AP@0.5
per_class_ap50 = {}
if hasattr(val_results.box, 'ap_class_index') and hasattr(val_results.box, 'ap50'):
    for idx, ap in zip(val_results.box.ap_class_index, val_results.box.ap50):
        cls_name = CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else f"cls{idx}"
        per_class_ap50[cls_name] = float(ap)

# Log final per-class AP to TensorBoard (at step = epochs)
if per_class_ap50:
    writer.add_scalars("Performance/AP_per_class", per_class_ap50, epochs)

print(f"\n[ FINAL METRICS — Val Set ]")
print(f"mAP@0.5: {final_map50:.4f}   (Exp 2 FRCNN: 0.5606  |  Exp 1 SSD: 0.2097)")
print(f"mAP@0.5:0.95: {final_map5095:.4f}  (Exp 2 FRCNN: 0.2442  |  Exp 1 SSD: 0.0836)")
print(f"Precision: {final_precision:.4f}")
print(f"Recall: {final_recall:.4f}")
print(f"Ratio (5095/50): {final_ratio:.3f}   (Exp 2 FRCNN: 0.436  |  Exp 1 SSD: 0.400)")

print(f"\n[ PER-CLASS AP @ IoU=0.5 ]")
for name in CLASS_NAMES:
    val = per_class_ap50.get(name, -1.0)
    bar  = "#" * int(max(val, 0) * 30)
    flag = " << minority class" if name == "D40" else ""
    print(f"  {name}: {val:.4f}  |{bar}{flag}")

delta_map50_ssd   = final_map50   - 0.2097
delta_map5095_ssd = final_map5095 - 0.0836
delta_map50_frcnn   = final_map50   - 0.5606
delta_map5095_frcnn = final_map5095 - 0.2442

print(f"\n[ COMPARISON — All 3 Experiments ]")
print(f"{'Metric':<22} {'Exp 1 SSD':>12} {'Exp 2 FRCNN':>12} {'Exp 3 YOLOv8s':>14}")
print(f"{'-'*62}")
print(f"{'mAP@0.5':<22} {'0.2097':>12} {'0.5606':>12} {final_map50:>14.4f}")
print(f"{'mAP@0.5:0.95':<22} {'0.0836':>12} {'0.2442':>12} {final_map5095:>14.4f}")
print(f"{'Loc. ratio':<22} {'0.400':>12} {'0.436':>12} {final_ratio:>14.3f}")
print(f"{'vs SSD (Δ mAP50)':<22} {'—':>12} {'+0.3509':>12} {delta_map50_ssd:>+14.4f}")
print(f"{'vs FRCNN (Δ mAP50)':<22} {'—':>12} {'—':>12} {delta_map50_frcnn:>+14.4f}")


# Save Sample Predictions
print(f"\n[3/4] Saving sample predictions ...")
pred_dir.mkdir(parents=True, exist_ok=True)

with open(YOLO_YAML, "r") as f:
    yolo_cfg = yaml.safe_load(f)

val_img_dir = Path(yolo_cfg["val"])
val_images  = sorted(val_img_dir.glob("*.jpg"))[:200]
if len(val_images) == 0:
    val_images = sorted(val_img_dir.glob("*.png"))[:200]

n_samples   = min(8, len(val_images))
sample_imgs = random.sample(val_images, n_samples)

sample_results = best_model.predict(
    source = [str(p) for p in sample_imgs],
    imgsz = imgsz,
    device = device,
    conf = 0.3,
    verbose = False,
)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(
    f"Exp 3 — YOLOv8s | Val Predictions | Best Model\n"
    f"mAP@0.5: {final_map50:.4f}  |  mAP@0.5:0.95: {final_map5095:.4f}",
    fontsize=12, fontweight="bold"
)
axes = axes.flatten()

for plot_idx, (img_path, result) in enumerate(zip(sample_imgs, sample_results)):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img) / 255.0
    ax = axes[plot_idx]
    ax.imshow(img_np)

    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_idx in zip(boxes, confs, labels):
            xmin, ymin, xmax, ymax = box
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else "?"
            color = CLASS_COLORS.get(cls_name, "#FFFFFF")
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(xmin, max(ymin - 5, 0), f"{cls_name} {conf:.2f}",
                    color=color, fontsize=7, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))
        n_pred = len(boxes)
    else:
        n_pred = 0

    ax.set_title(f"Pred: {n_pred}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
pred_save_path = pred_dir / "best_model_sample_predictions.png"
plt.savefig(pred_save_path, bbox_inches="tight", dpi=100)
plt.close(fig)
print(f"Sample predictions saved: {pred_save_path.name}")
print(f"\n[ ALL OUTPUTS SAVED ]")
print(f"TensorBoard : {tb_dir}")
print(f"Predictions : {pred_dir}")
print(f"YOLO val : {yolo_output_dir / 'val_best'}")
print("=" * 60)
