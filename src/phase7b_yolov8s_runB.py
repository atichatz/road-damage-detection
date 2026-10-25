###########################
# Import Python Packages
###########################
import os
import sys
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.paths import YOLO_YAML, RUNS_DIR, OUTPUT_DIR, FIGURES_DIR

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from PIL import Image
import yaml


####################
# Hyperparameters
####################
# Ablation: only imgsz changed from Run A (640 -> 832)
# batch reduced from 16 -> 8 to fit VRAM at higher resolution
model_variant = "yolov8s.pt"
learning_rate = 0.01
lr_final_ratio = 0.01
batch_size = 8
epochs = 50
imgsz = 832
patience = 20
warmup_epochs = 3
weight_decay = 5e-4
momentum = 0.937

num_workers = 8
use_cache = "ram"

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_COLORS = {"D00": "#E74C3C", "D10": "#3498DB", "D20": "#2ECC71", "D40": "#F39C12"}
NUM_CLASSES = 4

# Run A reference for comparison at end
RUN_A_MAP50 = 0.5555
RUN_A_MAP5095 = 0.2578
RUN_A_RATIO = 0.464
RUN_A_PER_CLASS = {"D00": 0.4861, "D10": 0.4388, "D20": 0.6851, "D40": 0.6120}


####################
# Device Check
####################
device = "0" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*65}")
print(f"YOLOv8s ABLATION -- RUN B: Higher Resolution (imgsz=832)")
print(f"{'='*65}")
if torch.cuda.is_available():
    print(f"Device: CUDA GPU 0 -- {torch.cuda.get_device_name(0)}")
else:
    print(f"Device: CPU")
print(f"Config: model={model_variant}  lr={learning_rate}  bs={batch_size}  ep={epochs}")
print(f"imgsz={imgsz}  (Run A was 640)")
print()
print(f"Problem from Run A: D10 AP={RUN_A_PER_CLASS['D10']:.4f} (weakest -- thin cracks)")
print(f"Hypothesis: higher resolution helps thin crack detection")
print()


####################
# Dataset
####################
if not YOLO_YAML.exists():
    raise FileNotFoundError(
        f"dataset.yaml not found at {YOLO_YAML}\n"
        "Run src/phase3_prepare_data.py or src/fix_data_configs.py first."
    )
print(f"[1/4] Dataset config: {YOLO_YAML}")


####################
# Model
####################
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"trainer_YOLOv8s_imgsz832_lr{learning_rate}_bs{batch_size}_ep{epochs}_{timestamp}"
run_dir = RUNS_DIR / run_name
run_dir.mkdir(parents=True, exist_ok=True)
tb_dir = run_dir / "tensorboard"
pred_dir = run_dir / "sample_predictions"

writer = SummaryWriter(log_dir=str(tb_dir))

print(f"\n[2/4] Building YOLOv8s from pretrained COCO weights ({model_variant}) ...")
model = YOLO(model_variant)

print(f"\n[3/4] Training -- run: {run_name}")
print(f"TensorBoard: tensorboard --logdir {tb_dir}")
print()


####################
# Model Training
####################
yolo_output_dir = run_dir / "yolo_output"

print("Starting Ultralytics training ...")
print()

results = model.train(
    data=str(YOLO_YAML),
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    lr0=learning_rate,
    lrf=lr_final_ratio,
    momentum=momentum,
    weight_decay=weight_decay,
    warmup_epochs=warmup_epochs,
    patience=patience,
    device=device,
    workers=num_workers,
    cache=use_cache,
    project=str(yolo_output_dir),
    name="train",
    exist_ok=True,
    verbose=True,
    save=True,
    save_period=-1,
    val=True,
    plots=True,
    augment=False,
    mosaic=1.0,
    mixup=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.0,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
)

print(f"\nUltralytics training complete.")


###########################
# Extract and Log Metrics to TensorBoard
###########################
results_csv = yolo_output_dir / "train" / "results.csv"
print(f"\n[Logging metrics to TensorBoard from {results_csv}]")
epoch_data = []
if results_csv.exists():
    import csv
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_data.append({k.strip(): v.strip() for k, v in row.items()})

    for i, row in enumerate(epoch_data):
        epoch = i

        def safe_float(key, default=0.0):
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
else:
    print(f"WARNING: results.csv not found. TensorBoard logging skipped.")


###########################
# Evaluate on the Best Model
###########################
best_pt = yolo_output_dir / "train" / "weights" / "best.pt"

print(f"\n{'='*65}")
print("FINAL EVALUATION ON BEST CHECKPOINT")
print(f"{'='*65}")

if not best_pt.exists():
    print(f"WARNING: best.pt not found. Falling back to last.pt.")
    best_pt = yolo_output_dir / "train" / "weights" / "last.pt"

best_model = YOLO(str(best_pt))

print(f"\nRunning validation on best model: {best_pt.name}")
val_results = best_model.val(
    data=str(YOLO_YAML),
    imgsz=imgsz,
    batch=batch_size,
    device=device,
    verbose=True,
    plots=True,
    save_json=False,
    project=str(yolo_output_dir),
    name="val_best",
    exist_ok=True,
)

final_map50 = float(val_results.box.map50)
final_map5095 = float(val_results.box.map)
final_precision = float(val_results.box.mp)
final_recall = float(val_results.box.mr)
final_ratio = (final_map5095 / final_map50) if final_map50 > 0 else 0.0

per_class_ap50 = {}
if hasattr(val_results.box, 'ap_class_index') and hasattr(val_results.box, 'ap50'):
    for idx, ap in zip(val_results.box.ap_class_index, val_results.box.ap50):
        cls_name = CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else f"cls{idx}"
        per_class_ap50[cls_name] = float(ap)

if per_class_ap50:
    writer.add_scalars("Performance/AP_per_class", per_class_ap50, epochs)

print(f"\n[ FINAL METRICS -- Run A vs Run B ]")
print(f"{'Metric':<20} {'Run A (640px)':>14} {'Run B (832px)':>14} {'Delta':>10}")
print(f"{'-'*60}")
print(f"{'mAP@0.5':<20} {RUN_A_MAP50:>14.4f} {final_map50:>14.4f} {final_map50-RUN_A_MAP50:>+10.4f}")
print(f"{'mAP@0.5:0.95':<20} {RUN_A_MAP5095:>14.4f} {final_map5095:>14.4f} {final_map5095-RUN_A_MAP5095:>+10.4f}")
print(f"{'Localization ratio':<20} {RUN_A_RATIO:>14.3f} {final_ratio:>14.3f} {final_ratio-RUN_A_RATIO:>+10.3f}")

print(f"\n[ PER-CLASS AP -- Run A vs Run B ]")
print(f"{'Class':<8} {'Run A':>12} {'Run B':>12} {'Delta':>10}")
print(f"{'-'*46}")
for name in CLASS_NAMES:
    run_a_val = RUN_A_PER_CLASS.get(name, -1.0)
    run_b_val = per_class_ap50.get(name, -1.0)
    delta = run_b_val - run_a_val if run_b_val >= 0 and run_a_val >= 0 else float("nan")
    note = " <- D10 target" if name == "D10" else ""
    print(f"  {name:<8} {run_a_val:>12.4f} {run_b_val:>12.4f} {delta:>+10.4f}  {note}")


###########################
# Save Sample Predictions
###########################
print(f"\n[4/4] Saving sample predictions ...")
pred_dir.mkdir(parents=True, exist_ok=True)

with open(YOLO_YAML, "r") as f:
    yolo_cfg = yaml.safe_load(f)

val_img_dir = Path(yolo_cfg["val"])
val_images = sorted(val_img_dir.glob("*.jpg"))[:200]
if len(val_images) == 0:
    val_images = sorted(val_img_dir.glob("*.png"))[:200]

n_samples = min(8, len(val_images))
sample_imgs = random.sample(val_images, n_samples)

sample_results = best_model.predict(
    source=[str(p) for p in sample_imgs],
    imgsz=imgsz,
    device=device,
    conf=0.3,
    verbose=False,
)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(
    f"YOLOv8s Run B -- imgsz=832 | Val Predictions | Best Model\n"
    f"mAP@0.5: {final_map50:.4f}  D10 AP: {per_class_ap50.get('D10', 0):.4f}",
    fontsize=11, fontweight="bold"
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


###########################
# Save Result Summary
###########################
best_epoch_from_csv = -1
best_map50_from_csv = 0.0
if epoch_data:
    for i, row in enumerate(epoch_data):
        try:
            m = float(row.get("metrics/mAP50(B)", 0))
        except (ValueError, TypeError):
            m = 0.0
        if m > best_map50_from_csv:
            best_map50_from_csv = m
            best_epoch_from_csv = i + 1

d10_run_b = per_class_ap50.get("D10", -1.0)
d10_delta = d10_run_b - RUN_A_PER_CLASS["D10"]
if d10_delta > 0.01:
    conclusion = "CONFIRMED -- imgsz=832 helps thin crack detection."
elif d10_delta > -0.01:
    conclusion = "INCONCLUSIVE -- resolution change had minimal effect."
else:
    conclusion = "NOT CONFIRMED -- D10 did not improve with higher resolution."

summary_lines = [
    "=" * 65,
    "YOLOv8s ABLATION -- RUN B: RESULT SUMMARY",
    "=" * 65,
    f"Run name: {run_name}",
    f"Change from Run A: imgsz 640 -> 832",
    f"Problem: D10 AP={RUN_A_PER_CLASS['D10']:.4f} (weakest class)",
    "",
    f"Best epoch: {best_epoch_from_csv} / {len(epoch_data)}",
    f"Best mAP@0.5: {best_map50_from_csv:.4f}  (from training log)",
    f"Final mAP@0.5: {final_map50:.4f}  (from best.pt re-eval)",
    f"Final mAP@0.5:95: {final_map5095:.4f}",
    f"Precision: {final_precision:.4f}",
    f"Recall: {final_recall:.4f}",
    f"Localization ratio: {final_ratio:.3f}  (Run A: {RUN_A_RATIO})",
    "",
    "[ PER-CLASS AP -- Run A vs Run B ]",
    f"{'Class':<8} {'Run A':>10} {'Run B':>10} {'Delta':>10}",
    f"{'-'*42}",
]
for name in CLASS_NAMES:
    run_a_val = RUN_A_PER_CLASS.get(name, -1.0)
    run_b_val = per_class_ap50.get(name, -1.0)
    delta = run_b_val - run_a_val if run_b_val >= 0 else float("nan")
    target_flag = "  <- PRIMARY" if name == "D10" else ""
    summary_lines.append(f"  {name:<8} {run_a_val:>10.4f} {run_b_val:>10.4f} {delta:>+10.4f}{target_flag}")

summary_lines += [
    "",
    "[ CONCLUSION ]",
    f"D10 delta: {d10_delta:+.4f}",
    f"Verdict: {conclusion}",
    "",
    f"imgsz: {imgsz}  batch: {batch_size}  lr0: {learning_rate}  epochs: {epochs}",
    "",
    "[ OUTPUTS ]",
    f"best.pt: {best_pt}",
    f"TensorBoard: {tb_dir}",
    f"YOLO outputs: {yolo_output_dir / 'train'}",
    f"Predictions: {pred_dir}",
    "=" * 65,
]

summary_path = run_dir / "result_summary.txt"
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

writer.close()

print(f"\n[ OUTPUTS ]")
print(f"best.pt: {best_pt}")
print(f"Summary: {summary_path}")
print(f"TensorBoard: {tb_dir}")
print(f"Predictions: {pred_dir}")
print("=" * 65)
