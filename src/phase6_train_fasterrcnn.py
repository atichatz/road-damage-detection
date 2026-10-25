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
from config.paths import DATA_PROC, MANIFEST_PATH, RUNS_DIR

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

from detection_utils import train_one_epoch, evaluate, save_sample_predictions


####################
# Hyperparameters
####################
learning_rate = 0.005
batch_size = 4
epochs = 20
weight_decay = 5e-4
momentum = 0.9
lr_step_size = 8
lr_gamma = 0.1
warmup_iters = 500
num_workers = 4

save_pred_every = 4


####################
# Dataset
####################
CLASS_NAMES = ["__background__", "D00", "D10", "D20", "D40"]
NUM_CLASSES = len(CLASS_NAMES)
DAMAGE_NAMES = ["D00", "D10", "D20", "D40"]


class RDDDataset(Dataset):
    def __init__(self, manifest_path, split="train", transforms=None):
        with open(manifest_path, "r") as f:
            all_records = json.load(f)
        self.records = [r for r in all_records if r["split"] == split]
        self.transforms = transforms
        print(f"Dataset [{split}]: {len(self.records)} images")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = Path(rec["image_path"])
        if not img_path.is_absolute():
            img_path = DATA_PROC / img_path
        img = Image.open(img_path).convert("RGB")

        labels_1idx = [l + 1 for l in rec["labels"]]

        if rec["boxes"]:
            boxes_t = torch.as_tensor(rec["boxes"], dtype=torch.float32)
            labels_t = torch.as_tensor(labels_1idx, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([idx])}

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor:
    def __call__(self, img, target):
        return TF.to_tensor(img), target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            img = TF.hflip(img)
            w = img.size[0]
            if target["boxes"].shape[0] > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return img, target


def get_train_transforms():
    return Compose([RandomHorizontalFlip(prob=0.5), ToTensor()])


def get_val_transforms():
    return Compose([ToTensor()])


def collate_fn(batch):
    return tuple(zip(*batch))


####################
# Model
####################
def build_faster_rcnn(num_classes, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_optimizer_params(model, lr, weight_decay):
    params_with_decay = []
    params_without_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)
    return [
        {"params": params_with_decay, "weight_decay": weight_decay},
        {"params": params_without_decay, "weight_decay": 0.0},
    ]


def _safe_per_class_ap(tensor, n_classes=4):
    if not isinstance(tensor, torch.Tensor):
        return [-1.0] * n_classes
    if tensor.dim() == 0:
        raw = [tensor.item()]
    else:
        raw = tensor.cpu().tolist()
    raw = list(raw)
    while len(raw) < n_classes:
        raw.append(-1.0)
    return raw[:n_classes]


####################
# Model Training
####################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*60}")
print(f"PHASE 6: Exp 2 -- Faster R-CNN (ResNet-50 FPN)")
print(f"{'='*60}")
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Config: lr={learning_rate} bs={batch_size} ep={epochs}")
print(f"StepLR(step={lr_step_size}, gamma={lr_gamma}) warmup={warmup_iters}")


print("[1/4] Loading datasets ...")
train_dataset = RDDDataset(MANIFEST_PATH, split="train", transforms=get_train_transforms())
val_dataset = RDDDataset(MANIFEST_PATH, split="val", transforms=get_val_transforms())

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

print("\n[2/4] Building Faster R-CNN (ResNet-50 FPN, COCO pretrained) ...")
model = build_faster_rcnn(num_classes=NUM_CLASSES, pretrained=True)
model.to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {n_params:,}")

optimizer = optim.SGD(
    get_optimizer_params(model, learning_rate, weight_decay),
    lr=learning_rate, momentum=momentum
)


def warmup_lambda(step):
    if step < warmup_iters:
        return float(step) / float(max(1, warmup_iters))
    return 1.0


warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"trainer_FasterRCNN_lr{learning_rate}_bs{batch_size}_ep{epochs}_{timestamp}"
run_dir = RUNS_DIR / run_name
run_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
pred_dir = run_dir / "sample_predictions"

print(f"\n[3/4] Training -- run: {run_name}")
print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")
print()

best_map50 = 0.0
best_epoch = 0
global_step = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if global_step < warmup_iters:
            warmup_scheduler.step()
        global_step += 1

        running_loss += total_loss.item()

        if (i + 1) % 50 == 0:
            breakdown = "  ".join(
                f"{k.replace('loss_', '')}:{v.item():.3f}"
                for k, v in loss_dict.items()
            )
            print(f"  Epoch {epoch+1}/{epochs}, Step {i+1}/{len(train_loader)}, "
                  f"Loss: {total_loss.item():.4f}  ({breakdown})")
            writer.add_scalar("Loss/train_step", total_loss.item(), global_step)

    avg_train_loss = running_loss / max(len(train_loader), 1)
    current_lr = optimizer.param_groups[0]["lr"]

    map50, map5095, per_class_ap_tensor = evaluate(val_loader, model, device)

    ap_vals = _safe_per_class_ap(per_class_ap_tensor)
    ap_d00, ap_d10, ap_d20, ap_d40 = ap_vals[:4]
    ratio = (map5095 / map50) if map50 > 0 else 0.0

    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"TrainLoss: {avg_train_loss:.4f} | "
          f"mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map5095:.4f} | "
          f"ratio: {ratio:.3f} | LR: {current_lr:.6f}")
    print(f"           Per-class AP@0.5 -- "
          f"D00:{ap_d00:.3f}  D10:{ap_d10:.3f}  D20:{ap_d20:.3f}  D40:{ap_d40:.3f}")

    writer.add_scalars("Train vs. Valid/loss", {"train": avg_train_loss}, epoch)
    writer.add_scalars("Performance/mAP", {"mAP50": map50, "mAP5095": map5095}, epoch)
    writer.add_scalars("Performance/AP_per_class", {
        "D00": max(ap_d00, 0.0),
        "D10": max(ap_d10, 0.0),
        "D20": max(ap_d20, 0.0),
        "D40": max(ap_d40, 0.0),
    }, epoch)
    writer.add_scalar("Performance/localization_ratio", ratio, epoch)
    writer.add_scalar("Training/learning_rate", current_lr, epoch)

    if map50 > best_map50:
        best_map50 = map50
        best_epoch = epoch + 1
        torch.save(model.state_dict(), run_dir / "model_best.pth")
        print(f"  *** New best mAP@0.5: {best_map50:.4f} at epoch {best_epoch} ***")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "map50": map50,
    }, run_dir / "checkpoint_latest.pth")

    if (epoch + 1) % save_pred_every == 0 or epoch == 0:
        save_sample_predictions(
            model, val_dataset, device, epoch,
            save_dir=str(pred_dir),
            exp_name="Exp 2 -- Faster R-CNN (ResNet-50 FPN)",
            n_samples=8, score_thresh=0.3
        )

    if global_step >= warmup_iters:
        step_scheduler.step()

    print()

writer.close()
print(f"\nTraining complete.")
print(f"Best epoch: {best_epoch}")
print(f"Best mAP@0.5: {best_map50:.4f}")


###########################
# Evaluate on the Best Model
###########################
print(f"\n{'='*60}")
print("FINAL EVALUATION ON BEST CHECKPOINT")
print(f"{'='*60}")

best_ckpt_path = run_dir / "model_best.pth"
model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
model.to(device)

final_map50, final_map5095, final_per_class_ap = evaluate(val_loader, model, device)
final_ap_vals = _safe_per_class_ap(final_per_class_ap)
final_ratio = (final_map5095 / final_map50) if final_map50 > 0 else 0.0

print(f"\n[ FINAL METRICS -- Val Set ]")
print(f"mAP@0.5:     {final_map50:.4f}")
print(f"mAP@0.5:0.95: {final_map5095:.4f}")
print(f"Ratio: {final_ratio:.3f}")

print(f"\n[ PER-CLASS AP @ IoU=0.5 ]")
for name, val in zip(DAMAGE_NAMES, final_ap_vals):
    bar = "#" * int(max(val, 0) * 30)
    print(f"  {name}: {val:.4f}  |{bar}")

print(f"\n[ COMPARISON vs Exp 1 (SSD) ]")
print(f"mAP@0.5:     0.2097 -> {final_map50:.4f}  ({final_map50-0.2097:+.4f})")
print(f"mAP@0.5:0.95: 0.0836 -> {final_map5095:.4f}  ({final_map5095-0.0836:+.4f})")
print(f"Ratio:        0.400  -> {final_ratio:.3f}  ({final_ratio-0.400:+.3f})")

save_sample_predictions(
    model, val_dataset, device, best_epoch - 1,
    save_dir=str(pred_dir),
    exp_name="Exp 2 -- Faster R-CNN (ResNet-50 FPN) | BEST MODEL",
    n_samples=8, score_thresh=0.3
)

print(f"\n[OUTPUTS]")
print(f"Checkpoint: {run_dir / 'model_best.pth'}")
print(f"TensorBoard: {run_dir / 'tensorboard'}")
print(f"Predictions: {pred_dir}")
print("=" * 60)
