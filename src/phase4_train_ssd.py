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
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image

from detection_utils import train_one_epoch, evaluate, save_sample_predictions


####################
# Hyperparameters
####################
learning_rate = 0.01
batch_size = 16
epochs = 30
weight_decay = 5e-4
lr_step_size = 10
lr_gamma = 0.1
num_workers = 4

save_pred_every = 5


####################
# Dataset
####################
CLASS_NAMES = ["__background__", "D00", "D10", "D20", "D40"]
NUM_CLASSES = len(CLASS_NAMES)


class RDDDataset(Dataset):
    def __init__(self, manifest_path, split="train", transforms=None, data_root=None):
        with open(manifest_path, "r") as f:
            all_records = json.load(f)
        self.records = [r for r in all_records if r["split"] == split]
        self.transforms = transforms
        self.data_root = Path(data_root) if data_root is not None else DATA_PROC
        print(f"  Dataset [{split}]: {len(self.records)} images  (root: {self.data_root})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(self.data_root / rec["image_path"]).convert("RGB")

        # shift from 0-indexed (phase3) to 1-indexed (torchvision)
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
            if len(target["boxes"]) > 0:
                w = img.shape[-1]
                boxes = target["boxes"].clone()
                boxes[:, 0] = w - target["boxes"][:, 2]
                boxes[:, 2] = w - target["boxes"][:, 0]
                target["boxes"] = boxes
        return img, target


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)

    def __call__(self, img, target):
        return self.jitter(img), target


def get_train_transforms():
    return Compose([ColorJitter(), ToTensor(), RandomHorizontalFlip(prob=0.5)])


def get_val_transforms():
    return Compose([ToTensor()])


def collate_fn(batch):
    return tuple(zip(*batch))


torch.manual_seed(42)

train_dataset = RDDDataset(MANIFEST_PATH, split="train", transforms=get_train_transforms())
val_dataset = RDDDataset(MANIFEST_PATH, split="val", transforms=get_val_transforms())

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")


####################
# Model
####################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = ssdlite320_mobilenet_v3_large(
    weights=None,
    weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    num_classes=NUM_CLASSES,
)
model = model.to(device)
print(model.__class__.__name__)
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)


####################
# Model Training
####################
run_name = (
    f"trainer_{model.__class__.__name__}"
    f"_lr{learning_rate}_bs{batch_size}_ep{epochs}"
    f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
)
run_dir = os.path.join(RUNS_DIR, run_name)
os.makedirs(run_dir, exist_ok=True)

writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
print(f"\nTensorBoard: tensorboard --logdir {run_dir}/tensorboard")
print(f"Run dir: {run_dir}\n")

best_vmap = 0.0
best_epoch = 0

for epoch in range(epochs):
    print(f"Epoch {epoch+1} / {epochs}  (LR={scheduler.get_last_lr()[0]:.5f})")

    train_loss = train_one_epoch(train_dl, model, optimizer, epoch, device, writer)
    map50, map5095, per_class_ap = evaluate(val_dl, model, device)

    scheduler.step()

    writer.add_scalars("Train vs. Valid/loss", {"train": train_loss}, epoch)
    writer.add_scalars("Performance/mAP", {"mAP50": map50, "mAP5095": map5095}, epoch)
    writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

    if per_class_ap.numel() >= 4:
        writer.add_scalars("Performance/AP_per_class", {
            "D00": per_class_ap[0].item(),
            "D10": per_class_ap[1].item(),
            "D20": per_class_ap[2].item(),
            "D40": per_class_ap[3].item(),
        }, epoch)
        print(f"  mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map5095:.4f} | "
              f"D00:{per_class_ap[0]:.3f} D10:{per_class_ap[1]:.3f} "
              f"D20:{per_class_ap[2]:.3f} D40:{per_class_ap[3]:.3f}")
    else:
        print(f"  mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map5095:.4f}")

    if map50 > best_vmap:
        best_vmap = map50
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pth"))
        print(f"  Saved best model (mAP@0.5={map50:.4f})")

    if (epoch + 1) % save_pred_every == 0 or epoch == 0:
        pred_dir = os.path.join(run_dir, "sample_predictions")
        save_sample_predictions(model, val_dataset, device, epoch, pred_dir)

print("Training Complete!")


###########################
# Evaluate on the Best Model
###########################
print(f"\nLoading best model (epoch {best_epoch}, mAP@0.5={best_vmap:.4f})...")
model.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pth")))
model.to(device)

map50, map5095, per_class_ap = evaluate(val_dl, model, device)

print(f"\nBest Model -- Final Val Results:")
print(f"  mAP@0.5     : {map50:.4f}")
print(f"  mAP@0.5:0.95: {map5095:.4f}")
if per_class_ap.numel() >= 4:
    for i, cls in enumerate(["D00", "D10", "D20", "D40"]):
        print(f"  {cls} AP@0.5: {per_class_ap[i].item():.4f}")

save_sample_predictions(model, val_dataset, device, -1,
                        os.path.join(run_dir, "sample_predictions"),
                        n_samples=8)
