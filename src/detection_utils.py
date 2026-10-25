import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchmetrics.detection.mean_ap import MeanAveragePrecision


# 1-indexed; 0 = background for torchvision
CLASS_NAMES = ["__background__", "D00", "D10", "D20", "D40"]
CLASS_COLORS = {1: "#E74C3C", 2: "#3498DB", 3: "#2ECC71", 4: "#F39C12"}


def train_one_epoch(dataloader, model, optimizer, epoch, device, writer, log_step_interval=50):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # torchvision detection returns a dict of losses in train mode
        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += total_loss.item()

        if (i + 1) % log_step_interval == 0:
            breakdown = "  ".join(
                f"{k.replace('loss_', '')}:{v.item():.3f}"
                for k, v in loss_dict.items()
            )
            print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, "
                  f"Loss: {total_loss.item():.4f}  ({breakdown})")
            writer.add_scalar("Loss/train_step", total_loss.item(), epoch * len(dataloader) + i)

    avg_loss = running_loss / max(len(dataloader), 1)
    return avg_loss


@torch.no_grad()
def evaluate(dataloader, model, device):
    model.eval()
    # class_metrics=True needed to get per-class AP (D00-D40)
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", class_metrics=True)

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        preds = model(images)

        preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
        targets_cpu = [
            {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
            for t in targets
        ]
        metric.update(preds_cpu, targets_cpu)

    result = metric.compute()
    map50 = result["map_50"].item()
    map5095 = result["map"].item()
    per_class_ap = result.get("map_per_class", torch.zeros(4))

    return map50, map5095, per_class_ap


def save_sample_predictions(model, dataset, device, epoch, save_dir,
                            exp_name="Detection Model", n_samples=8, score_thresh=0.3):
    import os
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{exp_name} | Val Predictions | Epoch {epoch+1}", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(indices):
            img_tensor, target = dataset[sample_idx]
            pred = model([img_tensor.to(device)])[0]

            keep = pred["scores"] > score_thresh
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

            # dashed white = ground truth
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
    save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Sample predictions saved: epoch_{epoch+1:03d}.png")
