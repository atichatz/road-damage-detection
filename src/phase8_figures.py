import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.paths import FIGURES_DIR, OUTPUT_DIR

PHASE8_FIG_DIR = FIGURES_DIR / "phase8_comparison"
PHASE8_FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print("PHASE 8: Final Evaluation Figures")
print(f"{'='*60}")
print(f"Output: {PHASE8_FIG_DIR}")
print()

####################
# Hardcoded Results
####################
MODELS = ["SSD\n(MobileNetV3)", "Faster R-CNN\n(ResNet-50 FPN)", "YOLOv8s\n(Anchor-free)"]
MODEL_LABELS = ["SSD", "Faster R-CNN", "YOLOv8s"]
COLORS = ["#E74C3C", "#3498DB", "#2ECC71"]
CLASS_NAMES = ["D00\n(Long. Crack)", "D10\n(Trans. Crack)", "D20\n(Alligator)", "D40\n(Pothole)"]
CLASS_COLORS = ["#9B59B6", "#E67E22", "#1ABC9C", "#E74C3C"]

# Final model: YOLOv8s Run B (imgsz=832)
MAP50 = [0.2097, 0.5606, 0.5616]
MAP5095 = [0.0836, 0.2442, 0.2648]
PRECISION = [None, None, 0.5782]
RECALL = [None, None, 0.5310]
LOC_RATIO = [0.400, 0.436, 0.472]
SPEED_MS = [10.0, 175.0, 11.0]

# SSD and FRCNN: per-class AP not available (class_metrics was not set during those runs)
PER_CLASS_AP = {
    "SSD": [0.0, 0.0, 0.0, 0.0],
    "Faster R-CNN": [0.0, 0.0, 0.0, 0.0],
    "YOLOv8s": [0.4711, 0.4614, 0.7014, 0.6123],
}

PARAMS_M = [5.1, 41.8, 11.1]

###########################
# Figure 1: mAP Comparison Bar Chart
###########################
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(MODELS))
width = 0.35
bars1 = ax.bar(x - width/2, MAP50, width, label="mAP@0.5",
               color=[c + "CC" for c in COLORS], edgecolor="black", linewidth=0.8)
bars2 = ax.bar(x + width/2, MAP5095, width, label="mAP@0.5:0.95",
               color=COLORS, edgecolor="black", linewidth=0.8, alpha=0.7)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xlabel("Model", fontsize=13)
ax.set_ylabel("mAP Score", fontsize=13)
ax.set_title("Overall Detection Performance -- 3 Experiments\n"
             "(RDD2022 Japan-only Val Set, 2099 images)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=11)
ax.set_ylim(0, 0.70)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

ax.annotate("Slightly better than FRCNN\n(Δ = +0.001) & 16x faster",
            xy=(2 - width/2, MAP50[2]),
            xytext=(1.52, 0.63),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

plt.tight_layout()
fig1_path = PHASE8_FIG_DIR / "fig1_map_comparison.png"
plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[1/5] Saved: {fig1_path.name}")


###########################
# Figure 2: Per-Class AP@0.5 (YOLOv8s)
###########################
fig, ax = plt.subplots(figsize=(9, 5))

class_ap = PER_CLASS_AP["YOLOv8s"]
bars = ax.bar(CLASS_NAMES, class_ap, color=CLASS_COLORS, edgecolor="black", linewidth=0.8, width=0.55)

for bar, val in zip(bars, class_ap):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_xlabel("Damage Class", fontsize=13)
ax.set_ylabel("AP @ IoU=0.5", fontsize=13)
ax.set_title("YOLOv8s Run B Per-Class AP@0.5 -- Final Model (imgsz=832)\n"
             "(SSD & Faster R-CNN per-class AP not available)", fontsize=11, fontweight="bold")
ax.set_ylim(0, 0.85)
ax.grid(axis="y", alpha=0.3)

ax.annotate("Minority class (13.6% train)\nAP=0.612 -- strong result",
            xy=(3, class_ap[3]),
            xytext=(2.45, 0.76),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

ax.annotate("Weakest -- thin vertical cracks",
            xy=(1, class_ap[1]),
            xytext=(0.3, 0.60),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFE4E1", edgecolor="gray"))

plt.tight_layout()
fig2_path = PHASE8_FIG_DIR / "fig2_perclass_ap.png"
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[2/5] Saved: {fig2_path.name}")


###########################
# Figure 3: Speed vs Accuracy
###########################
fig, ax = plt.subplots(figsize=(9, 6))

for i, (label, speed, map50, color) in enumerate(zip(MODEL_LABELS, SPEED_MS, MAP50, COLORS)):
    ax.scatter(speed, map50, s=250, color=color, zorder=5, edgecolors="black", linewidth=1.2)
    offset_x = 5 if label != "Faster R-CNN" else -45
    offset_y = 0.008 if label != "YOLOv8s" else -0.025
    ax.annotate(label,
                xy=(speed, map50),
                xytext=(speed + offset_x, map50 + offset_y),
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color + "44", edgecolor="none"))

ax.annotate("", xy=(SPEED_MS[2], MAP50[2]), xytext=(SPEED_MS[0], MAP50[0]),
            arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5, linestyle="dashed"))

ax.set_xlabel("Inference Speed (ms/image)  [lower is faster]", fontsize=12)
ax.set_ylabel("mAP@0.5  [higher is better]", fontsize=12)
ax.set_title("Speed vs Accuracy Trade-off\n"
             "Ideal = top-left corner (fast + accurate)", fontsize=12, fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(3, 400)
ax.set_ylim(0.10, 0.70)
ax.grid(True, alpha=0.3)

ax.annotate("Best trade-off:\nhigh accuracy + fast inference",
            xy=(SPEED_MS[2], MAP50[2]),
            xytext=(18, 0.45),
            arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=1.5),
            fontsize=9, color="#27AE60",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#D5F5E3", edgecolor="#2ECC71"))

plt.tight_layout()
fig3_path = PHASE8_FIG_DIR / "fig3_speed_vs_accuracy.png"
plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[3/5] Saved: {fig3_path.name}")


###########################
# Figure 4: Localization Ratio Progression
###########################
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(MODEL_LABELS, LOC_RATIO, "o-", color="#8E44AD", linewidth=2.5, markersize=12,
        markerfacecolor="white", markeredgewidth=2.5, zorder=5)

for i, (label, val) in enumerate(zip(MODEL_LABELS, LOC_RATIO)):
    ax.text(i, val + 0.004, f"{val:.3f}", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#8E44AD")

ax.axhline(y=0.50, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="0.50 target")
ax.set_xlabel("Experiment", fontsize=13)
ax.set_ylabel("Localization Ratio\n(mAP@0.5:0.95 / mAP@0.5)", fontsize=12)
ax.set_title("Localization Quality Progression\n"
             "Ratio closer to 1.0 = more precise bounding boxes", fontsize=11, fontweight="bold")
ax.set_ylim(0.35, 0.55)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.fill_between(range(len(MODEL_LABELS)), LOC_RATIO, 0.35, alpha=0.08, color="#8E44AD")

plt.tight_layout()
fig4_path = PHASE8_FIG_DIR / "fig4_localization_ratio.png"
plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[4/5] Saved: {fig4_path.name}")


###########################
# Figure 5: Summary Table
###########################
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")

table_data = [
    ["Metric", "Exp 1: SSD", "Exp 2: Faster R-CNN", "Exp 3: YOLOv8s (Run B)"],
    ["Architecture", "SSDLite320", "ResNet-50 FPN", "YOLOv8s"],
    ["Type", "1-stage anchor", "2-stage anchor", "1-stage anchor-free"],
    ["Params", "5.1M", "41.8M", "11.1M"],
    ["Input size", "320px", "~800px (FPN)", "832px"],
    ["mAP@0.5", "0.2097", "0.5606", "0.5616"],
    ["mAP@0.5:0.95", "0.0836", "0.2442", "0.2648"],
    ["Localization ratio", "0.400", "0.436", "0.472"],
    ["Speed (ms/img)", "~10ms", "~175ms", "~11ms"],
    ["D00 AP@0.5", "N/A", "N/A", "0.471"],
    ["D10 AP@0.5", "N/A", "N/A", "0.461"],
    ["D20 AP@0.5", "N/A", "N/A", "0.701"],
    ["D40 AP@0.5", "N/A", "N/A", "0.612"],
    ["Deploy ready?", "No (weak accuracy)", "No (slow 175ms)", "Yes -- Best choice"],
]

col_colors = [["#ECF0F1"]*4] + [
    ["#ECF0F1", "#FAD7A0", "#AED6F1", "#A9DFBF"] for _ in range(len(table_data)-1)
]
col_colors[0] = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71"]

table = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
    cellColours=col_colors[1:],
    colColours=col_colors[0],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for j in range(4):
    table[0, j].get_text().set_color("white")
    table[0, j].get_text().set_fontweight("bold")

ax.set_title("3-Experiment Summary -- Road Damage Detection (RDD2022 Japan)",
             fontsize=13, fontweight="bold", pad=15)

plt.tight_layout()
fig5_path = PHASE8_FIG_DIR / "fig5_summary_table.png"
plt.savefig(fig5_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[5/5] Saved: {fig5_path.name}")
print(f"\n[ ALL FIGURES SAVED TO: {PHASE8_FIG_DIR} ]")
print("=" * 60)
