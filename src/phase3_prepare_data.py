import os
import sys
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.paths import DATA_RAW, DATA_PROC, OUTPUT_DIR


RANDOM_SEED = 42
VAL_RATIO = 0.20

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

COUNTRIES = ["Japan"]


# --- helpers ---

def parse_xml(xml_path: str):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None, 0, 0, []

    fn_el = root.find("filename")
    filename = fn_el.text if fn_el is not None else Path(xml_path).stem

    size_el = root.find("size")
    img_w = int(size_el.find("width").text) if size_el is not None else 0
    img_h = int(size_el.find("height").text) if size_el is not None else 0

    valid_objects = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_el is None or bndbox is None:
            continue

        label = name_el.text.strip()
        if label not in CLASS_TO_IDX:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (TypeError, AttributeError):
            continue

        xmin = max(0.0, min(xmin, img_w))
        ymin = max(0.0, min(ymin, img_h))
        xmax = max(0.0, min(xmax, img_w))
        ymax = max(0.0, min(ymax, img_h))

        if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
            continue

        valid_objects.append({
            "label": label,
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
        })

    return filename, img_w, img_h, valid_objects


def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return cx, cy, w, h


# --- step 1: collect ---

def collect_image_records(countries, data_raw_dir, split="train"):
    records = []
    for country in countries:
        xml_dir = Path(data_raw_dir) / country / split / "annotations" / "xmls"
        img_dir = Path(data_raw_dir) / country / split / "images"

        if not xml_dir.exists():
            print(f"  [SKIP] {country}/{split} -- no XML directory")
            continue

        xml_files = sorted(xml_dir.glob("*.xml"))
        print(f"  Parsing {country}/{split}: {len(xml_files)} XML files...")

        for xml_path in xml_files:
            image_id = xml_path.stem
            filename, img_w, img_h, objects = parse_xml(str(xml_path))

            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = img_dir / (image_id + ext)
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                continue

            if objects:
                class_counts = defaultdict(int)
                for obj in objects:
                    class_counts[obj["label"]] += 1
                dominant_class = max(class_counts, key=class_counts.get)
            else:
                dominant_class = "BACKGROUND"

            records.append({
                "image_id": image_id,
                "country": country,
                "img_path": img_path,
                "xml_path": xml_path,
                "img_w": img_w,
                "img_h": img_h,
                "objects": objects,
                "dominant_class": dominant_class,
                "is_background": len(objects) == 0,
            })

    return records


# --- step 2: split ---

def stratified_split(records, val_ratio=0.20, seed=42):
    random.seed(seed)

    groups = defaultdict(list)
    for rec in records:
        groups[rec["dominant_class"]].append(rec)

    train_records, val_records = [], []

    for cls, cls_records in sorted(groups.items()):
        random.shuffle(cls_records)
        n_val = max(1, int(len(cls_records) * val_ratio))
        val_records.extend(cls_records[:n_val])
        train_records.extend(cls_records[n_val:])

    random.shuffle(train_records)
    random.shuffle(val_records)

    return train_records, val_records


# --- step 3: write YOLO files ---

def write_yolo_split(records, split_name, out_dir):
    img_out = Path(out_dir) / "images" / split_name
    lbl_out = Path(out_dir) / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for rec in records:
        src_img = rec["img_path"]
        dst_img = img_out / src_img.name
        txt_path = lbl_out / (rec["image_id"] + ".txt")

        try:
            shutil.copy2(src_img, dst_img)
        except Exception as e:
            print(f"  [WARN] Could not copy {src_img}: {e}")
            skipped += 1
            continue

        lines = []
        for obj in rec["objects"]:
            class_idx = CLASS_TO_IDX[obj["label"]]
            cx, cy, w, h = voc_to_yolo(
                obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                rec["img_w"], rec["img_h"]
            )
            lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        copied += 1

    return copied, skipped


# --- step 4: torchvision manifest ---

def write_torchvision_manifest(train_records, val_records, out_dir):
    manifest = []

    for split_name, records in [("train", train_records), ("val", val_records)]:
        for rec in records:
            rel_path = Path("images") / split_name / rec["img_path"].name

            boxes = [[o["xmin"], o["ymin"], o["xmax"], o["ymax"]] for o in rec["objects"]]
            labels = [CLASS_TO_IDX[o["label"]] for o in rec["objects"]]

            manifest.append({
                "image_id": rec["image_id"],
                "image_path": str(rel_path),
                "split": split_name,
                "img_w": rec["img_w"],
                "img_h": rec["img_h"],
                "boxes": boxes,
                "labels": labels,
            })

    out_path = Path(out_dir) / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Torchvision manifest saved: {out_path}  ({len(manifest)} entries)")


# --- step 5: YOLO yaml ---

def write_yolo_yaml(out_dir, train_img_dir, val_img_dir, class_names):
    cfg = {
        "path": str(Path(out_dir).resolve()),
        "train": str(Path(train_img_dir).resolve()),
        "val": str(Path(val_img_dir).resolve()),
        "nc": len(class_names),
        "names": class_names,
    }
    out_path = Path(out_dir) / "dataset.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  YOLO dataset.yaml saved: {out_path}")


# --- step 6: report ---

def print_and_save_report(train_records, val_records, out_dir):
    def count_classes(records):
        counts = defaultdict(int)
        for rec in records:
            for obj in rec["objects"]:
                counts[obj["label"]] += 1
        return counts

    train_cls = count_classes(train_records)
    val_cls = count_classes(val_records)

    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 3: DATA PREPARATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Countries: {COUNTRIES}")
    lines.append(f"Val ratio: {VAL_RATIO:.0%}")
    lines.append(f"Seed: {RANDOM_SEED}")
    lines.append(f"Output: {out_dir}")
    lines.append("")
    lines.append(f"{'Split':<10} {'Total':>10} {'BG-only':>10} {'D00':>7} {'D10':>7} {'D20':>7} {'D40':>7} {'Total Obj':>10}")
    lines.append("-" * 75)
    for split_name, records, cls_counts in [
        ("train", train_records, train_cls),
        ("val", val_records, val_cls),
    ]:
        n_bg = sum(1 for r in records if r["is_background"])
        n_total = len(records)
        n_obj = sum(cls_counts.values())
        lines.append(
            f"{split_name:<10}{n_total:>10}{n_bg:>10}"
            f"{cls_counts.get('D00',0):>7}{cls_counts.get('D10',0):>7}"
            f"{cls_counts.get('D20',0):>7}{cls_counts.get('D40',0):>7}{n_obj:>10}"
        )

    lines.append("")
    lines.append("Formats:")
    lines.append("YOLO txt -> data/processed/japan_only/labels/{train,val}/")
    lines.append("dataset.yaml -> data/processed/japan_only/dataset.yaml")
    lines.append("manifest.json -> data/processed/japan_only/manifest.json")
    lines.append("")
    lines.append("Class index (YOLO):")
    for cls, idx in CLASS_TO_IDX.items():
        lines.append(f"  {idx}: {cls}")
    lines.append("")
    lines.append("NEXT -> Phase 4: Exp 1 -- SSD Baseline Training")

    report = "\n".join(lines)
    print(report)

    report_path = Path(OUTPUT_DIR) / "phase3_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")


def main():
    print("Phase 3: Data Preparation")
    print(f"Countries: {COUNTRIES}")
    print(f"Val ratio: {VAL_RATIO:.0%}")
    print(f"Output: {DATA_PROC}")
    print()

    print("[Step 1] Collecting annotations...")
    records = collect_image_records(COUNTRIES, DATA_RAW, split="train")
    print(f"Total images: {len(records)}")
    print(f"Background-only: {sum(1 for r in records if r['is_background'])}")
    print()

    print("[Step 2] Stratified train/val split...")
    train_records, val_records = stratified_split(records, val_ratio=VAL_RATIO, seed=RANDOM_SEED)
    print(f"Train: {len(train_records)} images")
    print(f"Val: {len(val_records)} images")
    print()

    print("[Step 3] Writing YOLO labels + copying images...")
    print("(copying ~10K images, may take a few minutes...)")
    train_copied, train_skip = write_yolo_split(train_records, "train", DATA_PROC)
    val_copied, val_skip = write_yolo_split(val_records, "val", DATA_PROC)
    print(f"Train: {train_copied} written, {train_skip} skipped")
    print(f"Val: {val_copied} written, {val_skip} skipped")
    print()

    print("[Step 4] Writing torchvision manifest...")
    write_torchvision_manifest(train_records, val_records, DATA_PROC)
    print()

    print("[Step 5] Writing YOLO dataset.yaml...")
    write_yolo_yaml(
        out_dir=DATA_PROC,
        train_img_dir=f"{DATA_PROC}/images/train",
        val_img_dir=f"{DATA_PROC}/images/val",
        class_names=CLASS_NAMES,
    )
    print()

    print("[Step 6] Verification report:")
    print_and_save_report(train_records, val_records, DATA_PROC)


if __name__ == "__main__":
    main()
