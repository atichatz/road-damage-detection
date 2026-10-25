import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.paths import DATA_RAW, OUTPUT_DIR


DATA_DIR = DATA_RAW
REPORT_PATH = OUTPUT_DIR / "phase1_report.txt"

ALL_COUNTRIES = [
    "Japan",
    "India",
    "Norway",
    "United_States",
    "Czech",
    "China_MotorBike",
    "China_Drone",
]

EXPECTED_CLASSES = {"D00", "D10", "D20", "D40"}
SPLITS = ["train", "test"]


def parse_xml_annotation(xml_path: str) -> dict:
    result = {
        "filename": "",
        "width": 0,
        "height": 0,
        "objects": []
    }
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_el = root.find("filename")
        result["filename"] = filename_el.text if filename_el is not None else ""

        size_el = root.find("size")
        if size_el is not None:
            w_el = size_el.find("width")
            h_el = size_el.find("height")
            result["width"] = int(w_el.text) if w_el is not None else 0
            result["height"] = int(h_el.text) if h_el is not None else 0

        for obj in root.findall("object"):
            name_el = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_el is None or bndbox is None:
                continue
            try:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
            except (TypeError, AttributeError):
                continue

            result["objects"].append({
                "name": name_el.text.strip(),
                "xmin": xmin, "ymin": ymin,
                "xmax": xmax, "ymax": ymax,
                "w": xmax - xmin,
                "h": ymax - ymin,
            })
    except ET.ParseError as e:
        print(f"  [WARN] XML parse error: {xml_path} -> {e}")

    return result


def inspect_country(country: str, data_dir: str) -> dict:
    country_dir = Path(data_dir) / country
    stats = {
        "country": country,
        "splits": {},
        "all_classes_found": set(),
        "unknown_classes": set(),
        "total_images": 0,
        "total_annotations": 0,
        "total_objects": 0,
        "background_only_images": 0,
        "missing_xml": [],
        "degenerate_bbox": 0,
    }

    for split in SPLITS:
        img_dir = country_dir / split / "images"
        xml_dir = country_dir / split / "annotations" / "xmls"

        if not img_dir.exists():
            continue

        split_stats = {
            "images": 0,
            "xmls": 0,
            "objects": 0,
            "class_counts": defaultdict(int),
            "background_only": 0,
            "degenerate_bbox": 0,
        }

        img_files = set(p.stem for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        split_stats["images"] = len(img_files)

        xml_files = list(xml_dir.glob("*.xml")) if xml_dir.exists() else []
        split_stats["xmls"] = len(xml_files)

        xml_stems = set(p.stem for p in xml_files)
        missing = img_files - xml_stems
        if missing:
            stats["missing_xml"].extend(list(missing)[:5])

        for xml_path in xml_files:
            ann = parse_xml_annotation(str(xml_path))

            if len(ann["objects"]) == 0:
                split_stats["background_only"] += 1
                stats["background_only_images"] += 1

            for obj in ann["objects"]:
                label = obj["name"]
                split_stats["objects"] += 1
                split_stats["class_counts"][label] += 1
                stats["all_classes_found"].add(label)

                if label not in EXPECTED_CLASSES:
                    stats["unknown_classes"].add(label)

                if obj["w"] <= 0 or obj["h"] <= 0:
                    split_stats["degenerate_bbox"] += 1
                    stats["degenerate_bbox"] += 1

        stats["splits"][split] = split_stats
        stats["total_images"] += split_stats["images"]
        stats["total_annotations"] += split_stats["xmls"]
        stats["total_objects"] += split_stats["objects"]

    return stats


def format_report(all_stats: list) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 1: DATASET INSPECTION REPORT -- RDD2022")
    lines.append("=" * 70)
    lines.append("")

    lines.append("[ SUMMARY TABLE ]")
    lines.append(f"{'Country':<20} {'Train Img':>10} {'Train XML':>10} {'Train Obj':>10} {'Test Img':>9} {'Background':>11} {'Degen BBox':>11}")
    lines.append("-" * 83)
    for s in all_stats:
        train = s["splits"].get("train", {})
        test = s["splits"].get("test", {})
        lines.append(
            f"{s['country']:<20}"
            f"{train.get('images', 0):>10}"
            f"{train.get('xmls', 0):>10}"
            f"{train.get('objects', 0):>10}"
            f"{test.get('images', 0):>9}"
            f"{s['background_only_images']:>11}"
            f"{s['degenerate_bbox']:>11}"
        )
    lines.append("")

    lines.append("[ CLASS INSTANCE COUNT -- TRAIN SPLIT ]")
    lines.append(f"{'Country':<20} {'D00':>8} {'D10':>8} {'D20':>8} {'D40':>8} {'OTHER':>8} {'TOTAL':>8}")
    lines.append("-" * 72)
    for s in all_stats:
        train = s["splits"].get("train", {})
        cc = train.get("class_counts", {})
        d00 = cc.get("D00", 0)
        d10 = cc.get("D10", 0)
        d20 = cc.get("D20", 0)
        d40 = cc.get("D40", 0)
        other = sum(v for k, v in cc.items() if k not in EXPECTED_CLASSES)
        total = d00 + d10 + d20 + d40 + other
        lines.append(
            f"{s['country']:<20}"
            f"{d00:>8}{d10:>8}{d20:>8}{d40:>8}{other:>8}{total:>8}"
        )
    lines.append("")

    lines.append("[ DETAIL PER COUNTRY ]")
    for s in all_stats:
        lines.append(f"\n--- {s['country']} ---")
        lines.append(f"Classes found: {sorted(s['all_classes_found'])}")
        if s["unknown_classes"]:
            lines.append(f"!!UNKNOWN classes: {s['unknown_classes']}")
        else:
            lines.append(f"Unknown classes: None")
        if s["missing_xml"]:
            lines.append(f"!!Images missing XML (sample): {s['missing_xml']}")
        lines.append(f"Background-only: {s['background_only_images']} images")
        lines.append(f"Degenerate bbox: {s['degenerate_bbox']} (w or h <= 0)")

        for split_name, sp in s["splits"].items():
            cc = sp["class_counts"]
            lines.append(f"\n  [{split_name}]")
            lines.append(f"Images: {sp['images']}")
            lines.append(f"XMLs: {sp['xmls']}")
            lines.append(f"Objects: {sp['objects']}")
            lines.append(f"Classes: { {k: cc[k] for k in sorted(cc)} }")
            lines.append(f"Background-only: {sp['background_only']}")

    lines.append("\n" + "=" * 70)
    lines.append("[ CONCLUSIONS ]")
    lines.append("=" * 70)

    all_found = set()
    for s in all_stats:
        all_found |= s["all_classes_found"]
    all_unknown = set()
    for s in all_stats:
        all_unknown |= s["unknown_classes"]

    lines.append(f"All class labels found: {sorted(all_found)}")
    if all_unknown:
        lines.append(f"!!UNEXPECTED labels found: {all_unknown}")
    else:
        lines.append("No unexpected labels -- only D00, D10, D20, D40")

    total_bg = sum(s["background_only_images"] for s in all_stats)
    lines.append(f"Total background-only images: {total_bg}")
    lines.append("")
    lines.append("Next Step -> Phase 2: EDA (notebooks/phase2_eda.ipynb)")
    lines.append("")

    return "\n".join(lines)


def main():
    print("Phase 1: Dataset Inspection")
    print(f"Data dir: {DATA_DIR}")
    print(f"Report: {REPORT_PATH}")
    print()

    all_stats = []
    for country in ALL_COUNTRIES:
        print(f"  Inspecting {country}...")
        stats = inspect_country(country, DATA_DIR)
        all_stats.append(stats)
        train = stats["splits"].get("train", {})
        print(f"    -> train: {train.get('images', 0)} imgs, "
              f"{train.get('objects', 0)} objects, "
              f"classes={sorted(stats['all_classes_found'])}")

    report = format_report(all_stats)
    print()
    print(report)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
