import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.paths import DATA_PROC, MANIFEST_PATH, YOLO_YAML

CLASS_NAMES = ["D00", "D10", "D20", "D40"]


def backup(path: Path):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_suffix(f".bak_{ts}{path.suffix}")
    shutil.copy2(path, backup_path)
    print(f"Backup: {backup_path.name}")
    return backup_path


def fix_manifest(manifest_path: Path, data_proc: Path):
    print(f"\n[Fix 1] manifest.json -> {manifest_path}")

    if not manifest_path.exists():
        print("ERROR: manifest.json not found -- skipping")
        return

    with open(manifest_path, "r") as f:
        records = json.load(f)

    print(f"Total records: {len(records)}")

    already_relative = 0
    converted = 0
    failed = 0

    for rec in records:
        p = Path(rec["image_path"])

        if not p.is_absolute():
            already_relative += 1
            continue

        try:
            rel = p.relative_to(data_proc)
            rec["image_path"] = str(rel)
            converted += 1
        except ValueError:
            parts = p.parts
            if len(parts) >= 3:
                rec["image_path"] = str(Path(*parts[-3:]))
                converted += 1
                print(f"[WARN] Fallback used for: {p.name}")
            else:
                failed += 1
                print(f"[ERROR] Cannot convert: {rec['image_path']}")

    print(f"Already relative: {already_relative}")
    print(f"Converted: {converted}")
    print(f"Failed: {failed}")

    if converted > 0:
        backup(manifest_path)
        with open(manifest_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"manifest.json updated ({converted} paths converted)")
    else:
        print("No changes needed.")


def fix_yolo_yaml(yaml_path: Path, data_proc: Path):
    print(f"\n[Fix 2] dataset.yaml -> {yaml_path}")

    if not yaml_path.exists():
        print("  dataset.yaml not found -- creating fresh one")

    cfg = {
        "path": str(data_proc.resolve()),
        "train": str((data_proc / "images" / "train").resolve()),
        "val": str((data_proc / "images" / "val").resolve()),
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }

    if yaml_path.exists():
        backup(yaml_path)

    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"dataset.yaml written:")
    print(f"path: {cfg['path']}")
    print(f"train: {cfg['train']}")
    print(f"val: {cfg['val']}")
    print(f"nc: {cfg['nc']}  names: {cfg['names']}")


def main():
    print("fix_data_configs.py -- migrating existing processed data configs")
    print(f"DATA_PROC = {DATA_PROC}")
    print()

    if not DATA_PROC.exists():
        print(f"ERROR: DATA_PROC does not exist: {DATA_PROC}")
        print("Make sure you ran: cp -r <project>/data/processed/japan_only ~/data/")
        raise SystemExit(1)

    fix_manifest(MANIFEST_PATH, DATA_PROC)
    fix_yolo_yaml(YOLO_YAML, DATA_PROC)


if __name__ == "__main__":
    main()
