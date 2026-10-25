from pathlib import Path

# Project root: config/ lives one level below project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "RDD2022"

# Processed dataset
DATA_PROC = Path.home() / "data" / "japan_only"

# Outputs (always inside the project tree)
OUTPUT_DIR  = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RUNS_DIR    = OUTPUT_DIR / "runs"

# Dataset configs
MANIFEST_PATH = DATA_PROC / "manifest.json"
YOLO_YAML = DATA_PROC / "dataset.yaml"
