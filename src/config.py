"""Central configuration: paths, selected classes, seed and thresholds.

Everything lives inside the project folder (on C:), so the project is self-contained.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_REPO = DATA_DIR / "GroceryStoreDataset"      # sparse git clone of the official repo
DATASET_ROOT = DATASET_REPO / "dataset"              # split files list image paths relative to this folder
SPLIT_FILES = {
    "train": DATASET_ROOT / "train.txt",
    "val": DATASET_ROOT / "val.txt",
    "test": DATASET_ROOT / "test.txt",
}
CLASSES_CSV = DATASET_ROOT / "classes.csv"
MANIFEST_CSV = DATA_DIR / "manifest.csv"             # one row per selected image (written by prepare_data)
SPARSE_PATHS_TXT = DATA_DIR / "sparse_paths.txt"     # folders to sparse-checkout (written by prepare_data)
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"                    # test metrics and predictions
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
TRAINING_DIR = RESULTS_DIR / "training"                  # history, curves, run config
ERROR_DIR = RESULTS_DIR / "error_analysis"               # misclassifications and examples
DATA_AUDIT_DIR = RESULTS_DIR / "data_audit"              # counts, integrity audit
DUPLICATES_DIR = DATA_AUDIT_DIR / "duplicates"          # duplicate-analysis reports
LOG_DIR = RESULTS_DIR / "logs"                           # runtime logs (not committed)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"

SPLITS = ["train", "val", "test"]

# --- Selected classes ----------------------------------------------------------
# Dataset folder names, grouped by the visual-similarity question each group tests.
# The flattened order defines the model's label indices 0..11.
CLASS_GROUPS = {
    "milk": [            # near-identical cartons, differ by fat / eco label (and brand)
        "Arla-Standard-Milk",
        "Arla-Medium-Fat-Milk",
        "Arla-Ecological-Medium-Fat-Milk",
        "Garant-Ecological-Medium-Fat-Milk",
    ],
    "oatly": [           # same brand and design language, different product
        "Oatly-Oat-Milk",
        "Oatly-Natural-Oatghurt",
    ],
    "juice": [           # same brand / different flavour and same flavour / different brand
        "Bravo-Orange-Juice",
        "Bravo-Apple-Juice",
        "God-Morgon-Orange-Juice",
    ],
    "yoghurt": [         # same flavour across brands, similar pots
        "Valio-Vanilla-Yoghurt",
        "Arla-Mild-Vanilla-Yoghurt",
        "Yoggi-Strawberry-Yoghurt",
    ],
}
CLASSES = [name for group in CLASS_GROUPS.values() for name in group]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
CLASS_TO_GROUP = {name: g for g, names in CLASS_GROUPS.items() for name in names}

# Minimum images per class and split agreed in the plan (checked by prepare_data).
MIN_IMAGES = {"train": 23, "val": 4, "test": 22}

# --- Reproducibility / training defaults ----------------------------------------
SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)   # normalisation used by the pretrained weights
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 16
NUM_WORKERS = 2          # use 0 if RAM becomes tight

# --- Duplicate check thresholds (report-only) ----------------------------------
PHASH_REPORT_MAX = 10    # Hamming distance (of 64 bits) to include a pair in the report
PHASH_FLAG_MAX = 6       # ... to mark it as a likely near-duplicate
COSINE_REPORT_MIN = 0.90 # pretrained-feature cosine similarity to include a pair
COSINE_FLAG_MIN = 0.95   # ... to mark it as a likely same-shot pair
