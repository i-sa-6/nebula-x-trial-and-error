"""Shared constants for the ACV refrigerant-leak fault-localization pipeline.

Scope note: this pipeline covers only the 8-parameter schema shared by Model A
(cases 01/02/03) and Model C (cases 05/06). A sixth labeled file, acv_case_04.xlsx
(Model B, ~60 parameters), is deliberately excluded -- see README.md.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
LABELS_FILE = TRAIN_DIR / "labels.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

CAR_NUMBERS = [f"{i:02d}" for i in range(1, 9)]  # '01'..'08', N=8 cars/train

# Canonical (post-normalization) parameter names. "Outside Temperature" is the
# normalized name for the two train-model-specific aliases below.
CATEGORICAL_PARAMS = [
    "ACV Running Mode",
    "ACV Setting Mode",
    "Load Halved",
    "ACV Information Valid",
]
NUMERIC_PARAMS = [
    "ACV Control Temperature (Cooling)",
    "ACV Control Temperature (Heating)",
    "Indoor Average Temperature",
    "Outside Temperature",
]
BASE_PARAMS = CATEGORICAL_PARAMS + NUMERIC_PARAMS  # the 8 parameters per car

# Source-file column-name aliases that both normalize to "Outside Temperature".
TEMPERATURE_ALIASES = ["Outdoor Average Temperature", "Outside Temperature Sensor Reading"]

# Cars whose "Outside Temperature" sensor is physically present in Model C
# (cases 05/06); the other 6 cars report the literal string "Invalid" 100% of
# the time. See README.md "Data-reality deviations" for the finding and the
# chosen fix (end-car broadcast).
END_CARS = ["01", "08"]

METADATA_COLS = ["Car model", "Train number", "Time"]

RARE_CATEGORY_THRESHOLD = 0.01  # merge categories under 1% of non-missing rows
DROP_VS_FFILL_THRESHOLD = 0.05  # per-file threshold from the task spec

RARE_CATEGORY_BUCKET = "Other"

ENGINEERED_FEATURE = "cross_car_temp_deviation"

RANDOM_SEED = 42

N_CARS = 8
