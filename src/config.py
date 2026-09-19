"""
Configuration and constants for Rail Corrugation Subsystem (PS3)
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "Train_Labels.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Physical & Sensor Constants
SAMPLING_RATE = 10000          # 10,000 Hz
DURATION_SEC = 1.0             # 1.0 second per file
NUM_SAMPLES = 10000            # 10,000 rows per file
WHEEL_DIAMETER = 0.85          # 0.85 meters
WHEEL_CIRCUMFERENCE = 3.141592653589793 * WHEEL_DIAMETER  # ~2.67035 m
SPEED_SENSOR_TEETH = 90        # 90 teeth on the pulse wheel

# Train Topology
NUM_CARS = 8                   # Cars 1 through 8
AXLE_POSITIONS_PER_CAR = 8     # Positions 1 through 8
TOTAL_AXLE_BOXES = 64          # 8 cars * 8 positions

# Side Mapping:
# Positions 1, 3, 5, 7 -> Side I (Left rail)
# Positions 2, 4, 6, 8 -> Side II (Right rail)
SIDE_I_POSITIONS = [1, 3, 5, 7]
SIDE_II_POSITIONS = [2, 4, 6, 8]

# Target Classes
CLASSES = ["Normal", "Side I", "Side II"]
CLASS_TO_IDX = {"Normal": 0, "Side I": 1, "Side II": 2}
IDX_TO_CLASS = {0: "Normal", 1: "Side I", 2: "Side II"}

# Characteristic Corrugation Frequency & Wavelength Ranges
# Typical corrugation wavelengths: 20 mm to 300 mm (0.02 m to 0.30 m)
# For v = 15 m/s (~54 km/h), f = v / lambda -> f in [50 Hz, 750 Hz]
FREQ_BANDS = {
    "sub_corrugation": (10, 80),      # Low-freq suspension / track bounce
    "short_pitch": (80, 250),        # 60 - 200 mm corrugation
    "resonant_wear": (250, 600),     # 25 - 60 mm corrugation
    "roaring_rail": (600, 1200),     # High-frequency acoustic squeal
    "high_freq_noise": (1200, 3000)  # Structural / gear noise
}

# Physical Wavelength Bins (in meters): lambda = v / f
WAVELENGTH_BINS = [
    ("micro_pitch", (0.02, 0.05)),   # 20 - 50 mm
    ("short_pitch", (0.05, 0.10)),   # 50 - 100 mm
    ("medium_pitch", (0.10, 0.20)),  # 100 - 200 mm
    ("long_pitch", (0.20, 0.40)),    # 200 - 400 mm
]
