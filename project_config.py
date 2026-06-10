import os
from dotenv import load_dotenv

load_dotenv()

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

# Dataset files
TRAIN_FILE = os.path.join(RAW_DATA_DIR, "train_FD001.txt")
TEST_FILE = os.path.join(RAW_DATA_DIR, "test_FD001.txt")
RUL_FILE = os.path.join(RAW_DATA_DIR, "RUL_FD001.txt")

# Column names
COLUMN_NAMES = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

# Feature engineering settings
WINDOW_SIZE = 30        # rolling window for moving averages
RUL_CLIP = 130
WINDOW_SIZE = 15         # cap RUL at 125 cycles — engines degrade non-linearly

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# MLflow settings
MLFLOW_EXPERIMENT_NAME = "predictive-maintenance"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000