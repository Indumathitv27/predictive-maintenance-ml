import os
import pandas as pd
import numpy as np
import logging
from project_config import (
    TRAIN_FILE, TEST_FILE, RUL_FILE,
    COLUMN_NAMES, PROCESSED_DATA_DIR,
    WINDOW_SIZE, RUL_CLIP, LOGS_DIR
)

# ── Logging setup ──────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── Load raw data ──────────────────────────────────────────
def load_data() -> tuple:
    """
    Loads train, test, and RUL files.
    Returns three dataframes.
    """
    train = pd.read_csv(TRAIN_FILE, sep="\s+", header=None,
                    names=COLUMN_NAMES, engine="python")
    test = pd.read_csv(TEST_FILE, sep="\s+", header=None,
                   names=COLUMN_NAMES, engine="python")
    rul = pd.read_csv(RUL_FILE, sep="\s+", header=None,
                  engine="python")
    rul = rul.iloc[:, 0].to_frame(name="RUL")

    # Drop empty columns caused by trailing spaces in NASA file
    train = train.dropna(axis=1, how="all")
    test = test.dropna(axis=1, how="all")
    rul = rul.dropna(axis=1, how="all")

    logger.info(f"Loaded train: {train.shape}, test: {test.shape}, rul: {rul.shape}")
    return train, test, rul


# ── Compute RUL for training data ─────────────────────────
def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Remaining Useful Life for each row in training data.
    RUL = max cycle for that engine - current cycle.
    Clips RUL at RUL_CLIP to focus on degradation phase.
    """
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=RUL_CLIP)
    df = df.drop(columns=["max_cycle"])
    logger.info(f"RUL computed and clipped at {RUL_CLIP}")
    return df


# ── Drop low-variance sensors ──────────────────────────────
def drop_low_variance_sensors(df: pd.DataFrame,
                               threshold: float = 0.01) -> pd.DataFrame:
    """
    Removes sensor columns with near-zero variance.
    These sensors don't change and carry no predictive signal.
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor")]
    variances = df[sensor_cols].var()
    low_var = variances[variances < threshold].index.tolist()
    df = df.drop(columns=low_var)
    logger.info(f"Dropped low-variance sensors: {low_var}")
    return df


# ── Feature engineering ────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-series features per engine:
    - Rolling mean (last WINDOW_SIZE cycles)
    - Rolling std (captures variability/noise)
    - Lag features (previous cycle sensor readings)
    - Cycle normalized per engine (0 to 1 degradation progress)
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor")]

    # Sort by engine and cycle
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    for col in sensor_cols:
        # Rolling mean
        df[f"{col}_roll_mean"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).mean())
        )
        # Rolling std
        df[f"{col}_roll_std"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).std().fillna(0))
        )
        # Lag 1 feature
        df[f"{col}_lag1"] = (df.groupby("engine_id")[col]
                            .transform(lambda x: x.shift(1).bfill()))

    # Normalized cycle — degradation progress from 0 to 1
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_cycles

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


# ── Normalize sensor readings ──────────────────────────────
def normalize_features(train: pd.DataFrame,
                        test: pd.DataFrame) -> tuple:
    """
    Min-max normalizes all numeric feature columns.
    Fit on train only, apply to both train and test.
    This prevents data leakage.
    """
    from sklearn.preprocessing import MinMaxScaler

    exclude = ["engine_id", "cycle", "RUL"]
    feature_cols = [c for c in train.columns if c not in exclude]

    train[feature_cols] = train[feature_cols].replace(
    [float('inf'), float('-inf')], float('nan'))

    train[feature_cols] = train[feature_cols].fillna(
    train[feature_cols].median())

    test[feature_cols] = test[feature_cols].replace(
    [float('inf'), float('-inf')], float('nan'))

    test[feature_cols] = test[feature_cols].fillna(
    test[feature_cols].median()
)

    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    logger.info("Normalization complete — fit on train, applied to test")
    return train, test, scaler, feature_cols


# ── Full preprocessing pipeline ────────────────────────────
def run_preprocessing() -> tuple:
    """
    Runs the complete preprocessing pipeline:
    Load -> RUL -> Drop low variance -> Feature engineering -> Normalize
    Returns train and test dataframes ready for modeling.
    """
    logger.info("Starting preprocessing pipeline")

    # Load
    train, test, rul = load_data()

    # Compute RUL for training data
    train = compute_rul(train)

    # Compute RUL for test data using provided RUL file
    rul_values = rul["RUL"].values
    engine_ids = test["engine_id"].unique()
    rul_map = dict(zip(engine_ids, rul_values))
    test["RUL"] = test["engine_id"].map(rul_map)
    test["RUL"] = test["RUL"].clip(upper=RUL_CLIP)

    # Drop low variance sensors
    train = drop_low_variance_sensors(train)
    sensor_cols_kept = [c for c in train.columns if c.startswith("sensor")]
    test = test[["engine_id", "cycle"] +
                [c for c in test.columns
                 if c in sensor_cols_kept or
                 c in ["op_setting_1", "op_setting_2",
                        "op_setting_3", "RUL"]]]

    # Feature engineering
    train = engineer_features(train)
    test = engineer_features(test)

    # Normalize
    train_rul = train["RUL"].copy()
    test_rul = test["RUL"].copy()

    train, test, scaler, feature_cols = normalize_features(train, test)

    # Restore RUL after normalization
    train["RUL"] = train_rul
    test["RUL"] = test_rul

    # Save processed data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train.to_csv(
        os.path.join(PROCESSED_DATA_DIR, "train_processed.csv"),
        index=False
    )
    test.to_csv(
        os.path.join(PROCESSED_DATA_DIR, "test_processed.csv"),
        index=False
    )

    logger.info("Preprocessing complete — files saved to data/processed/")
    return train, test, scaler, feature_cols