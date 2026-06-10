import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import MODELS_DIR, LOGS_DIR, PROCESSED_DATA_DIR

# ── Logging setup ──────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "api.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── FastAPI app ────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="Turbofan engine RUL prediction using LightGBM — Ford Motor Company use case",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load model and feature columns at startup ──────────────
print("Loading predictive maintenance model...")
try:
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    print("Model loaded successfully.")
    logger.info("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None
    feature_cols = None
    scaler = None


# ── Request schema ─────────────────────────────────────────
class SensorReading(BaseModel):
    engine_id: int = Field(..., description="Engine ID")
    cycle: int = Field(..., description="Current cycle number")
    op_setting_1: float = Field(..., description="Operational setting 1")
    op_setting_2: float = Field(..., description="Operational setting 2")
    op_setting_3: float = Field(..., description="Operational setting 3")
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class PredictionResponse(BaseModel):
    engine_id: int
    cycle: int
    predicted_rul: float
    risk_level: str
    recommendation: str
    model_version: str


# ── Helper — build feature vector ─────────────────────────
def build_feature_vector(reading: SensorReading) -> pd.DataFrame:
    """
    Converts a sensor reading into a feature vector
    the model can predict from.
    Uses median values for rolling/lag features
    since we only have one cycle of data.
    """
    # Base features from reading
    base = {
        "op_setting_1": reading.op_setting_1,
        "op_setting_2": reading.op_setting_2,
        "op_setting_3": reading.op_setting_3,
        "sensor_2": reading.sensor_2,
        "sensor_3": reading.sensor_3,
        "sensor_4": reading.sensor_4,
        "sensor_6": reading.sensor_6,
        "sensor_7": reading.sensor_7,
        "sensor_8": reading.sensor_8,
        "sensor_9": reading.sensor_9,
        "sensor_11": reading.sensor_11,
        "sensor_12": reading.sensor_12,
        "sensor_13": reading.sensor_13,
        "sensor_14": reading.sensor_14,
        "sensor_15": reading.sensor_15,
        "sensor_17": reading.sensor_17,
        "sensor_20": reading.sensor_20,
        "sensor_21": reading.sensor_21,
        "cycle_norm": min(reading.cycle / 300.0, 1.0)
    }

    # Build full feature vector matching training features
    row = {}
    for col in feature_cols:
        if col in base:
            row[col] = base[col]
        else:
            # For rolling/lag features use the base sensor value
            for sensor in base:
                if col.startswith(sensor):
                    row[col] = base[sensor]
                    break
            else:
                row[col] = 0.0

    df = pd.DataFrame([row])
    df = df[feature_cols]

    # Scale using fitted scaler
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_cols
    )
    return df_scaled


# ── Risk classification ────────────────────────────────────
def classify_risk(rul: float) -> tuple:
    """
    Classifies engine risk based on predicted RUL.
    Returns risk level and maintenance recommendation.
    """
    if rul <= 20:
        return "CRITICAL", "Immediate maintenance required — engine failure imminent"
    elif rul <= 50:
        return "HIGH", "Schedule maintenance within next 2 weeks"
    elif rul <= 80:
        return "MEDIUM", "Plan maintenance within next month"
    else:
        return "LOW", "Engine healthy — continue normal operations"


# ── Health check ───────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "Predictive Maintenance API",
        "model": "LightGBM — Optuna tuned",
        "version": "1.0.0"
    }


# ── Predict RUL endpoint ───────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict_rul(reading: SensorReading):
    """
    Accepts sensor readings for one engine cycle
    and returns predicted Remaining Useful Life (RUL)
    with risk classification and maintenance recommendation.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    logger.info(
        f"Prediction request | engine_id={reading.engine_id} "
        f"| cycle={reading.cycle}"
    )

    try:
        # Build feature vector
        features = build_feature_vector(reading)

        # Predict RUL
        rul = float(model.predict(features)[0])
        rul = max(0.0, round(rul, 2))

        # Classify risk
        risk_level, recommendation = classify_risk(rul)

        logger.info(
            f"Prediction complete | engine_id={reading.engine_id} "
            f"| RUL={rul} | risk={risk_level}"
        )

        return PredictionResponse(
            engine_id=reading.engine_id,
            cycle=reading.cycle,
            predicted_rul=rul,
            risk_level=risk_level,
            recommendation=recommendation,
            model_version="LightGBM-Optuna-v1.0"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ── Batch predict endpoint ─────────────────────────────────
@app.post("/predict/batch")
def predict_batch(readings: List[SensorReading]):
    """
    Accepts multiple sensor readings and returns
    RUL predictions for all engines at once.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    results = []
    for reading in readings:
        try:
            features = build_feature_vector(reading)
            rul = float(model.predict(features)[0])
            rul = max(0.0, round(rul, 2))
            risk_level, recommendation = classify_risk(rul)
            results.append({
                "engine_id": reading.engine_id,
                "cycle": reading.cycle,
                "predicted_rul": rul,
                "risk_level": risk_level,
                "recommendation": recommendation
            })
        except Exception as e:
            results.append({
                "engine_id": reading.engine_id,
                "error": str(e)
            })

    logger.info(f"Batch prediction complete | {len(results)} engines")
    return {"predictions": results, "count": len(results)}