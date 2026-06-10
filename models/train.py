import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR,
    MLFLOW_EXPERIMENT_NAME, RANDOM_STATE, CV_FOLDS, TEST_SIZE
)

# ── Logging setup ──────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "training.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── Load processed data ────────────────────────────────────
def load_processed_data() -> tuple:
    """
    Loads preprocessed train data and splits into
    train/validation for proper model evaluation.
    Also loads test set for final prediction.
    """
    from sklearn.model_selection import train_test_split

    train = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, "train_processed.csv")
    )
    test = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, "test_processed.csv")
    )

    exclude = ["engine_id", "cycle", "RUL"]
    feature_cols = [c for c in train.columns if c not in exclude]

    X = train[feature_cols]
    y = train["RUL"]

    # Split by engine ID to prevent data leakage
    # Engines in validation set should not appear in training
    engine_ids = train["engine_id"].unique()
    np.random.seed(RANDOM_STATE)
    val_engines = np.random.choice(
        engine_ids,
        size=int(len(engine_ids) * TEST_SIZE),
        replace=False
    )

    val_mask = train["engine_id"].isin(val_engines)
    X_train = X[~val_mask]
    y_train = y[~val_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    # Test set — last cycle per engine
    test_last = test.groupby("engine_id").last().reset_index()
    X_test = test_last[feature_cols]
    y_test = test_last["RUL"]

    logger.info(
        f"Train engines: {(~val_mask).sum()} rows | "
        f"Val engines: {val_mask.sum()} rows | "
        f"Test: {X_test.shape[0]} engines"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# ── Evaluation metrics ─────────────────────────────────────
def evaluate_model(model, X_test, y_test,
                   test_df=None) -> dict:
    """
    Evaluates model on test set.
    If test_df provided, evaluates on last cycle per engine only
    which is the standard NASA dataset evaluation approach.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "y_pred": y_pred
    }


# ── Train a single model with MLflow tracking ──────────────
def train_model(model, model_name: str,
                X_train, y_train,
                X_val, y_val,
                params: dict) -> dict:
    """
    Trains a model, evaluates on validation set,
    and logs everything to MLflow.
    """
    with mlflow.start_run(run_name=model_name):

        mlflow.log_params(params)

        # Cross-validation on training data
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error"
        )
        cv_rmse = -cv_scores.mean()
        cv_std = cv_scores.std()

        # Train on full training set
        model.fit(X_train, y_train)

        # Evaluate on validation set
        metrics = evaluate_model(model, X_val, y_val)

        mlflow.log_metric("cv_rmse", round(cv_rmse, 4))
        mlflow.log_metric("cv_std", round(cv_std, 4))
        mlflow.log_metric("val_rmse", metrics["rmse"])
        mlflow.log_metric("val_mae", metrics["mae"])
        mlflow.log_metric("val_r2", metrics["r2"])

        mlflow.sklearn.log_model(model, model_name)

        logger.info(
            f"{model_name} | CV RMSE: {cv_rmse:.4f} "
            f"| Val RMSE: {metrics['rmse']} "
            f"| Val R2: {metrics['r2']}"
        )

        print(f"\n{model_name}")
        print(f"  CV RMSE    : {cv_rmse:.4f} (+/- {cv_std:.4f})")
        print(f"  Val RMSE   : {metrics['rmse']}")
        print(f"  Val MAE    : {metrics['mae']}")
        print(f"  Val R2     : {metrics['r2']}")

        return {
            "model": model,
            "model_name": model_name,
            "cv_rmse": cv_rmse,
            "val_rmse": metrics["rmse"],
            "val_r2": metrics["r2"],
            "y_pred": metrics["y_pred"]
        }


# ── Train all models ───────────────────────────────────────
def train_all_models(X_train, y_train,
                     X_val, y_val,
                     X_test, y_test) -> list:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    models = [
        {
            "model": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "name": "RandomForest",
            "params": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5
            }
        },
        {
            "model": XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0
            ),
            "name": "XGBoost",
            "params": {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.01,
                "subsample": 0.8
            }
        },
        {
            "model": LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            ),
            "name": "LightGBM",
            "params": {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.01,
                "subsample": 0.8
            }
        }
    ]

    results = []
    for m in models:
        result = train_model(
            m["model"], m["name"],
            X_train, y_train,
            X_val, y_val,
            m["params"]
        )
        results.append(result)
    return results


# ── Save best model ────────────────────────────────────────
def save_best_model(results: list) -> dict:
    best = min(results, key=lambda x: x["val_rmse"])
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best["model"], f)
    logger.info(
        f"Best model: {best['model_name']} "
        f"| Val RMSE: {best['val_rmse']} "
        f"| Val R2: {best['val_r2']}"
    )
    print(f"\nBest model: {best['model_name']}")
    print(f"Saved to: {model_path}")
    return best


# ── Main training pipeline ─────────────────────────────────
def run_training() -> dict:
    logger.info("Starting training pipeline")

    X_train, y_train, X_val, y_val, \
        X_test, y_test, feature_cols = load_processed_data()

    print("=" * 50)
    print("Training 3 models with MLflow tracking")
    print("=" * 50)

    results = train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    for r in results:
        print(
            f"{r['model_name']:<15} "
            f"Val RMSE: {r['val_rmse']:<10} "
            f"Val R2: {r['val_r2']}"
        )

    best = save_best_model(results)
    logger.info("Training pipeline complete")
    return best