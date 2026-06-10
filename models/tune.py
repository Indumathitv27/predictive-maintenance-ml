import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR,
    MLFLOW_EXPERIMENT_NAME, RANDOM_STATE, CV_FOLDS
)

# ── Logging setup ──────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "tuning.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Load data ──────────────────────────────────────────────
def load_data_for_tuning() -> tuple:
    """
    Loads training data split by engine ID.
    Returns train and validation sets.
    """
    train = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, "train_processed.csv")
    )

    exclude = ["engine_id", "cycle", "RUL"]
    feature_cols = [c for c in train.columns if c not in exclude]

    engine_ids = train["engine_id"].unique()
    np.random.seed(RANDOM_STATE)
    val_engines = np.random.choice(
        engine_ids,
        size=int(len(engine_ids) * 0.2),
        replace=False
    )

    val_mask = train["engine_id"].isin(val_engines)
    X_train = train[~val_mask][feature_cols]
    y_train = train[~val_mask]["RUL"]
    X_val = train[val_mask][feature_cols]
    y_val = train[val_mask]["RUL"]

    return X_train, y_train, X_val, y_val, feature_cols


# ── Optuna objective function ──────────────────────────────
def objective(trial, X_train, y_train, X_val, y_val) -> float:
    """
    Optuna objective function.
    Each trial suggests a set of hyperparameters,
    trains LightGBM, and returns validation RMSE.
    Optuna minimizes this value across trials.
    """
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 1000
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 12
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.1, log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.5, 1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 5, 50
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 1.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-8, 1.0, log=True
        ),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    }

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse


# ── Run Optuna study ───────────────────────────────────────
def run_tuning(n_trials: int = 50) -> dict:
    """
    Runs Optuna hyperparameter search for LightGBM.
    Logs best params and model to MLflow.
    Returns best model and params.
    """
    logger.info(f"Starting Optuna tuning — {n_trials} trials")

    X_train, y_train, X_val, y_val, feature_cols = \
        load_data_for_tuning()

    print("=" * 50)
    print(f"Optuna hyperparameter tuning — {n_trials} trials")
    print("Optimizing LightGBM for minimum RMSE")
    print("=" * 50)

    # Create Optuna study — minimize RMSE
    study = optuna.create_study(
        direction="minimize",
        study_name="lgbm-predictive-maintenance"
    )

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_val, y_val
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best_params = study.best_params
    best_rmse = study.best_value

    print(f"\nBest RMSE found : {best_rmse:.4f}")
    print(f"Best params     :")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Train final model with best params
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1

    final_model = LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_val)
    final_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    final_r2 = r2_score(y_val, y_pred)

    print(f"\nFinal tuned model:")
    print(f"  Val RMSE : {final_rmse:.4f}")
    print(f"  Val R2   : {final_r2:.4f}")

    # Log to MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="LightGBM_Optuna_Tuned"):
        mlflow.log_params(best_params)
        mlflow.log_metric("val_rmse", round(final_rmse, 4))
        mlflow.log_metric("val_r2", round(final_r2, 4))
        mlflow.log_metric("n_trials", n_trials)
        mlflow.sklearn.log_model(final_model, "LightGBM_Tuned")

    # Save tuned model
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    # Save best params
    params_path = os.path.join(MODELS_DIR, "best_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(best_params, f)

    logger.info(
        f"Tuning complete | Best RMSE: {final_rmse:.4f} "
        f"| R2: {final_r2:.4f}"
    )

    print(f"\nTuned model saved to {model_path}")

    return {
        "model": final_model,
        "best_params": best_params,
        "val_rmse": round(final_rmse, 4),
        "val_r2": round(final_r2, 4),
        "n_trials": n_trials,
        "study": study
    }