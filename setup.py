import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup():
    """
    Runs on container startup.
    Downloads data and trains model if not already present.
    """
    from project_config import MODELS_DIR, PROCESSED_DATA_DIR

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Model already exists — skipping training")
        return

    print("Model not found — running training pipeline...")

    # Check if processed data exists
    train_path = os.path.join(
        PROCESSED_DATA_DIR, "train_processed.csv"
    )

    if not os.path.exists(train_path):
        print("Processed data not found — running preprocessing...")
        from features.preprocess import run_preprocessing
        run_preprocessing()

    print("Training model...")
    from models.train import run_training
    run_training()

    print("Running Optuna tuning...")
    from models.tune import run_tuning
    run_tuning(n_trials=20)

    print("Setup complete — model ready")

if __name__ == "__main__":
    setup()