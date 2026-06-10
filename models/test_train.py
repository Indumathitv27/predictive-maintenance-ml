import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train import run_training

print("Starting model training pipeline...")
print("This will take 3-5 minutes — training 3 models with cross-validation")
print()

best = run_training()

print()
print("=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)
print(f"Best model      : {best['model_name']}")
print(f"Best RMSE       : {best['val_rmse']}")
print(f"Best R2 Score   : {best['val_r2']}")
print()
print("All experiments tracked in MLflow")
print("Best model saved to models/best_model.pkl")