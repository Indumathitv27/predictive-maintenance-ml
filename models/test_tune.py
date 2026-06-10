import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tune import run_tuning

print("Starting Optuna hyperparameter tuning...")
print("Running 50 trials — takes about 5 minutes")
print()

result = run_tuning(n_trials=50)

print()
print("=" * 50)
print("TUNING COMPLETE")
print("=" * 50)
print(f"Best Val RMSE : {result['val_rmse']}")
print(f"Best Val R2   : {result['val_r2']}")
print(f"Trials run    : {result['n_trials']}")
print("Tuned model saved and logged to MLflow")