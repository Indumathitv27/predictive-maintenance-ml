import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.preprocess import run_preprocessing

print("=" * 50)
print("TEST: Running preprocessing pipeline")
print("=" * 50)

train, test, scaler, feature_cols = run_preprocessing()

print(f"Train shape      : {train.shape}")
print(f"Test shape       : {test.shape}")
print(f"Features created : {len(feature_cols)}")
print(f"Train RUL range  : {train['RUL'].min():.0f} to {train['RUL'].max():.0f}")
print(f"Test RUL range   : {test['RUL'].min():.0f} to {test['RUL'].max():.0f}")
print()
print("Sample feature columns:")
for col in feature_cols[:5]:
    print(f"  {col}")
print(f"  ... and {len(feature_cols)-5} more")
print()
print("Processed files saved to data/processed/")