import pandas as pd
import numpy as np  # <--- Added this import
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
from pathlib import Path

# --- ROBUST PATH CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "featured_data.parquet"
MODEL_PATH = PROJECT_ROOT / "data" / "model.pkl"

def train_model():
    print(f" Script started...")
    print(f"   Looking for data at: {DATA_PATH}")

    if not DATA_PATH.exists():
        print(" ERROR: File not found!")
        print(f"   Python is looking here: {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    print(f" Data loaded. Shape: {df.shape}")

    # Prepare X and y
    X = df.drop(columns=['trip_duration'])
    y = df['trip_duration']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Train two models
    print("   Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print("   Training Random Forest (Small)...")
    rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    # 2. Evaluate
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    # --- FIX IS HERE: Calculate MSE first, then take Square Root ---
    lr_mse = mean_squared_error(y_test, lr_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)

    lr_rmse = np.sqrt(lr_mse)
    rf_rmse = np.sqrt(rf_mse)

    print(f"   Linear Regression RMSE: {lr_rmse:.4f}")
    print(f"   Random Forest RMSE:     {rf_rmse:.4f}")

    # 3. Select and Save best model
    if rf_rmse < lr_rmse:
        best_model = rf
        best_name = "RandomForest"
    else:
        best_model = lr
        best_name = "LinearRegression"

    print(f" Best model is {best_name}")

    # Ensure the directory exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_model, MODEL_PATH)
    print(f" Model saved successfully to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()