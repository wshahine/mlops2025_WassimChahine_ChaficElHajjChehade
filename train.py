import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
 
# --- ROBUST PATH CONFIGURATION ---
# 1. Get the folder where THIS script (train.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))
 
# 2. Go up one level to the project root (mlops.../mlops...)
project_root = os.path.dirname(script_dir)
 
# 3. Define paths relative to that root
# Reads from: src/mlproject/data/featured_data.parquet
DATA_PATH = os.path.join(project_root, "src", "mlproject", "data", "featured_data.parquet")
 
# Saves to: data/model.pkl (The main data folder sibling to scripts)
MODEL_PATH = os.path.join(project_root, "data", "model.pkl")
# ---------------------------------
 
def train_model():
    print(f"Script location: {script_dir}")
    print(f"Loading feature data from: {DATA_PATH}")
 
    if not os.path.exists(DATA_PATH):
        print("❌ ERROR: File not found!")
        print(f"Python is looking here: {DATA_PATH}")
        return
 
    df = pd.read_parquet(DATA_PATH)
 
    X = df.drop(columns=['trip_duration'])
    y = df['trip_duration']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    # 1. Train two models
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
 
    print("Training Random Forest (Small)...")
    rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
 
    # 2. Select best model
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
 
    lr_rmse = mean_squared_error(y_test, lr_pred, squared=False)
    rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
 
    print(f"Linear Regression RMSE: {lr_rmse}")
    print(f"Random Forest RMSE: {rf_rmse}")
 
    best_model = rf if rf_rmse < lr_rmse else lr
    print("Saving best model...")
 
    # Ensure the directory for the model exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
   
    joblib.dump(best_model, MODEL_PATH)
    print(f"✅ Model saved successfully to: {MODEL_PATH}")
 
if __name__ == "__main__":
    train_model()
