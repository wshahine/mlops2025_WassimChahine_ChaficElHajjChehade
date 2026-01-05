import pandas as pd
import joblib
import os
from datetime import datetime
from pathlib import Path

# --- ROBUST PATH CONFIGURATION ---
# 1. Get the project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Define correct paths relative to root
MODEL_PATH = PROJECT_ROOT / "data" / "model.pkl"

# Corrected Input Path: Points to 'data/featured_data.parquet' in the root
INPUT_PATH = PROJECT_ROOT / "data" / "featured_data.parquet"

# Output Directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
# ---------------------------------

def predict():
    print(f" Starting Batch Inference...")
    print(f"   Loading model from: {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        print(" ERROR: Model not found!")
        print("   Run 'uv run scripts/train.py' first to generate the model.")
        return

    model = joblib.load(MODEL_PATH)
    
    print(f"   Loading data from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        print(f" ERROR: Input data not found at {INPUT_PATH}")
        return

    df = pd.read_parquet(INPUT_PATH)

    # Prepare features (Remove target 'trip_duration' if it exists)
    # This ensures we are simulating real inference where we don't know the answer
    X = df.drop(columns=['trip_duration'], errors='ignore') 

    print(f"   Running predictions on {len(X)} rows...")
    predictions = model.predict(X)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a filename with today's date
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{date_str}_predictions.csv"

    # Save predictions
    pd.DataFrame({'prediction': predictions}).to_csv(output_file, index=False)
    
    print(f" Predictions saved to: {output_file}")

if __name__ == "__main__":
    predict()