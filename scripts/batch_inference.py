import pandas as pd
import joblib
import os
from datetime import datetime
 
# --- ROBUST PATH CONFIGURATION ---
# 1. Get the folder where THIS script (batch_inference.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))
 
# 2. Go up one level to the project root
project_root = os.path.dirname(script_dir)
 
# 3. Define paths relative to the root
MODEL_PATH = os.path.join(project_root, "data", "model.pkl")
# Using the same featured data for now (or change this to your new inference file)
INPUT_PATH = os.path.join(project_root, "src", "mlproject", "data", "featured_data.parquet")
OUTPUT_DIR = os.path.join(project_root, "data", "predictions")
# ---------------------------------
 
def predict():
    print(f"Loading model from: {MODEL_PATH}")
   
    if not os.path.exists(MODEL_PATH):
        print("❌ ERROR: Model not found!")
        print("Run train.py first to generate the model.")
        return
 
    model = joblib.load(MODEL_PATH)
   
    print(f"Loading data from: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print(f"❌ ERROR: Input data not found at {INPUT_PATH}")
        return
 
    df = pd.read_parquet(INPUT_PATH)
 
    # Prepare features (Remove target if it exists, to simulate real inference)
    X = df.drop(columns=['trip_duration'], errors='ignore') # Ensure no target
 
    print("Running predictions...")
    predictions = model.predict(X)
 
    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
   
    # Create a filename with today's date
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"{date_str}_predictions.csv")
 
    # Save just the predictions (or you can include X to see inputs too)
    pd.DataFrame({'prediction': predictions}).to_csv(output_file, index=False)
   
    print(f"✅ Predictions saved to: {output_file}")
 
if __name__ == "__main__":
    predict()
