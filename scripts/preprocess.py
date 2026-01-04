import pandas as pd
import os
from pathlib import Path

# --- 1. Define Paths Dynamically ---
# This finds the root directory of your project automatically
# It assumes this script is running from the project root (where pyproject.toml is)
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
# If running via 'uv run scripts/preprocess.py', we can often rely on relative paths directly:
RAW_DATA_PATH = Path("data/train.csv")
OUTPUT_PATH = Path("data/cleaned_data.parquet")

def preprocess():
    print(" Starting Preprocessing...")
    
    # 2. Check if file exists
    if not RAW_DATA_PATH.exists():
        # Fallback: Print the current working directory to help debug
        print(f" Error: File not found at {RAW_DATA_PATH.absolute()}")
        print(f"   Current working directory is: {os.getcwd()}")
        print("   Make sure you are running the command from the root folder!")
        return
    
    # 3. Load Data
    df = pd.read_csv(RAW_DATA_PATH, skipinitialspace=True)
    
    # Clean column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    print(f"   Detected columns: {list(df.columns)}")

    # 4. Convert to datetime
    date_cols = ['pickup_datetime', 'dropoff_datetime']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            print(f" Warning: Column '{col}' missing from dataset.")

    print(f"   Original shape: {df.shape}")

    # 5. Handle Missing Values
    df.dropna(inplace=True)

    # 6. Remove Invalid Rows (Basic Logic)
    if 'trip_duration' in df.columns:
        # The target variable is usually trip_duration (seconds) in this dataset
        df = df[df['trip_duration'] > 0]
        # Convert seconds to minutes for sanity check
        df['duration_min'] = df['trip_duration'] / 60
    elif 'pickup_datetime' in df.columns and 'dropoff_datetime' in df.columns:
        # Calculate manually if target column is missing
        df['duration_min'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
    
    # Filter: Trips between 1 min and 3 hours
    if 'duration_min' in df:
        df = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 180)]

    if 'passenger_count' in df.columns:
        df = df[df['passenger_count'] > 0]

    print(f"   Final shape after cleaning: {df.shape}")

    # 7. Save Cleaned Data
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet (requires pyarrow or fastparquet)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f" Preprocessing complete. Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()