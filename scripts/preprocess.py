import pandas as pd
import os

# Define Paths
RAW_DATA_PATH = r"C:\Users\chafi\OneDrive\Desktop\mlops_WassimChahine_ChaficElHajjChehade\mlops_WassimChahine_ChaficElHajjChehade\src\mlproject\data\train.csv"
OUTPUT_PATH = "data/cleaned_data.parquet"

def preprocess():
    print("🚀 Starting Preprocessing...")
    
    # 1. Load Data
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"File not found: {RAW_DATA_PATH}")
    
    # Load and immediately clean column names for robustness
    df = pd.read_csv(RAW_DATA_PATH, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    
    # UPDATED COLUMN NAMES: Based on your terminal output
    # The columns in your CSV are 'pickup_datetime' and 'dropoff_datetime'
    date_cols = ['pickup_datetime', 'dropoff_datetime']
    
    print(f"   Detected columns: {list(df.columns)}")

    # Convert to datetime
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            print(f"❌ Error: Required column '{col}' is still missing!")
            return

    print(f"   Original shape: {df.shape}")

    # 2. Handle Missing Values
    df.dropna(inplace=True)

    # 3. Remove Invalid Rows
    # Based on your CSV, 'trip_distance' and 'passenger_count' exist
    if 'trip_distance' in df.columns and 'passenger_count' in df.columns:
        df = df[df['trip_distance'] > 0]
        df = df[df['passenger_count'] > 0]
    
    # Calculate duration
    df['duration_min'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
    
    # Filter trips < 1 min or > 3 hours
    df = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 180)]
    print(f"   Final shape after cleaning: {df.shape}")

    # 4. Save Cleaned Data
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Ensure you have 'pyarrow' installed: pip install pyarrow
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Preprocessing complete. Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
