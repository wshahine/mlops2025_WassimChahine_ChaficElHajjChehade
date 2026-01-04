import os
import pandas as pd

# --- 1. DYNAMIC PATH SETUP ---
# This ensures the code finds the correct folder on Windows without path errors
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Build the correct paths to src/mlproject/data
INPUT_PATH = os.path.join(project_root, "src", "mlproject", "data", "cleaned_data.parquet")
OUTPUT_PATH = os.path.join(project_root, "src", "mlproject", "data", "featured_data.parquet")
ZONE_PATH = os.path.join(project_root, "src", "mlproject", "data", "taxi_zone_lookup.csv")

def create_features():
    # Check if input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"Error: File not found at: {INPUT_PATH}")
        print("Run preprocessing first!")
        return

    print(f"Loading cleaned data from {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)

    # --- 2. SMART COLUMN DETECTION ---
    # Find the actual pickup time column (whether it is 'tpep', 'lpep', or just 'pickup')
    datetime_col = None
    for col in df.columns:
        if "pickup_datetime" in col:
            datetime_col = col
            break
    
    if datetime_col is None:
        print(f"CRITICAL ERROR: No 'pickup_datetime' column found. Available columns: {df.columns.tolist()}")
        return

    # Automatically determine the dropoff column name based on the pickup name
    dropoff_col = datetime_col.replace("pickup", "dropoff")
    print(f"Using columns: '{datetime_col}' and '{dropoff_col}'")

    # --- 3. CREATE FEATURES ---
    # Convert to datetime objects using the DETECTED column names
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[dropoff_col] = pd.to_datetime(df[dropoff_col])

    df['pickup_hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek

    # Create target: trip_duration (in minutes)
    df['trip_duration'] = (df[dropoff_col] - df[datetime_col]).dt.total_seconds() / 60
    
    # Filter outliers (trips between 1 min and 3 hours)
    df = df[(df['trip_duration'] > 1) & (df['trip_duration'] < 180)]

    # --- 4. JOIN WITH TAXI ZONES ---
    if os.path.exists(ZONE_PATH):
        zones = pd.read_csv(ZONE_PATH)
        # Join Pickup Zone
        df = df.merge(zones[['LocationID', 'Borough']], left_on='PULocationID', right_on='LocationID', how='left')
        df.rename(columns={'Borough': 'PU_Borough'}, inplace=True)
        
        # Encoding (Convert Boroughs to numeric codes)
        df['PU_Borough_Code'] = df['PU_Borough'].astype('category').cat.codes
    else:
        print(f"Warning: Zone file not found at {ZONE_PATH}. Skipping zone features.")
        df['PU_Borough_Code'] = 0 

    # --- 5. SELECT & SAVE ---
    features = ['passenger_count', 'trip_distance', 'pickup_hour', 'day_of_week', 'PU_Borough_Code', 'trip_duration']
    
    # Only keep features that actually exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    final_df = df[available_features]

    print(f"Saving features to {OUTPUT_PATH}...")
    final_df.to_parquet(OUTPUT_PATH, index=False)
    print("Feature Engineering Complete.")

if __name__ == "__main__":
    create_features()
