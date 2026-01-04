import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 1. SETUP PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "featured_data.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots"

def visualize():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if data exists
    if not INPUT_PATH.exists():
        print(f" Error: File not found at {INPUT_PATH}")
        print("   Make sure you ran 'scripts/feature_engineering.py' first!")
        return

    print(f" Loading data from {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    
    # Sample data if it's too huge (for faster plotting)
    if len(df) > 100000:
        print(f"   Data is large ({len(df)} rows). Sampling 100,000 rows for plotting...")
        plot_df = df.sample(100000, random_state=42)
    else:
        plot_df = df

    print(" Generating plots...")

    # --- PLOT 1: Trip Duration Distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(plot_df['trip_duration'], bins=50, kde=True)
    plt.title("Distribution of Trip Duration (min)")
    plt.xlabel("Minutes")
    plt.xlim(0, 60) # Focus on trips under 1 hour
    plt.savefig(OUTPUT_DIR / "distribution_duration.png")
    plt.close()
    print(f"   Saved: {OUTPUT_DIR / 'distribution_duration.png'}")

    # --- PLOT 2: Pickup Hour vs Trip Duration ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='pickup_hour', y='trip_duration', data=plot_df)
    plt.title("Trip Duration by Hour of Day")
    plt.xlabel("Hour (0-23)")
    plt.ylabel("Duration (min)")
    plt.ylim(0, 60) # Zoom in
    plt.savefig(OUTPUT_DIR / "duration_by_hour.png")
    plt.close()
    print(f"   Saved: {OUTPUT_DIR / 'duration_by_hour.png'}")

    # --- PLOT 3: Trip Distance vs Duration (Correlation) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='trip_distance', y='trip_duration', data=plot_df.sample(10000)) # Smaller sample for scatter
    plt.title("Trip Distance vs. Duration")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Duration (min)")
    plt.xlim(0, 20)
    plt.ylim(0, 100)
    plt.savefig(OUTPUT_DIR / "distance_vs_duration.png")
    plt.close()
    print(f"   Saved: {OUTPUT_DIR / 'distance_vs_duration.png'}")

    print(" Visualization Complete!")

if __name__ == "__main__":
    visualize()