"""
This script predicts the winners of the 2025 Japanese Grand Prix based on qualifying times
and sector times from the previous races. It uses a Gradient Boosting Regressor model to make predictions.

Author: Himanshu Kuchekar
Date: 04/01/2025
"""
import fastf1
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

cache_dir = "/Users/id-dr/Desktop/F1_Prediction/cache"  # Path to the directory which should be used to store cached data
cache_dir = os.path.expandvars(cache_dir)  # Expand environment variables if any
cache_dir = os.path.expanduser(cache_dir)  # Expand ~ to the user's home directory path

# Check if the cache directory exists, and if not, create it
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)  # This creates the directory and any necessary parents

# Then proceed with enabling the cache
fastf1.Cache.enable_cache(cache_dir)

# Load 2024 Japanese GP race session
session_2024 = fastf1.get_session(2024, "Japan", "R")
session_2024.load()

# Extract lap and sector times
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[
    ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "LapTime (s)"]].mean().reset_index()

# 2025 Qualifying Data Japanese GP with Tsunoda replacing Bearman at RedBull
qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
               "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
               "Pierre Gasly", "Daniel Ricciardo", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
    "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                           91.021, 91.079, 91.103, 91.438, 91.706,  # Improved time for Tsunoda (home race advantage)
                           91.625, 91.632, 91.688, 91.773, 91.840,
                           91.992, 92.118, 92.092, 92.141, 92.174]
})

# Add team information for better context
team_mapping = {
    "Oscar Piastri": "McLaren", "George Russell": "Mercedes", "Lando Norris": "McLaren", "Max Verstappen": "Red Bull",
    "Lewis Hamilton": "Ferrari", "Charles Leclerc": "Ferrari", "Isack Hadjar": "VCARB",
    "Andrea Kimi Antonelli": "Mercedes",
    "Yuki Tsunoda": "Red Bull", "Alexander Albon": "Williams", "Esteban Ocon": "Alpine", "Nico Hülkenberg": "Audi",
    "Fernando Alonso": "Aston Martin", "Lance Stroll": "Aston Martin", "Carlos Sainz Jr.": "Williams",
    "Pierre Gasly": "Alpine", "Daniel Ricciardo": "VCARB", "Jack Doohan": "Alpine", "Gabriel Bortoleto": "Audi",
    "Liam Lawson": "Haas"
}
qualifying_2025["Team"] = qualifying_2025["Driver"].map(team_mapping)

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Daniel Ricciardo": "RIC", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge qualifying data with sector times
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")

# Print which drivers don't have sector data
missing_data_drivers = merged_data[merged_data["Sector1Time (s)"].isna()]["Driver_x"].tolist()
if missing_data_drivers:
    print(f"Drivers without 2024 sector data: {', '.join(missing_data_drivers)}")
    print("Using median sector times for these drivers")

# Fill missing sector times with median values from available drivers
for col in ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]:
    median_value = merged_data[col].median()
    merged_data[col].fillna(median_value, inplace=True)

# Adjust sector times for Tsunoda at Red Bull (assuming better performance in Red Bull car)
# Check if Tsunoda's data exists and adjust it based on team performance
if "Yuki Tsunoda" in qualifying_2025["Driver"].values:
    # Get average Red Bull performance improvement factor using Verstappen's data
    verstappen_row = merged_data[merged_data["Driver_x"] == "Max Verstappen"]
    if not verstappen_row.empty:
        # Calculate Red Bull performance factor (approx. 2-3% better than midfield)
        rb_factor = 0.98  # Represents a 2% improvement

        # Apply this factor to Tsunoda's sector times
        tsunoda_idx = merged_data[merged_data["Driver_x"] == "Yuki Tsunoda"].index
        if not tsunoda_idx.empty:
            for col in ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]:
                if pd.notna(merged_data.loc[tsunoda_idx, col]).any():
                    merged_data.loc[tsunoda_idx, col] = merged_data.loc[tsunoda_idx, col] * rb_factor

# Get the data for drivers with complete information
complete_data = merged_data.dropna(subset=["LapTime (s)"])

# Define feature set
X = complete_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
y = complete_data["LapTime (s)"]

# Create a proper train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Gradient Boosting Model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# For prediction, use all drivers including those with estimated sector times
X_predict = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
qualifying_2025["PredictedRaceTime (s)"] = model.predict(X_predict)

# Convert predicted race time to a more readable format
qualifying_2025["PredictedRaceTime"] = qualifying_2025["PredictedRaceTime (s)"].apply(
    lambda x: f"{int(x // 60)}:{x % 60:.3f}"
)

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Japanese GP Winner with Tsunoda at Red Bull 🏁\n")
print(qualifying_2025[["Driver", "Team", "PredictedRaceTime", "PredictedRaceTime (s)"]].to_string(index=False))

# Calculate home advantage for Tsunoda
print("\n🇯🇵 Yuki Tsunoda Home Race Advantage & Red Bull Performance Boost Applied 🇯🇵")

# Use test data for proper model evaluation
y_pred_test = model.predict(X_test)
print(f"\n🔍 Model Evaluation (MAE on test data): {mean_absolute_error(y_test, y_pred_test):.2f} seconds")