import os
import pickle
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Read the local CSV file
csv_filename = "AP_Crop_Wholesale_Prices_2020_2024.csv"
df = pd.read_csv(csv_filename)

# Directory to store the models (using relative path for Windows compatibility)
model_dir = "crop_price_models"
os.makedirs(model_dir, exist_ok=True)

# Get all unique crop names from the dataset
all_crops = df["Crop"].str.lower().unique()

# Dictionary to store forecast results
forecast_summary = {}

# Loop through each crop and train/save ARIMA model
for crop in all_crops:
    crop_df = df[df["Crop"].str.lower() == crop].copy()
    crop_df["Date"] = pd.to_datetime(crop_df["Month"] + " " + crop_df["Year"].astype(str))
    crop_df.sort_values("Date", inplace=True)
    crop_df.set_index("Date", inplace=True)
    crop_df.rename(columns={"Average_Wholesale_Price (Rs/quintal)": "Price"}, inplace=True)

    ts = crop_df["Price"]

    # Only train if sufficient data points
    if len(ts) >= 12:
        try:
            model = ARIMA(ts, order=(1, 1, 1))
            model_fit = model.fit()

            # Save the model to .pkl
            model_filename = os.path.join(model_dir, f"{crop}_arima_model.pkl")
            with open(model_filename, "wb") as f:
                pickle.dump(model_fit, f)

            # Forecast next 6 months
            forecast = model_fit.forecast(steps=6)
            forecast_summary[crop.title()] = forecast.round(2).tolist()
            print(f"Successfully trained model for {crop.title()}")

        except Exception as e:
            forecast_summary[crop.title()] = f"Model Error: {str(e)}"
            print(f"Error training model for {crop.title()}: {str(e)}")
    else:
        forecast_summary[crop.title()] = "Insufficient data"
        print(f"Insufficient data for {crop.title()} (only {len(ts)} data points)")

print("\nForecast Summary:")
for crop, forecast in forecast_summary.items():
    print(f"{crop}: {forecast}")

print(f"\nModels saved in directory: {model_dir}")