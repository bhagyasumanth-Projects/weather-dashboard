# forecast_test.py
# Self-contained script to test Prophet forecast for Weather Data

import pandas as pd
from prophet import Prophet

# -----------------------------
# Step 1: Forecast Function
# -----------------------------
def forecast_city(data, city, column, days=7):
    """
    Forecast a given column (e.g., Avg_Temperature, Rainfall (mm), AQI) for a city
    Args:
        data   : Pandas DataFrame with at least ['Date','City',column]
        city   : City name (string)
        column : Column to forecast (string)
        days   : Number of days to predict ahead
    Returns:
        forecast : DataFrame containing historical + predicted values
    """
    # Filter city data
    city_data = data[data['City'] == city]

    # Select only Date & chosen column
    df = city_data[['Date', column]].rename(columns={'Date':'ds', column:'y'})
    df['ds'] = pd.to_datetime(df['ds'])   # Ensure correct datetime format

    # Build the model
    model = Prophet()
    model.fit(df)

    # Create future dates
    future = model.make_future_dataframe(periods=days)

    # Predict
    forecast = model.predict(future)

    return forecast


# -----------------------------
# Step 2: Load Data & Test
# -----------------------------
# Your cleaned dataset file (adjust file name if needed)
data = pd.read_excel(r"C:\Users\pylab\OneDrive\Documents\Desktop\3_cites(1)_modified.xlsx")

# Forecast Temperature for Tirupati
forecast_temp = forecast_city(data, "Tirupati", "Avg_Temperature", days=7)

# Forecast Rainfall for Tirupati
forecast_rain = forecast_city(data, "Tirupati", "Rainfall (mm)", days=7)

# -----------------------------
# Step 3: Show Output
# -----------------------------
print("✅ Model Ready! Here are the next 7-day predictions:")
print("\n--- Tirupati Temperature Forecast ---")
print(forecast_temp[['ds','yhat']].tail(7))  # Last 7 rows only

print("\n--- Tirupati Rainfall Forecast ---")
print(forecast_rain[['ds','yhat']].tail(7))
