# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# ------------------------------
# Forecast Function
# ------------------------------
def forecast_city(data, city, column, days=7):
    """
    Forecast future values for a given column & city
    Returns ONLY the future predictions (not historical fit).
    """
    city_data = data[data['City'] == city]
    df = city_data[['Date', column]].rename(columns={'Date':'ds', column:'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Split history vs future
    last_date = df['ds'].max()
    forecast_future = forecast[forecast['ds'] > last_date]

    return forecast_future


# ------------------------------
# Load Data
# ------------------------------
data = pd.read_csv("ap_cities_with_rainfall.csv")
# Clean date strings
data['Date'] = data['Date'].astype(str).str.strip()

# Convert to datetime with error coercion to handle invalid formats gracefully
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Optionally, drop rows where date conversion failed (NaT values)
data = data.dropna(subset=['Date'])

st.title("🌦 Weather, Rainfall & AQI Dashboard")

# ------------------------------
# City Selector
# ------------------------------
city = st.selectbox("Select City", data['City'].unique())
city_data = data[data['City'] == city]

# ------------------------------
# KPIs
# ------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("🌡 Avg Temp (°C)", round(city_data['Avg_Temperature'].mean(), 2))
c2.metric("☔ Total Rainfall (mm)", round(city_data['Rainfall (mm)'].sum(), 2))
c3.metric("💨 Mean AQI", round(city_data['AQI'].mean(), 2))

# ------------------------------
# Historical Charts
# ------------------------------
st.subheader(f"📊 Historical Data for {city} (up to {city_data['Date'].max().date()})")

fig_temp = px.line(city_data, x='Date', y='Avg_Temperature', title="Historical Temperature")
st.plotly_chart(fig_temp, use_container_width=True, key="hist_temp")

fig_rain = px.bar(city_data, x='Date', y='Rainfall (mm)', title="Historical Rainfall")
st.plotly_chart(fig_rain, use_container_width=True, key="hist_rain")

fig_aqi = px.line(city_data, x='Date', y='AQI', title="Historical AQI")
st.plotly_chart(fig_aqi, use_container_width=True, key="hist_aqi")

# ------------------------------
# Forecast Section
# ------------------------------
st.subheader(f"🔮 7-Day Forecast for {city} (after {city_data['Date'].max().date()})")

# Temperature Forecast
forecast_temp = forecast_city(data, city, "Avg_Temperature", days=7)

# Rainfall Forecast (apply clipping fix)
forecast_rain = forecast_city(data, city, "Rainfall (mm)", days=7)
forecast_rain['yhat'] = forecast_rain['yhat'].clip(lower=0)
forecast_rain['yhat_lower'] = forecast_rain['yhat_lower'].clip(lower=0)
forecast_rain['yhat_upper'] = forecast_rain['yhat_upper'].clip(lower=0)

# --- Forecast Charts ---
fig_temp_future = px.line(forecast_temp, x='ds', y='yhat', title="Predicted Temperature (Next 7 Days)")
st.plotly_chart(fig_temp_future, use_container_width=True, key="pred_temp")

fig_rain_future = px.line(forecast_rain, x='ds', y='yhat', title="Predicted Rainfall (Next 7 Days)")
st.plotly_chart(fig_rain_future, use_container_width=True, key="pred_rain")


st.markdown("### 📋 Predicted Temperature Table")
temp_table = forecast_temp[['ds','yhat','yhat_lower','yhat_upper']].rename(
    columns={'ds':'Date','yhat':'Predicted Temp (°C)',
             'yhat_lower':'Lower Bound','yhat_upper':'Upper Bound'}
)
st.dataframe(
    temp_table.style.background_gradient(
        subset=["Predicted Temp (°C)"], cmap="RdYlGn_r"  # red high, green low
    ),
    use_container_width=True
)

st.markdown("### 📋 Predicted Rainfall Table")
rain_table = forecast_rain[['ds','yhat','yhat_lower','yhat_upper']].rename(
    columns={'ds':'Date','yhat':'Predicted Rainfall (mm)',
             'yhat_lower':'Lower Bound','yhat_upper':'Upper Bound'}
)
st.dataframe(
    rain_table.style.background_gradient(
        subset=["Predicted Rainfall (mm)"], cmap="Blues"  # light blue low, dark blue high
    ),
    use_container_width=True
)

# ------------------------------
# Insights with Confidence Interval
# ------------------------------

# Get the last predicted row for temperature & rainfall
temp_pred = forecast_temp.tail(1).iloc[0]
rain_pred = forecast_rain.tail(1).iloc[0]

st.markdown("### 📝 Forecast Insights")

st.info(
    f"In the next 7 days, **{city}** is expected to have an average temperature "
    f"between **{round(temp_pred['yhat_lower'],1)}°C** and **{round(temp_pred['yhat_upper'],1)}°C** "
    f"(best estimate: **{round(temp_pred['yhat'],1)}°C**). \n\n"
    f"Rainfall is forecasted to be between **{round(rain_pred['yhat_lower'],1)} mm** "
    f"and **{round(rain_pred['yhat_upper'],1)} mm** "
    f"(best estimate: **{round(rain_pred['yhat'],1)} mm**)."
)

# Extra storytelling
if rain_pred['yhat'] > 50:
    st.warning("⚠️ Heavy rainfall expected → possible flood risk.")
elif rain_pred['yhat'] > 10:
    st.success("🌧️ Moderate rainfall → favorable for agriculture.")
else:
    st.write("☀️ Mostly dry conditions predicted.")
# --- Forecast Tables with Color Highlights ---
