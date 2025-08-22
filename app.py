# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import plotly.graph_objects as go

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="AP Weather & AQI Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS for Styling
# ------------------------------
def style_dashboard():
    """
    Injects custom CSS to style the dashboard with a pale sky blue theme
    and sets a unique color for the sidebar controls.
    """
    st.markdown("""
    <style>
        /* Main background with a sky-like gradient */
        [data-testid="stAppViewContainer"] {
            background-image: linear-gradient(to bottom, #a1c4fd 0%, #c2e9fb 100%);
            background-size: cover;
        }

        /* Main content block with a semi-transparent white background */
        [data-testid="stVerticalBlock"] .st-emotion-cache-15i5057 {
            background-color: rgba(255, 255, 255, 0.5); /* Semi-transparent white */
            border-radius: 15px;
            padding: 2rem;
        }
        
        /* KPI Card Style for the light theme */
        .kpi-card {
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            color: #333; /* Dark text for readability */
            border: 1px solid #ddd;
            height: 100%;
        }
        .kpi-card h3 { font-size: 1.25rem; margin-bottom: 10px; color: #555; }
        .kpi-card .value { font-size: 2.2rem; font-weight: bold; color: #000; }
        .kpi-card .sub-text { font-size: 0.9rem; color: #666; }

        /* General text and headers for the light theme */
        [data-testid="stMarkdownContainer"] p, h1, h3 { color: #333 !important; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }

        /* --- UNIQUE COLOR FOR SIDEBAR CONTROLS --- */
        /* Make all headings and labels in the sidebar a deep indigo color */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] label {
            color: #4B0082 !important; /* Deep Indigo */
        }
        
        /* Keep the info box text white for contrast on its blue background */
        [data-testid="stSidebar"] .st-emotion-cache-1g6goon {
            color: white !important;
        }
        
        /* Make the text inside the dropdowns black for readability */
        [data-testid="stSidebar"] .st-emotion-cache-b7h6b3,
        [data-testid="stSidebar"] .st-emotion-cache-1n76a9l {
             color: black !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply the custom styles
style_dashboard()

# ------------------------------
# Caching Functions for Performance
# ------------------------------
@st.cache_data
def load_data(file_path):
    """
    Load, clean, and preprocess data.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
        return pd.DataFrame()

    rename_dict = {
        'DATE': 'Date',
        'T2M': 'Avg_Temperature',
        'T2M_MAX': 'Max_Temperature',
        'T2M_MIN': 'Min_Temperature',
        'PRECTOTCORR': 'Rainfall(mm)'
    }
    data.rename(columns=rename_dict, inplace=True)
    
    if 'Date' not in data.columns:
        st.error(f"A 'Date' or 'DATE' column was not found in the file. Columns found: {list(data.columns)}")
        return pd.DataFrame()

    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    data = data.dropna(subset=['Date'])

    coords = {
        'Anantapur': (14.6819, 77.6006), 'Chittoor': (13.2176, 79.1001),
        'Eluru': (16.7118, 81.0963), 'Guntur': (16.3067, 80.4365),
        'Hindupur': (13.8295, 77.4929), 'Kadapa': (14.4673, 78.8242),
        'Kakinada': (16.9891, 82.2475), 'Kurnool': (15.8281, 78.0373),
        'Machilipatnam': (16.175, 81.1325), 'Nellore': (14.4426, 79.9865),
        'Ongole': (15.5057, 80.0483), 'Rajahmundry': (17.0005, 81.8040),
        'Tirupati': (13.6288, 79.4192), 'Vijayawada': (16.5062, 80.6480),
        'Visakhapatnam': (17.6868, 83.2185)
    }
    data['lat'] = data['City'].map(lambda x: coords.get(x, (None, None))[0])
    data['lon'] = data['City'].map(lambda x: coords.get(x, (None, None))[1])

    return data

@st.cache_data
def forecast_city(_data, city, column, days=7):
    """
    Forecast future values for a given column & city using Prophet.
    """
    city_data = _data[_data['City'] == city]
    df = city_data[['Date', column]].rename(columns={'Date':'ds', column:'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    last_date = df['ds'].max()
    forecast_future = forecast[forecast['ds'] > last_date].copy()

    if column == "Rainfall(mm)":
        forecast_future['yhat'] = forecast_future['yhat'].clip(lower=0)
        forecast_future['yhat_lower'] = forecast_future['yhat_lower'].clip(lower=0)
        forecast_future['yhat_upper'] = forecast_future['yhat_upper'].clip(lower=0)

    return forecast_future

# ------------------------------
# Load Data
# ------------------------------
DATA_FILE = "all_cities_combined_AQI.csv"
data = load_data(DATA_FILE)

if data.empty:
    st.stop()

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.title("⚙️ Dashboard Controls")

city = st.sidebar.selectbox("Select a City for Detailed View", sorted(data['City'].unique()))

min_date = data['Date'].min().date()
max_date = data['Date'].max().date()

st.sidebar.info("Select the start and end date for your desired range.")

date_range = st.sidebar.date_input(
    "Select Date Range for Historical Data",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) != 2:
    st.sidebar.warning("Please select both a start and an end date.")
    st.stop()

start_date, end_date = date_range

st.sidebar.markdown("---")
compare_cities = st.sidebar.multiselect(
    "Select Cities to Compare",
    sorted(data['City'].unique()),
    default=['Vijayawada', 'Visakhapatnam', 'Tirupati']
)

# ------------------------------
# Main Page
# ------------------------------
st.title(f"🌦️ Weather & AQI Dashboard: {city}")
st.markdown(f"Displaying data from **{start_date.strftime('%d-%m-%Y')}** to **{end_date.strftime('%d-%m-%Y')}**.")

city_data = data[
    (data['City'] == city) &
    (data['Date'].dt.date >= start_date) &
    (data['Date'].dt.date <= end_date)
]

# ------------------------------
# KPIs
# ------------------------------
st.subheader("📊 Key Metrics Overview")
c1, c2, c3 = st.columns(3)
with c1:
    c1.metric("🌡️ Avg Temp (°C)", f"{city_data['Avg_Temperature'].mean():.2f}")
with c2:
    c2.metric("☔ Total Rainfall (mm)", f"{city_data['Rainfall(mm)'].sum():.2f}")
with c3:
    c3.metric("💨 Mean AQI", f"{city_data['AQI'].mean():.2f}")

# ------------------------------
# Tabs for Organization
# ------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Historical Analysis & Map", "🔮 7-Day Forecast", "🆚 City Comparison"])

with tab1:
    st.header(f"Historical Data for {city}")

    st.subheader("🗺️ Geographical Overview of Average Temperatures")
    map_data = data.groupby('City').agg({
        'Avg_Temperature': 'mean',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()
    st.map(map_data, latitude='lat', longitude='lon', size='Avg_Temperature', color='#FF4B4B')
    st.info("Map markers are sized based on the average temperature over the entire period.")

    st.subheader("📈 Interactive Time Series Chart")
    metric_to_plot = st.selectbox(
        "Choose a metric to display:",
        ["Avg_Temperature", "Rainfall(mm)", "AQI"]
    )

    if metric_to_plot == "Rainfall(mm)":
        fig = px.bar(city_data, x='Date', y=metric_to_plot, title=f"Historical {metric_to_plot} in {city}")
    else:
        fig = px.line(city_data, x='Date', y=metric_to_plot, title=f"Historical {metric_to_plot} in {city}", markers=True)

    fig.update_layout(
        title_x=0.5, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(255,255,255,0.3)',
        title_font_color="darkblue",
        xaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
        yaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
        legend=dict(font_color="darkblue")
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header(f"7-Day Forecast for {city}")
    st.markdown(f"Predictions for the week following **{max_date.strftime('%d-%m-%Y')}**.")

    forecast_temp = forecast_city(data, city, "Avg_Temperature", days=7)
    forecast_rain = forecast_city(data, city, "Rainfall(mm)", days=7)

    avg_pred_temp = forecast_temp['yhat'].mean()
    total_pred_rain = forecast_rain['yhat'].sum()

    st.info(
        f"**Forecast Summary:** Over the next 7 days, the average temperature in **{city}** is predicted to be around **{avg_pred_temp:.1f}°C**, "
        f"with a total expected rainfall of **{total_pred_rain:.1f} mm**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌡️ Predicted Temperature")
        fig_temp_future = px.line(
            forecast_temp, 
            x='ds', 
            y='yhat', 
            title="Temperature Forecast",
            labels={'ds': 'Date', 'yhat': 'Predicted Temperature (°C)'}
        )
        fig_temp_future.add_scatter(x=forecast_temp['ds'], y=forecast_temp['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='rgba(255, 0, 0, 0.2)'), name='Upper Bound')
        fig_temp_future.add_scatter(x=forecast_temp['ds'], y=forecast_temp['yhat_lower'], fill='tozeroy', mode='lines', line=dict(color='rgba(0, 0, 255, 0.2)'), name='Lower Bound')
        fig_temp_future.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(255,255,255,0.3)',
            title_font_color="darkblue",
            xaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
            yaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
            legend=dict(font_color="darkblue")
        )
        st.plotly_chart(fig_temp_future, use_container_width=True)

        temp_table = forecast_temp[['ds','yhat','yhat_lower','yhat_upper']].rename(
            columns={'ds':'Date','yhat':'Predicted Temp (°C)','yhat_lower':'Lower','yhat_upper':'Upper'}
        )
        st.dataframe(
            temp_table.style.format({'Date': '{:%d-%m-%Y}'}).background_gradient(subset=["Predicted Temp (°C)"], cmap="RdYlGn_r"),
            use_container_width=True
        )

    with col2:
        st.subheader("☔ Predicted Rainfall")
        fig_rain_future = px.line(
            forecast_rain, 
            x='ds', 
            y='yhat', 
            title="Rainfall Forecast",
            labels={'ds': 'Date', 'yhat': 'Predicted Rainfall (mm)'}
        )
        fig_rain_future.add_scatter(x=forecast_rain['ds'], y=forecast_rain['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0, 100, 80, 0.2)'), name='Upper Bound')
        fig_rain_future.add_scatter(x=forecast_rain['ds'], y=forecast_rain['yhat_lower'], fill='tozeroy', mode='lines', line=dict(color='rgba(0, 100, 80, 0.2)'), name='Lower Bound')
        fig_rain_future.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(255,255,255,0.3)',
            title_font_color="darkblue",
            xaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
            yaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
            legend=dict(font_color="darkblue")
        )
        st.plotly_chart(fig_rain_future, use_container_width=True)

        rain_table = forecast_rain[['ds','yhat','yhat_lower','yhat_upper']].rename(
            columns={'ds':'Date','yhat':'Predicted Rainfall (mm)','yhat_lower':'Lower','yhat_upper':'Upper'}
        )
        st.dataframe(
            rain_table.style.format({'Date': '{:%d-%m-%Y}'}).background_gradient(subset=["Predicted Rainfall (mm)"], cmap="Blues"),
            use_container_width=True
        )
        
    st.markdown("---")
    st.subheader("💡 Insights & Suggestions")

    temp_insight = ""
    if avg_pred_temp > 30:
        temp_insight = f"🥵 **Hot Weather Alert:** With an average temperature of **{avg_pred_temp:.1f}°C**, expect hot conditions. It's a good idea to stay hydrated and avoid direct sun during peak hours."
    elif avg_pred_temp > 25:
        temp_insight = f"☀️ **Warm & Pleasant:** The weather looks warm and pleasant, with an average of **{avg_pred_temp:.1f}°C**. Great for outdoor activities!"
    else:
        temp_insight = f"😊 **Cool & Comfortable:** The forecast predicts a comfortable average temperature of **{avg_pred_temp:.1f}°C**."
    
    st.markdown(temp_insight)

    rain_insight = ""
    if total_pred_rain > 50:
        rain_insight = f"🌧️ **Heavy Rain Expected:** A significant amount of rain (**{total_pred_rain:.1f} mm**) is forecasted for the week. Be prepared for wet conditions and carry an umbrella."
    elif total_pred_rain > 10:
        rain_insight = f"🌦️ **Moderate Showers:** Expect some moderate showers throughout the week, with a total of **{total_pred_rain:.1f} mm** predicted. "
    else:
        rain_insight = f"🌤️ **Mostly Dry:** The forecast shows very little rain (**{total_pred_rain:.1f} mm**), so you can expect mostly dry and clear skies."

    st.markdown(rain_insight)


with tab3:
    st.header("City Comparison")

    if not compare_cities:
        st.warning("Please select at least one city from the sidebar to compare.")
    else:
        comparison_data = data[
            (data['City'].isin(compare_cities)) &
            (data['Date'].dt.date >= start_date) &
            (data['Date'].dt.date <= end_date)
        ]

        st.subheader("🌡️ Average Temperature Comparison")
        fig_comp_temp = px.line(comparison_data, x='Date', y='Avg_Temperature', color='City',
                                title='Temperature Trends Across Cities')
        fig_comp_temp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(255,255,255,0.3)',
            title_font_color="darkblue",
            xaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
            yaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
            legend=dict(font_color="darkblue")
        )
        st.plotly_chart(fig_comp_temp, use_container_width=True)

        st.subheader("☔ Total Rainfall Comparison")
        total_rain = comparison_data.groupby('City')['Rainfall(mm)'].sum().reset_index()
        fig_comp_rain = px.bar(total_rain, x='City', y='Rainfall(mm)', color='City',
                               title='Total Rainfall Across Cities (for selected date range)')
        fig_comp_rain.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(255,255,255,0.3)',
            title_font_color="darkblue",
            xaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
            yaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
            legend=dict(font_color="darkblue")
        )
        st.plotly_chart(fig_comp_rain, use_container_width=True)
        
        st.subheader("💨 AQI Trend Comparison")
        fig_comp_aqi = px.line(comparison_data, x='Date', y='AQI', color='City',
                               title='AQI Trends Across Cities')
        fig_comp_aqi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(255,255,255,0.3)',
            title_font_color="darkblue",
            xaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
            yaxis=dict(title_font_color="darkblue", tickfont_color="darkblue"),
            legend=dict(font_color="darkblue")
        )
        st.plotly_chart(fig_comp_aqi, use_container_width=True)
