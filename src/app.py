import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import joblib
import os

st.set_page_config(page_title="Daily Insect Hitlist", page_icon="🐝", layout="wide")
st.title("All non new jersey users will be OBLITERATED with my EYE BEAMS")
st.markdown("Adjust the settings to match the conditions you plan to go bug hunting in")

model = joblib.load("species_rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
weather_columns = joblib.load("weather_columns.pkl")
#_________________________________________________________________________________________________

st.sidebar.header("Conditions")

date = st.sidebar.datetime_input("Date", step = 1800)
lat = st.sidebar.number_input("Latitude", value=40.4709, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=-74.7002, format="%.4f")

temp = st.sidebar.slider("Temperature (farenheit)", min_value=-20, max_value=120, value=67, step=1)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=67)

weather_options = [col.replace("weather_", "") for col in weather_columns]
weather_condition = st.sidebar.selectbox("Weather", options=weather_options)
#_________________________________________________________________________________________________

day_of_year = date.timetuple().tm_yday
hour = getattr(date, "hour")

day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
hour_sin = np.sin(2 * np.pi * hour / 24.0)
hour_cos = np.cos(2 * np.pi * hour / 24.0)
temp = (temp - 32)/1.8

input_data = {
        "hourly_temp_C": temp,
        "hourly_humidity_percent": humidity,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "latitude": lat,
        "longitude": lon
    }

for col in weather_columns:
        if col == f"weather_{weather_condition}":
            input_data[col] = 1
        else:
            input_data[col] = 0

input_df = pd.DataFrame([input_data])

user_probs = model.predict_proba(input_df)[0]
topN = 20
topbugs = np.argsort(user_probs)[::-1][:topN]

topprobs = user_probs[topbugs]

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("your hitlist")
    chart_records = []
    for rank, (idx, scaled_prob) in enumerate(zip(topbugs, topprobs), 1):
        species_name = label_encoder.inverse_transform([idx])[0]
        raw_prob = user_probs[idx]
        if raw_prob > 0:
            chart_records.append({
                "Rank": rank,
                "Species": species_name,
                "Raw Probability": round(raw_prob, 2)
            })
    
    results_df = pd.DataFrame(chart_records)
    if not results_df.empty:
        st.dataframe(
            results_df.set_index("Rank"),
            column_config={
                "Raw Probability": st.column_config.NumberColumn("Match %")
            },
            use_container_width=True
        )



#python -m streamlit run app.py