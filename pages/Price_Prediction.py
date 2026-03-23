
import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor

# 1. Page Config
st.set_page_config(layout='wide', page_title='AI Price Predictor', page_icon="💸")

# 2. Title & Intro
st.markdown("<h1 style='text-align: center; color: #00FFCC;'> 💸 AI Market Value Predictor </h1>", unsafe_allow_html=True)
st.write("### Adjust the features below to see the AI's valuation.")
st.divider()

# 3. Load Resources
@st.cache_data
def get_data():
    return pd.read_parquet('Cleaned_df.parquet')

@st.cache_resource
def get_model():
    
    return joblib.load('Catboost_Model.pkl')

df = get_data()
model = get_model()

# 4. User Inputs Layout
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("📍 Location & Brand")
    # Brand selection
    manufacturer = st.selectbox('Manufacturer', sorted(df['manufacturer'].unique()))
    
    # DYNAMIC: Only show models belonging to the selected manufacturer
    brand_models = df[df['manufacturer'] == manufacturer]['model'].unique()
    car_model = st.selectbox('Specific Model', sorted(brand_models))
    
    # State & Region
    state_choice = st.selectbox('State', sorted(df['state'].unique()))
    state_regions = df[df['state'] == state_choice]['region'].unique()
    region_choice = st.selectbox('Region', sorted(state_regions))

with c2:
    st.subheader("⚙️ Vehicle Specs")
    year = st.slider('Year of Manufacture', int(df.year.min()), int(df.year.max()), 2016)
    odometer = st.number_input('Mileage (Odometer)', min_value=0, value=65000, step=1000)
    
    # Condition is a Select Slider to respect the ordinal nature
    condition = st.select_slider('Condition', 
                                options=['salvage', 'fair', 'unknown', 'good', 'excellent', 'like new', 'new'],
                                value='good')
    
    fuel = st.selectbox('Fuel Type', df['fuel'].unique())
    transmission = st.selectbox('Transmission', df['transmission'].unique())

with c3:
    st.subheader("🎨 Aesthetics & Body")
    v_type = st.selectbox('Vehicle Type', df['type'].unique())
    drive = st.selectbox('Drivetrain', df['drive'].unique())
    paint = st.selectbox('Paint Color', df['paint_color'].unique())
    title = st.selectbox('Title Status', df['title_status'].unique())

# 5. Hidden Data Processing (Behind the Scenes)
# Get the average coordinates for the selected region to keep prediction accurate
region_stats = df[df['region'] == region_choice][['lat', 'long']].mean()
lat = region_stats['lat']
long = region_stats['long']

# Capture current date info
now = datetime.datetime.now()
current_day = now.weekday()
current_month = now.month

# 6. Predict Button
st.divider()
if st.button('✨ Predict Estimated Market Value', use_container_width=True):
    
    # Create Input DataFrame
    input_df = pd.DataFrame([{
        'region': region_choice,
        'year': year,
        'manufacturer': manufacturer,
        'model': car_model,
        'condition': condition,
        'fuel': fuel,
        'odometer': odometer,
        'title_status': title,
        'transmission': transmission,
        'drive': drive,
        'type': v_type,
        'paint_color': paint,
        'state': state_choice,
        'lat': lat,
        'long': long,
        'posting_date': now, 
        'day_of_week': current_day,
        'month': current_month
    }])
    
    # Calculate Prediction
    
    prediction = model.predict(input_df)[0]
    
    # Show Results
    res1, res2, res3 = st.columns([1, 2, 1])
    with res2:
        st.success(f"### Estimated Price: **${prediction:,.2f}**")
        st.info(f"**AI Confidence:** This value is calculated based on historical trends for {manufacturer.upper()} in {state_choice.upper()}.")
        st.balloons()
