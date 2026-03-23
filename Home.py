
import streamlit as st
import pandas as pd

# Page Layout
st.set_page_config(layout='wide', page_title='Used Cars Market Analysis', page_icon="📊")


st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    h1 { text-align: center; color: #00FFCC; }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("<h1> 🚗 Used Cars Price Project </h1>", unsafe_allow_html=True)


st.image('https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-1.2.1&auto=format&fit=crop&w=1500&q=80')

# Project Statistics
st.write("## 📊 Project Scope")
c1, c2, c3 = st.columns(3)
c1.metric("Dataset Size", "1.5 GB", "Cleaned")
c2.metric("Total Listings", "357,692", "Rows")
c3.metric("Model Accuracy", "80.86%", "R² Score")

st.divider()


st.write('### 🔍 Column Descriptions')
col_a, col_b = st.columns(2)

with col_a:
    st.write("**Manufacturer**: The brand of the vehicle (e.g., GMC, Toyota, Ford).")
    st.write("**Model**: The specific model name (e.g., Sierra 1500, Camry).")
    st.write("**Year**: The manufacturing year (influences depreciation).")
    st.write("**Odometer**: Total miles driven (crucial for value).")
    st.write("**Condition**: Visual/Mechanical state (Salvage to New).")

with col_b:
    st.write("**Region/State**: Geographic location influencing local demand.")
    st.write("**Drive/Type**: Drivetrain (4WD, FWD) and body style (Sedan, Pickup).")
    st.write("**Lat/Long**: Precise coordinates used for spatial pricing trends.")
    st.write("**Posting Date**: When the ad was created (handles seasonality).")

# Load Data Sample
@st.cache_data
def load_data():
    
    return pd.read_parquet('Cleaned_df.parquet')

df = load_data()

st.write("### 📄 Cleaned Data Preview")
st.dataframe(df.head(20), use_container_width=True)
