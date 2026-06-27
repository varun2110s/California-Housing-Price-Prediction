
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and pipeline
model = joblib.load('xgb_model.pkl')
pipeline = joblib.load('housing_pipeline.pkl')

st.title('🏠 Housing Price Predictor')
st.write('Enter the details below to predict house price')

# User inputs
longitude = st.slider('Longitude', -124.0, -114.0, -119.0)
latitude = st.slider('Latitude', 32.0, 42.0, 37.0)
housing_median_age = st.slider('Housing Median Age', 1, 52, 20)
total_rooms = st.number_input('Total Rooms', 100, 10000, 2000)
total_bedrooms = st.number_input('Total Bedrooms', 50, 2000, 400)
population = st.number_input('Population', 100, 10000, 1000)
households = st.number_input('Households', 50, 2000, 400)
median_income = st.slider('Median Income (in $10,000s)', 0.5, 15.0, 3.0)
ocean_proximity = st.selectbox('Ocean Proximity', 
    ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Predict button
if st.button('Predict Price'):
    input_data = pd.DataFrame([{
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }])
    
    transformed = pipeline.transform(input_data)
    prediction = model.predict(transformed)
    
    st.success(f'Predicted House Price: ${prediction[0]:,.0f}')