# main.py

import streamlit as st
import numpy as np
import joblib
from prediction_helper import preprocess_input, predict_species

# Load model and preprocessing objects
scaler = joblib.load('scaler.joblib')
feature_selector = joblib.load('feature_selector.joblib')
model = joblib.load('model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

st.title('Iris Flower Species Prediction')

# Input fields for features (example selected based on selected features)
inputs = {}
inputs['sepal_length'] = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.0)
inputs['sepal_width'] = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.5)
inputs['petal_length'] = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=1.5)
inputs['petal_width'] = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=0.2)

# Add other features if needed as per feature selection

if st.button('Predict'):
    input_df = preprocess_input(inputs, scaler, feature_selector)
    prediction = predict_species(input_df, model, label_encoder)
    st.write(f'Predicted Iris Species: **{prediction}**')
