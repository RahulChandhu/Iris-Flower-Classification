# main.py
import streamlit as st
from prediction_helper import train_and_save_model, predict_species
import os

st.set_page_config(page_title="Iris Flower Classifier 🌸", page_icon="🌿", layout="centered")

st.title("🌸 Iris Flower Classification App")
st.write("Enter the flower measurements below to predict its species.")

# Train model if not available
if not os.path.exists("iris_model.pkl"):
    train_and_save_model()

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Predict button
if st.button("🔍 Predict"):
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"🌼 Predicted Iris Species: **{species}**")

st.markdown("---")
st.caption("Made with ❤️ by Rahul | Powered by Streamlit")
