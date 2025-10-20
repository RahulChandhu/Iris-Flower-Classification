# prediction_helper.py

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def preprocess_input(input_dict, scaler, feature_selector):
    # Convert input to DataFrame with correct column names
    df = pd.DataFrame([input_dict])

    # --- Feature engineering using input keys from main.py ---
    df["sepalarea"] = df["sepal_length"] * df["sepal_width"]
    df["petalarea"] = df["petal_length"] * df["petal_width"]
    df["sepalareasqrt"] = np.sqrt(df["sepalarea"])
    df["petalareasqrt"] = np.sqrt(df["petalarea"])
    df["arearatios"] = df["sepalarea"] / (df["petalarea"] + 1e-6)
    df["sepaltopetallengthratio"] = df["sepal_length"] / (df["petal_length"] + 1e-6)
    df["sepaltopetalwidthratio"] = df["sepal_width"] / (df["petal_width"] + 1e-6)
    df["sepalpetallengthdiff"] = df["sepal_length"] - df["petal_length"]
    df["sepalpetalwidthdiff"] = df["sepal_width"] - df["petal_width"]
    # Add additional engineered features below as needed

    # Make sure all columns needed by scaler exist
    trained_columns = scaler.feature_names_in_
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0  # Or an appropriate default

    df = df[trained_columns]  # Ensure correct order

    scaled = scaler.transform(df)
    selected = feature_selector.transform(scaled)
    return selected

    # Now you can safely scale and select
    scaled = scaler.transform(df)
    selected = feature_selector.transform(scaled)
    return selected

def predict_species(input_data, model, label_encoder):
    pred = model.predict(input_data)
    species = label_encoder.inverse_transform(pred)
    return species[0]
