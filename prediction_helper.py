import joblib
import numpy as np
import pandas as pd

class IrisPredictor:
    def __init__(self, model_path="iris_model.joblib", label_encoder_path="species_encoder.joblib"):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def preprocess_input(self, features: dict):
        """
        features: dict with keys like
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "soil_type": "loamy"
        }
        """
        df = pd.DataFrame([features])

        # Encode soil_type same way as training
        soil_map = {"loamy": 0, "clay": 1, "sandy": 2}  # adjust based on training encoding
        if "soil_type" in df.columns:
            df["soil_type"] = df["soil_type"].map(soil_map).fillna(0).astype(int)

        return df.values

    def predict(self, features: dict):
        X = self.preprocess_input(features)
        pred = self.model.predict(X)
        species = self.label_encoder.inverse_transform(pred)
        return species[0]

