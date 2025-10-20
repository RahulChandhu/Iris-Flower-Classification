import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
from prediction_helper import IrisPredictor


def train_and_save():
    """Train the model using only 5 features and save it with encoders."""
    df = pd.read_csv("iris_extended.csv")

    # Encode soil_type
    soil_encoder = LabelEncoder()
    df["soil_type"] = soil_encoder.fit_transform(df["soil_type"])

    # Encode species (target)
    species_encoder = LabelEncoder()
    df["species"] = species_encoder.fit_transform(df["species"])

    # Use only the 5 features for training
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width", "soil_type"]
    X = df[features]
    y = df["species"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # RandomForest + GridSearch
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 10, None],
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model + encoder
    joblib.dump(best_model, "iris_model.joblib")
    joblib.dump(species_encoder, "species_encoder.joblib")
    print("✅ Model trained and saved with 5 features!")


def run_app():
    """Streamlit Web App"""
    st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

    st.title("🌸 Iris Flower Classifier")
    st.write("Enter the measurements below to predict the species of Iris flower.")

    # Inputs
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)
    soil_type = st.selectbox("Soil Type", ["loamy", "clay", "sandy"])

    if st.button("🔮 Predict"):
        try:
            predictor = IrisPredictor("iris_model.joblib", "species_encoder.joblib")
            result = predictor.predict({
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
                "soil_type": soil_type
            })
            st.success(f"🌟 Predicted Species: **{result}**")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")


if __name__ == "__main__":
    # Train model (runs once at start)
    train_and_save()

    # Launch Streamlit app
    run_app()
