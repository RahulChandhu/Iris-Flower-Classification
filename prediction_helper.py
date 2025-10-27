# prediction_helper.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.datasets import load_iris

MODEL_PATH = "iris_model.pkl"

def train_and_save_model():
    """Train a RandomForest model on Iris dataset and save it."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved as iris_model.pkl")


def load_model():
    """Load trained model from file."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """Predict Iris species given the input features."""
    model = load_model()
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    prediction = model.predict(input_data)[0]
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    return species_map[prediction]
