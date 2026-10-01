from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


# File paths
DATA_PATH = Path("data/building_data.csv")
MODEL_PATH = Path("models/isolation_forest.pkl")


# Features used by the ML model
FEATURES = [
    "indoor_temperature",
    "outdoor_temperature",
    "humidity",
    "occupancy",
    "hvac_power_kw",
]


def load_data():
    """
    Load the generated building dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist. "
            "Run generate_data.py first."
        )

    data = pd.read_csv(DATA_PATH)

    return data


def train_model(data):
    """
    Train an Isolation Forest anomaly detection model.
    """

    X = data[FEATURES]

    model = IsolationForest(    #unsupervised anomaly detection algorithm
        n_estimators=100,       #tells Isolation Forest to build 100 decision trees 
        contamination=0.05,     #expect 5% of observations to be anomalies
        random_state=42,        #If we train the model again with the same data and configuration, we should get consistent results.
    )

    model.fit(X)

    return model


def save_model(model):
    """
    Save the trained model so the API can load it later.
    """

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"Model saved successfully to: {MODEL_PATH}"
    )


def main():

    print("Loading dataset...")

    data = load_data()

    print(
        f"Loaded {len(data)} building sensor records."
    )

    print("Training Isolation Forest model...")

    model = train_model(data)

    print("Model training completed.")

    ####
    predictions = model.predict(
    data[FEATURES]
    )

    anomaly_count = (
        predictions == -1
    ).sum()

    normal_count = (
        predictions == 1
    ).sum()

    print(
        f"Normal readings: {normal_count}"
    )

    print(
        f"Anomalies detected: {anomaly_count}"
    )

    save_model(model)


if __name__ == "__main__":
    main()