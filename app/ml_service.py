from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/isolation_forest.pkl")

FEATURES = [
    "indoor_temperature",
    "outdoor_temperature",
    "humidity",
    "occupancy",
    "hvac_power_kw",
]


class AnomalyDetector:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run train_model.py first."
            )

        self.model = joblib.load(MODEL_PATH)

    def predict(
        self,
        indoor_temperature: float,
        outdoor_temperature: float,
        humidity: float,
        occupancy: int,
        hvac_power_kw: float,
    ):
        sample = pd.DataFrame(
            [
                {
                    "indoor_temperature": indoor_temperature,
                    "outdoor_temperature": outdoor_temperature,
                    "humidity": humidity,
                    "occupancy": occupancy,
                    "hvac_power_kw": hvac_power_kw,
                }
            ],
            columns=FEATURES,
        )

        prediction = self.model.predict(sample)[0]
        score = self.model.decision_function(sample)[0]

        return {
            "is_anomaly": bool(prediction == -1),
            "anomaly_score": float(score),
        }