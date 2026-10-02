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


model = joblib.load(MODEL_PATH)


def create_reading(
    indoor_temperature: float,
    outdoor_temperature: float,
    humidity: float,
    occupancy: int,
    hvac_power_kw: float,
):
    return pd.DataFrame(
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


def test_normal_reading():
    reading = create_reading(
        indoor_temperature=22,
        outdoor_temperature=10,
        humidity=45,
        occupancy=30,
        hvac_power_kw=12,
    )

    prediction = model.predict(reading)[0]

    assert prediction == 1


def test_anomalous_reading():
    reading = create_reading(
        indoor_temperature=22,
        outdoor_temperature=21,
        humidity=40,
        occupancy=2,
        hvac_power_kw=80,
    )

    prediction = model.predict(reading)[0]

    assert prediction == -1


def test_anomaly_score_is_numeric():
    reading = create_reading(
        indoor_temperature=22,
        outdoor_temperature=10,
        humidity=45,
        occupancy=30,
        hvac_power_kw=12,
    )

    score = model.decision_function(reading)[0]

    assert isinstance(float(score), float)