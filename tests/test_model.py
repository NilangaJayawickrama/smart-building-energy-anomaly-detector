import joblib
import pandas as pd


MODEL_PATH = "models/isolation_forest.pkl"


FEATURES = [
    "indoor_temperature",
    "outdoor_temperature",
    "humidity",
    "occupancy",
    "hvac_power_kw",
]


model = joblib.load(MODEL_PATH)


def test_reading(
    indoor_temperature,
    outdoor_temperature,
    humidity,
    occupancy,
    hvac_power_kw,
):

    reading = pd.DataFrame(
        [
            {
                "indoor_temperature":
                    indoor_temperature,

                "outdoor_temperature":
                    outdoor_temperature,

                "humidity":
                    humidity,

                "occupancy":
                    occupancy,

                "hvac_power_kw":
                    hvac_power_kw,
            }
        ],
        columns=FEATURES,
    )

    prediction = model.predict(reading)[0]

    anomaly_score = (
        model.decision_function(reading)[0]
    )

    status = (
        "ANOMALY"
        if prediction == -1
        else "NORMAL"
    )

    print("\nBuilding Sensor Reading")
    print("------------------------")

    print(
        f"Indoor Temperature: {indoor_temperature} °C"
    )

    print(
        f"Outdoor Temperature: {outdoor_temperature} °C"
    )

    print(
        f"Humidity: {humidity}%"
    )

    print(
        f"Occupancy: {occupancy}"
    )

    print(
        f"HVAC Power: {hvac_power_kw} kW"
    )

    print(
        f"Prediction: {status}"
    )

    print(
        f"Anomaly Score: {anomaly_score:.4f}"
    )


print(
    "\nTEST 1 - Expected normal reading"
)

test_reading(
    indoor_temperature=22,
    outdoor_temperature=10,
    humidity=45,
    occupancy=30,
    hvac_power_kw=12,
)


print(
    "\nTEST 2 - Suspicious high energy usage"
)

test_reading(
    indoor_temperature=22,
    outdoor_temperature=21,
    humidity=40,
    occupancy=2,
    hvac_power_kw=80,
)