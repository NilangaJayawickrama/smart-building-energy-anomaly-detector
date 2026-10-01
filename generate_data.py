import numpy as np
import pandas as pd


# Makes the generated data reproducible.
# Every run will generate the same random values.
np.random.seed(42)


# Number of building sensor records
NUM_SAMPLES = 3000


# Generate outdoor temperatures.
# Average = 15°C, standard deviation = 8°C.
outdoor_temperature = np.random.normal(
    loc=15,
    scale=8,
    size=NUM_SAMPLES,
)


# Indoor building temperatures are normally
# maintained close to 22°C.
indoor_temperature = np.random.normal(
    loc=22,
    scale=1.5,
    size=NUM_SAMPLES,
)


# Generate humidity values around 45%.
humidity = np.random.normal(
    loc=45,
    scale=10,
    size=NUM_SAMPLES,
)


# Generate building occupancy between 0 and 79 people.
occupancy = np.random.randint(
    low=0,
    high=80,
    size=NUM_SAMPLES,
)


# HVAC systems usually work harder when the indoor
# and outdoor temperatures are significantly different.
temperature_difference = np.abs(
    indoor_temperature - outdoor_temperature
)


# Simulate HVAC power consumption.
#
# Power increases based on:
# 1. Indoor/outdoor temperature difference
# 2. Number of occupants
# 3. Humidity
#
# Random noise is added to make the dataset
# more realistic.
hvac_power_kw = (
    3
    + temperature_difference * 0.45
    + occupancy * 0.07
    + humidity * 0.015
    + np.random.normal(
        loc=0,
        scale=1.2,
        size=NUM_SAMPLES,
    )
)


# Prevent unrealistic humidity values.
humidity = np.clip(
    humidity,
    15,
    90,
)


# HVAC power should never be negative.
hvac_power_kw = np.clip(
    hvac_power_kw,
    0,
    None,
)


# Create a pandas DataFrame.
data = pd.DataFrame(
    {
        "indoor_temperature": indoor_temperature,
        "outdoor_temperature": outdoor_temperature,
        "humidity": humidity,
        "occupancy": occupancy,
        "hvac_power_kw": hvac_power_kw,
    }
)


# Round values to make the CSV easier to read.
data = data.round(
    {
        "indoor_temperature": 2,
        "outdoor_temperature": 2,
        "humidity": 2,
        "hvac_power_kw": 2,
    }
)


# Save the generated dataset.
output_path = "data/building_data.csv"

data.to_csv(
    output_path,
    index=False,
)


print("Dataset generated successfully.")
print(f"Number of records: {len(data)}")
print(f"Saved to: {output_path}")

print("\nFirst five records:")
print(data.head())

print("\nDataset statistics:")
print(data.describe())