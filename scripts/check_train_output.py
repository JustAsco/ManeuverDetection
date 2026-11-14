
import pandas as pd

from maneuver_detection.config import get_default_config

config = get_default_config()

# Load processed features
features = pd.read_parquet(config.train_data_path)

# Load corresponding time information
metadata = pd.read_csv(
    config.data_path / "train_window_metadata.csv",
    parse_dates=["Window_Start", "Window_End"]
)

print("Feature shape:", features.shape)
print("Metadata shape:", metadata.shape)

print("\nMetadata example:")
print(metadata.head())

print("\nWindow label distribution:")
print(features["Maneuver_Label"].value_counts().sort_index())

# Basic checks
assert len(features) == len(metadata)
assert metadata["Flight_ID"].nunique() == 13
assert features.isna().sum().sum() == 0

# Check the length of every time window
window_duration = (
    metadata["Window_End"] - metadata["Window_Start"]
).dt.total_seconds()

assert (window_duration == 5.0).all()

# Check label coverage
assert metadata["Label_Coverage"].between(0, 1).all()

# Check chronological order within each flight
for flight_id, flight_data in metadata.groupby("Flight_ID"):
    assert flight_data["Window_Start"].is_monotonic_increasing

print("\nAll preprocessing checks passed!")
