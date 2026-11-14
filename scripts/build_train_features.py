
import pandas as pd
import numpy as np

from maneuver_detection.config import get_default_config
from maneuver_detection.preprocess import Preprocessor

config = get_default_config()
preprocessor = Preprocessor(config)

all_features = []
flight_ids = []

# Process only the 13 training flights
for flight_id in config.train_data_ids:

    print(f"Processing flight: {flight_id}")

    # Complete preprocessing including windowing
    df = preprocessor.preprocess_file(flight_id)

    # Check the generated features
    assert not df.empty
    assert df.isna().sum().sum() == 0
    assert df["Maneuver_Label"].isin(config.all_labels).all()

    feature_values = df.drop(columns=["Maneuver_Label"])
    assert np.isfinite(feature_values.to_numpy(dtype=float)).all()

    # Store features and corresponding flight ID
    all_features.append(df)
    flight_ids.extend([flight_id] * len(df))

    print("Generated windows:", len(df))

# Combine all training flights
train_data = pd.concat(all_features, ignore_index=True)

# Store flight IDs separately to avoid using them as ML features
groups = pd.DataFrame({"Flight_ID": flight_ids})

assert len(train_data) == len(groups)

# Save results
train_data.to_parquet(config.train_data_path, index=False)
groups.to_csv(config.data_path / "train_flight_ids.csv", index=False)

print("\nTraining data shape:", train_data.shape)

print("\nWindow class distribution:")
print(train_data["Maneuver_Label"].value_counts().sort_index())

print("\nNumber of flights:", groups["Flight_ID"].nunique())
print("Missing values:", train_data.isna().sum().sum())
print("\nTraining dataset created successfully!")
