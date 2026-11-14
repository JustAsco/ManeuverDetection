
import pandas as pd
from pathlib import Path
import yaml

from maneuver_detection.config import Config
from maneuver_detection.preprocess import (
    Preprocessor,
    _convert_dtypes,
    _fix_index,
    _expand_vectors
)

# Project directory
project_path = Path(__file__).resolve().parents[1]

# Load configuration
with open(project_path / "configs/default.yml", encoding="utf-8") as file:
    config_data = yaml.safe_load(file)

# Temporary settings (no windowing in this test)
config_data["preprocess"]["block_size"] = 5.0
config_data["preprocess"]["stride_size"] = 2.5

config = Config(**config_data)
preprocessor = Preprocessor(config)

successful = 0
failed = []

# Only use training flights
for flight_id in config.train_data_ids:

    print(f"\nChecking flight: {flight_id}")

    try:
        annotations = preprocessor._read_annotations(flight_id)
        df = preprocessor._read_parquet_file(flight_id)

        original_rows = len(df)

        # Apply our preprocessing functions
        df = preprocessor._rename_columns(df)
        df = preprocessor._extract_relevant_columns(df)
        df = _convert_dtypes(df)
        df = preprocessor._fix_missing_values(df)
        df = _fix_index(df)
        df = _expand_vectors(df)
        df = preprocessor._assign_maneuver_labels(df, annotations)

        # Check results
        assert len(df) == original_rows
        assert df.isna().sum().sum() == 0
        assert df["Maneuver_Label"].isin(config.all_labels).all()

        # Check all annotated intervals
        for maneuver in annotations:

            start = pd.Timestamp(maneuver["startTimestamp"])
            end = pd.Timestamp(maneuver["endTimeStamp"])

            name = maneuver["comment"].strip().lower()
            expected_label = config.label2id[name]

            interval = df.loc[
                (df.index >= start) &
                (df.index < end)
            ]

            assert len(interval) > 0
            assert (interval["Maneuver_Label"] == expected_label).all()

        print("Rows:", len(df))
        print("Missing values:", df.isna().sum().sum())
        print("Annotations:", len(annotations))
        print("Status: OK")

        successful += 1

    except Exception as error:
        print("Status: FAILED")
        print("Error:", error)
        failed.append((flight_id, str(error)))

# Final summary
print("\n-------------------------")
print("PREPROCESSING TEST SUMMARY")
print("-------------------------")

print("Successful flights:", successful)
print("Failed flights:", len(failed))

for flight_id, error in failed:
    print(f"{flight_id}: {error}")
