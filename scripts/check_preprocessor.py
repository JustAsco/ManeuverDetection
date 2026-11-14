
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

# Load project configuration
with open(project_path / "configs/default.yml", encoding="utf-8") as file:
    config_data = yaml.safe_load(file)

# Temporary values for loading the configuration.
# Windowing is not tested yet.
config_data["preprocess"]["block_size"] = 5.0
config_data["preprocess"]["stride_size"] = 2.5

config = Config(**config_data)
preprocessor = Preprocessor(config)

# Use our known training flight
flight_id = "0bbf0c4e-fb3c-4213-bff8-ef21ee5ebf79"

annotations = preprocessor._read_annotations(flight_id)
df = preprocessor._read_parquet_file(flight_id)

print("Original shape:", df.shape)

# Apply the preprocessing steps from the template
df = preprocessor._rename_columns(df)
df = preprocessor._extract_relevant_columns(df)
df = _convert_dtypes(df)
df = preprocessor._fix_missing_values(df)
df = _fix_index(df)
df = _expand_vectors(df)
df = preprocessor._assign_maneuver_labels(df, annotations)

print("\nProcessed shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:", df.isna().sum().sum())

print("\nClass distribution:")
counts = df["Maneuver_Label"].value_counts().sort_index()
print(counts)

# Basic checks
assert len(df) == 189764
assert df.isna().sum().sum() == 0
assert df["Maneuver_Label"].between(0, 7).all()

print("\nIntegration test successful!")
