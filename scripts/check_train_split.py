
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from maneuver_detection.config import get_default_config

config = get_default_config()

# Load processed training data
df = pd.read_parquet(config.train_data_path)
groups = pd.read_csv(
    config.data_path / "train_flight_ids.csv"
)["Flight_ID"]

# Separate features and labels
X = df.drop(columns=["Maneuver_Label"])
y = df["Maneuver_Label"]

assert len(df) == len(groups)

# Split by complete flights, not by individual windows
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=3,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

# Get flight IDs in both groups
train_flights = set(groups.iloc[train_idx])
val_flights = set(groups.iloc[val_idx])

# Make sure no flight appears in both groups
assert train_flights.isdisjoint(val_flights)

print("Training flights:", len(train_flights))
print("Validation flights:", len(val_flights))

print("Training windows:", len(train_idx))
print("Validation windows:", len(val_idx))

print("Data leakage check passed!")
