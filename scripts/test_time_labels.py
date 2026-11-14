
import pandas as pd

from maneuver_detection.config import get_default_config
from maneuver_detection.preprocess import Preprocessor

config = get_default_config()
preprocessor = Preprocessor(config)

# Artificial time series
origin = pd.Timestamp("2024-02-15 08:00:00")

seconds = [
    0, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9,
    1.5, 4.5, 5.0
]

times = pd.DatetimeIndex([
    origin + pd.Timedelta(seconds=s)
    for s in seconds
])

df = pd.DataFrame({
    "GroundSpeed": [2.0] * len(times),
    "Maneuver_Label": [0] * len(times)
}, index=times)

# Hover lasts only one second
annotations = pd.Series([{
    "comment": "Hover",
    "startTimestamp": origin.isoformat(),
    "endTimeStamp": (
        origin + pd.Timedelta(seconds=1)
    ).isoformat()
}])

# Generate time-weighted features
features, metadata = preprocessor._summarize_features_in_blocks(
    df, annotations, "example-flight"
)

print("Window label:", features.loc[0, "Maneuver_Label"])
print("Label coverage:", metadata.loc[0, "Label_Coverage"])

assert features.loc[0, "Maneuver_Label"] == 0
assert abs(metadata.loc[0, "Label_Coverage"] - 0.8) < 1e-9

print("Time-weighted labeling test passed!")
