
from pathlib import Path
from collections import Counter, defaultdict
import json
import pandas as pd
import yaml

project_path = Path(__file__).resolve().parents[1]
data_path = project_path / "data"

# Load configuration to use only training flights
with open(project_path / "configs/default.yml", encoding="utf-8") as file:
    config = yaml.safe_load(file)

train_ids = config["train_data_ids"]

# Store annotation counts and total durations
maneuver_counts = Counter()
maneuver_durations = defaultdict(float)

for flight_id in train_ids:

    with open(data_path / f"{flight_id}.json", encoding="utf-8") as file:
        annotations = json.load(file)["annotations"]

    for maneuver in annotations:

        name = maneuver["comment"].strip().lower()

        start = pd.to_datetime(maneuver["startTimestamp"], utc=True)
        end = pd.to_datetime(maneuver["endTimeStamp"], utc=True)

        duration = (end - start).total_seconds()

        maneuver_counts[name] += 1
        maneuver_durations[name] += duration

# Create summary table
results = []

for name in sorted(maneuver_counts):
    results.append({
        "Maneuver": name,
        "Count": maneuver_counts[name],
        "Total duration [s]": round(maneuver_durations[name], 1)
    })

summary = pd.DataFrame(results)

print("\nTraining data annotation summary:")
print(summary.to_string(index=False))

print("\nTotal annotations:", summary["Count"].sum())

# Save results
output_path = project_path / "plots"
output_path.mkdir(exist_ok=True)

summary.to_csv(
    output_path / "annotation_summary.csv",
    index=False
)

# Visualize class distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(summary["Maneuver"], summary["Count"])

plt.ylabel("Number of annotated maneuvers")
plt.title("Maneuver distribution in training data")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(output_path / "class_distribution.png", dpi=200)
plt.show()
