
from pathlib import Path
from collections import Counter
import json
import pandas as pd

# Projektverzeichnis
project_path = Path(__file__).resolve().parents[1]
data_path = project_path / "data"

# Ein Trainingsflug
flight_id = "0bbf0c4e-fb3c-4213-bff8-ef21ee5ebf79"

# Messdaten laden
df = pd.read_parquet(data_path / f"{flight_id}.parquet")

# Ground-Truth-Annotationen laden
with open(data_path / f"{flight_id}.json", encoding="utf-8") as file:
    annotations = json.load(file)["annotations"]

# Signalbeschreibungen laden
with open(data_path / "StateDescriptions.json", encoding="utf-8") as file:
    state_descriptions = json.load(file)

# Datenstruktur untersuchen
print("Shape:", df.shape)
print("Index:", df.index.names)
print("Chronologically sorted:", df.index.is_monotonic_increasing)

print("\nFirst five rows and columns:")
print(df.iloc[:5, :5])

print("\nFirst ten column names:")
print(list(df.columns[:10]))

# Zeitstempel untersuchen
time = pd.to_datetime(
    df.index.get_level_values("TimeStamp")
)

print("\nFirst timestamp:", time.min())
print("Last timestamp:", time.max())

# Manöver untersuchen
print("\nNumber of annotations:", len(annotations))

for maneuver in annotations:
    print(
        maneuver["comment"],
        maneuver["startTimestamp"],
        maneuver["endTimeStamp"]
    )

# Unterschiedliche Schreibweisen feststellen
labels = Counter(
    maneuver["comment"] for maneuver in annotations
)

print("\nManeuver names:", labels)

# Verfügbarkeit ausgewählter Messsignale
column_ids = {str(column) for column in df.columns}

selected_signals = [
    "Aerofly_Out_Aircraft_Height",
    "Aerofly_Out_Aircraft_VerticalSpeed",
    "Aerofly_Out_Aircraft_GroundSpeed",
    "Aerofly_Out_Aircraft_Bank",
]

print("\nSelected signals:")

for signal in selected_signals:
    info = state_descriptions[signal]
    print(
        signal,
        "ID:", info["stateId"],
        "Unit:", info["unit"],
        "Available:", str(info["stateId"]) in column_ids
    )






# Convert numeric column names to readable signal names
id_to_name = {
    info["stateId"]: name
    for name, info in state_descriptions.items()
    if "stateId" in info
}

df_named = df.rename(columns=id_to_name)

# Select our four signals
selected_columns = [
    "Aerofly_Out_Aircraft_Height",
    "Aerofly_Out_Aircraft_VerticalSpeed",
    "Aerofly_Out_Aircraft_GroundSpeed",
    "Aerofly_Out_Aircraft_Bank"
]

df_selected = df_named[selected_columns].copy()

print("\nSelected data shape:", df_selected.shape)
print(df_selected.head())








# Check missing values before preprocessing
print("\nMissing values before:")
print(df_selected.isna().sum())

# Apply missing value handling
df_clean = df_selected.sort_index().ffill()

# Check missing values after preprocessing
print("\nMissing values after:")
print(df_clean.isna().sum())

# Show the first rows of the cleaned dataset
print("\nFirst rows after preprocessing:")
print(df_clean.head())

# Simple validation
assert df_clean.isna().sum().sum() == 0

print("\nMissing value handling successful!")







import matplotlib.pyplot as plt
import numpy as np

# Messgrössen, die wir untersuchen möchten
signals = [
    ("Aerofly_Out_Aircraft_Height", "Height [m]"),
    ("Aerofly_Out_Aircraft_VerticalSpeed", "Vertical Speed [m/s]"),
    ("Aerofly_Out_Aircraft_GroundSpeed", "Ground Speed [m/s]"),
    ("Aerofly_Out_Aircraft_Bank", "Bank Angle [deg]")
]

# Zeit aus dem MultiIndex auslesen
time = df.index.get_level_values("TimeStamp")

# Vier Diagramme untereinander
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

for i, (signal_name, ylabel) in enumerate(signals):

    # Numerische Signal-ID aus StateDescriptions holen
    signal_id = state_descriptions[signal_name]["stateId"]

    # Signal auswählen und fehlende Werte auffüllen
    raw_signal = pd.to_numeric(df[signal_id], errors="coerce")
    signal = raw_signal.ffill()

    # Bankwinkel von rad in Grad umrechnen
    if signal_name == "Aerofly_Out_Aircraft_Bank":
        signal = np.degrees(signal)

    # Einige Informationen ausgeben
    print("\n", signal_name)
    print("Valid raw values:", raw_signal.notna().sum())
    print("Missing after ffill:", signal.isna().sum())
    print("Min:", signal.min())
    print("Max:", signal.max())

    # Nur jeden 50. Datenpunkt zeichnen
    axes[i].plot(time[::50], signal.iloc[::50], linewidth=1)

    axes[i].set_ylabel(ylabel)
    axes[i].grid(True)

    # Manöverbereiche markieren
    for maneuver in annotations:
        start = pd.to_datetime(maneuver["startTimestamp"], utc=True)
        end = pd.to_datetime(maneuver["endTimeStamp"], utc=True)

        axes[i].axvspan(start, end, color="orange", alpha=0.2)

# Namen der Manöver nur im obersten Plot darstellen
for maneuver in annotations:
    start = pd.to_datetime(maneuver["startTimestamp"], utc=True)
    end = pd.to_datetime(maneuver["endTimeStamp"], utc=True)

    middle = start + (end - start) / 2

    axes[0].text(
        middle, 0.95, maneuver["comment"],
        transform=axes[0].get_xaxis_transform(),
        rotation=90, ha="center", va="top", fontsize=8
    )

axes[0].set_title("Flight Data Analysis")
axes[-1].set_xlabel("Time")

fig.autofmt_xdate()
plt.tight_layout()

plot_path = project_path / "plots"
plot_path.mkdir(exist_ok=True)

plt.savefig(plot_path / "flight_signals.png", dpi=200)
plt.show()









# Prepare cleaned data for label assignment
df_labeled = df_clean.copy()

# Remove FrameCounter and timezone, as done in the template
df_labeled.index = (
    df_labeled.index.get_level_values("TimeStamp")
    .tz_localize(None)
)

# Use the same class IDs as in default.yml
label2id = {
    "no maneuver": 0,
    "hover": 1,
    "climb": 2,
    "descent": 3,
    "straight and level": 4,
    "right turn": 5,
    "left turn": 6,
    "autorotation": 7
}

# Initially, every measurement is No Maneuver
df_labeled["Maneuver_Label"] = 0

for maneuver in annotations:

    start = pd.Timestamp(maneuver["startTimestamp"])
    end = pd.Timestamp(maneuver["endTimeStamp"])

    name = maneuver["comment"].strip().lower()
    label_id = label2id[name]

    mask = (df_labeled.index >= start) & (df_labeled.index < end)

    df_labeled.loc[mask, "Maneuver_Label"] = label_id

# Show class distribution
print("\nLabel distribution:")
print(df_labeled["Maneuver_Label"].value_counts().sort_index())

# Check whether unexpected labels exist
assert df_labeled["Maneuver_Label"].isin(label2id.values()).all()

print("\nLabel assignment successful!")








# Check the first Hover maneuver
first_hover = annotations[0]

start = pd.Timestamp(first_hover["startTimestamp"])
end = pd.Timestamp(first_hover["endTimeStamp"])

hover_data = df_labeled.loc[
    (df_labeled.index >= start) &
    (df_labeled.index < end)
]

print("\nFirst Hover interval:")
print("Number of measurements:", len(hover_data))
print("Labels:", hover_data["Maneuver_Label"].unique())

assert len(hover_data) > 0
assert (hover_data["Maneuver_Label"] == 1).all()

print("Hover labels are correct!")








# Check all annotated maneuvers
print("\nChecking all maneuver intervals:")

for maneuver in annotations:

    start = pd.Timestamp(maneuver["startTimestamp"])
    end = pd.Timestamp(maneuver["endTimeStamp"])

    maneuver_name = maneuver["comment"].strip().lower()
    expected_label = label2id[maneuver_name]

    # Select all measurements in this time interval
    interval = df_labeled.loc[
        (df_labeled.index >= start) &
        (df_labeled.index < end)
    ]

    # Check whether all labels are correct
    correct = (
        len(interval) > 0
        and (interval["Maneuver_Label"] == expected_label).all()
    )

    print(
        maneuver_name,
        "| Measurements:", len(interval),
        "| Correct:", correct
    )

    assert correct, f"Incorrect labels for {maneuver_name}"

print("\nAll maneuver intervals checked successfully!")
