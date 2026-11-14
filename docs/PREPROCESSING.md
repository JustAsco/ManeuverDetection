# Preprocessing – Maneuver Detection

## 1. Overview

This preprocessing pipeline converts raw helicopter flight recordings into a feature dataset for maneuver classification.

The pipeline was successfully executed on all 13 training flights.

Current output:
- 9,023 training windows
- 8 numerical features per window
- 1 maneuver label per window
- Separate metadata containing flight IDs and window timestamps

The six reserved test flights are not processed by default.

## 2. Selected signals

Four signals were selected for the initial baseline:

| Signal | Purpose |
|---|---|
| Aircraft Height [m] | Altitude and vertical flight behavior |
| Aircraft VerticalSpeed [m/s] | Distinguishing climb, descent and hover |
| Aircraft GroundSpeed [m/s] | Distinguishing hover from forward flight |
| Aircraft Bank [rad] | Identifying lateral aircraft attitude during turns |

These signals were selected based on their physical relevance and an initial visualization of a training flight.

The selection is not optimized yet. Additional signals, especially angular velocity, pitch and rotor-related measurements, could improve maneuver classification.

## 3. Missing values

The raw data contains many missing values because unchanged measurements are not necessarily recorded again.

Missing measurements are therefore filled using forward fill (`ffill`) within each individual flight.

All 13 training flights were successfully processed without remaining missing values in the four selected signals.

## 4. Maneuver labeling

Ground-truth maneuvers are obtained from the JSON annotation files.

Maneuver names are normalized using `strip().lower()` to handle differences in capitalization.

Measurements outside annotated maneuver intervals are assigned class 0 (`No Maneuver`).

Annotation intervals are sorted chronologically and checked for overlaps.

The annotation timestamps are interpreted as UTC.

## 5. Windowing and feature extraction

The current parameters are:

- Window size: 5.0 seconds
- Stride: 2.5 seconds
- Window overlap: 50%

For each selected signal, the mean and standard deviation are calculated.

This results in eight features per window.

### Time-weighted window labels

The maneuver label of each window is determined by the total duration of each class within that window.

For each annotation, the temporal overlap with the current window is calculated.

The class covering the largest duration is selected as the window label.

Unannotated time within a window is assigned to `No Maneuver`.

This method was selected instead of the statistical mode of measurement labels because the raw data does not have a perfectly uniform sampling rate.

## 6. Generated files

Running:

`python -m maneuver_detection.preprocess`

generates:

- `data/train.parquet`: numerical features and maneuver labels
- `data/train_window_metadata.csv`: flight IDs, window start/end timestamps and label coverage
- `data/train_flight_ids.csv`: flight IDs in the same row order as the features

The metadata must remain aligned with the corresponding feature rows.

The test dataset can be created separately using `create_test_set()` or the `--include-test` option, but it is reserved for final evaluation.

## 7. Validation and data leakage

Model validation must be performed using complete flights rather than randomly selected windows.

Overlapping windows from the same flight are highly correlated.

Use a group-based splitting method, such as `GroupKFold` or `GroupShuffleSplit`, with `Flight_ID` as the group variable.

Flight IDs and timestamps are metadata and must not be used as classifier input features.

## 8. Known limitations and future improvements

- Only four aircraft signals are currently used.
- The feature statistics are calculated across measurement rows, not weighted by time duration.
- Window size and stride have not been optimized.
- Windows containing multiple classes are assigned the longest-duration class.
- Unannotated intervals are assumed to represent `No Maneuver`.
- Absolute altitude may contain flight-specific information rather than purely maneuver-related information.
- The final maneuver-detection metric, including the 80% duration threshold, still needs to be evaluated.

## 9. Test results

- 13/13 training flights processed successfully
- 9,023 windows generated
- 0 missing feature values
- All eight classes present
- Metadata aligned with feature rows
- Window duration and chronological order checked

This pipeline provides an initial reproducible baseline for model development. It is not yet an optimized maneuver detector.