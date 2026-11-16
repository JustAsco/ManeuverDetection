# Preprocessing – Maneuver Detection

## 1. Overview

This preprocessing pipeline converts raw helicopter flight recordings into a feature dataset for maneuver classification.

The pipeline was successfully executed on all 13 training flights.

Current output:
- 9,023 training windows
- 32 numerical features per window
- 1 maneuver label per window
- Separate metadata containing flight IDs and window timestamps

The six reserved test flights are not processed by default.

## 2. Selected signals

| Signal | Purpose |
|---|---|
| Aircraft Height [m] | On ground vs. hover vs. flight, vertical trend |
| Aircraft VerticalSpeed [m/s] | Climb, descent, hover |
| Aircraft GroundSpeed [m/s] | Hover vs. forward flight |
| Aircraft Bank [rad] | Lateral attitude during turns |
| Aircraft Pitch [rad] | Nose attitude (climb vs. level vs. descent) |
| Aircraft AngularVelocity [rad/s] | Rotation rates (3D vector, expanded to `_0`, `_1`, `_2`) |
| Aircraft TrueHeading [rad] | Turn rate (heading is unwrapped first, so 2π → 0 is not a jump) |
| Aircraft RotorRotationSpeed [rad/s] | Rotor behaviour, especially in autorotation |
| Instrument EngineTorqueFraction [0-1] | Engine power; drops to near zero in autorotation |

No collective lever position is recorded in the data (only throttle and rotor brake have continuous control data), so engine torque is used as a proxy for collective.

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

Per window the following features are computed:

- For most signals: mean, standard deviation and slope (change per second from first to last sample).
- Height: slope, standard deviation, mean of `log(1 + height)` and `OnGround_Fraction` (share of samples below 0.3 m). The absolute height mean was dropped because it differs strongly between flights (median 5 m to 500 m) and encoded the flight rather than the maneuver. The on-ground fraction separates hover (about 0.5–16 m) from standing on the ground (about −0.1–0 m).
- Heading: only the turn rate (slope of the unwrapped heading); the absolute heading carries no maneuver information.

This results in 32 features per window.

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

- The feature statistics are calculated across measurement rows, not weighted by time duration.
- Window size and stride have not been optimized.
- Windows containing multiple classes are assigned the longest-duration class.
- Unannotated intervals are assumed to represent `No Maneuver`.
- No collective lever position is available; engine torque is used as a proxy.
- `no maneuver` has the lowest window recall (0.66 with LDA); it mixes ground time, transitions and unannotated flight.

## 9. Test results

- 13/13 training flights processed successfully
- 9,023 windows generated
- 0 missing feature values
- All eight classes present (autorotation only in 7 of 13 flights)
- Metadata aligned with feature rows
- Window duration and chronological order checked

Grouped 5-fold cross-validation with LDA (`train.py`), before → after the feature update:

| Metric | 8 features | 32 features |
|---|---|---|
| Maneuver recall | 0.44 | 0.46 |
| Maneuver recall (excl. no maneuver) | 0.33 | 0.63 |
| Precision | 0.57 | 0.71 |
| Accuracy | 0.62 | 0.69 |
| Hover window recall | 0.00 | 0.98 |

This pipeline provides a reproducible baseline for model development. It is not yet an optimized maneuver detector.