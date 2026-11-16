"""Preprocess raw data files into feature dataframes."""

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

from maneuver_detection.config import Config

HEIGHT_COLUMN = "Aerofly_Out_Aircraft_Height"
HEADING_COLUMN = "Aerofly_Out_Aircraft_TrueHeading"

# Raw signals read from each flight
SELECTED_COLUMNS = [
    HEIGHT_COLUMN,
    "Aerofly_Out_Aircraft_VerticalSpeed",
    "Aerofly_Out_Aircraft_GroundSpeed",
    "Aerofly_Out_Aircraft_Bank",
    "Aerofly_Out_Aircraft_Pitch",
    "Aerofly_Out_Aircraft_AngularVelocity",  # 3D vector, expanded to _0, _1, _2
    HEADING_COLUMN,
    "Aerofly_Out_Aircraft_RotorRotationSpeed",
    # No collective lever position is recorded; engine torque is used as proxy
    "Aerofly_Out_Instrument_EngineTorqueFraction",
]

# Height [m] below which the aircraft is considered to be on the ground.
# On-ground windows sit at about -0.1..0 m, hover starts at about 0.5 m.
GROUND_HEIGHT_THRESHOLD = 0.3


class Preprocessor:
    """Preprocess raw data files into feature dataframes."""

    def __init__(self, config: Config) -> None:
        """Initialize.

        Args:
            config: Project configuration.
        """
        self._config = config
        state_descriptions = pd.read_json(self._config.state_description_path)
        self._state_id_to_name_mapper = {
            value: column for column, value in state_descriptions.loc["stateId"].dropna().items()
        }

    def preprocess(self, data_ids: list[str]) -> pd.DataFrame:

        features, metadata = self._preprocess_with_metadata(data_ids)

        return features

    
    
    def _preprocess_with_metadata(
        self, data_ids: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        all_features = []
        all_metadata = []

        for data_id in tqdm(data_ids):
            features, metadata = self._preprocess_file_with_metadata(data_id)

            all_features.append(features)
            all_metadata.append(metadata)

        if not all_features:
            raise ValueError("No flights provided")

        df_features = pd.concat(all_features, ignore_index=True)
        df_metadata = pd.concat(all_metadata, ignore_index=True)

        return df_features, df_metadata



    def preprocess_file(self, data_id: str) -> pd.DataFrame:

        features, metadata = self._preprocess_file_with_metadata(data_id)

        return features


    def _preprocess_file_with_metadata(
        self, data_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        annotations = self._read_annotations(data_id)

        df = self._read_parquet_file(data_id)
        df = self._rename_columns(df)
        df = self._extract_relevant_columns(df)
        df = _convert_dtypes(df)
        df = self._fix_missing_values(df)
        df = _fix_index(df)
        df = _expand_vectors(df)
        df = _unwrap_heading(df)
        df = self._assign_maneuver_labels(df, annotations)

        features, metadata = self._summarize_features_in_blocks(
            df, annotations, data_id
        )

        return features, metadata


    def create_train_set(self) -> pd.DataFrame:

        features, metadata = self._preprocess_with_metadata(
            self._config.train_data_ids
        )

        self._save_dataset(
            features,
            metadata,
            self._config.train_data_path,
            "train"
        )

        return features


    def create_test_set(self) -> pd.DataFrame:

        features, metadata = self._preprocess_with_metadata(
            self._config.test_data_ids
        )

        self._save_dataset(
            features,
            metadata,
            self._config.test_data_path,
            "test"
        )

        return features



    def _save_dataset(
        self,
        features: pd.DataFrame,
        metadata: pd.DataFrame,
        path,
        name: str
    ) -> None:

        # Features and metadata must have equal length
        assert len(features) == len(metadata)

        # Check missing and invalid values
        assert not features.isna().any().any()

        feature_columns = features.columns.drop("Maneuver_Label")

        assert np.isfinite(
            features[feature_columns].to_numpy(dtype=float)
        ).all()

        # Save feature data
        features.to_parquet(path, index=False)

        # Save corresponding time information
        metadata.to_csv(
            self._config.data_path / f"{name}_window_metadata.csv",
            index=False
        )

        # Save flight IDs for compatibility with earlier scripts
        metadata[["Flight_ID"]].to_csv(
            self._config.data_path / f"{name}_flight_ids.csv",
            index=False
        )

        print(f"Saved {len(features)} windows to {path}")



    def _read_parquet_file(self, data_id: str) -> pd.DataFrame:
        return pd.read_parquet(self._config.data_path / f"{data_id}.parquet")

    def _read_annotations(self, data_id: str) -> pd.DataFrame:
        return pd.read_json(self._config.data_path / f"{data_id}.json")["annotations"]

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self._state_id_to_name_mapper)

    
    def _extract_relevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        selected_columns = SELECTED_COLUMNS

        # Check if all columns exist
        for column in selected_columns:
            if column not in df.columns:
                raise ValueError(f"Missing column: {column}")

        # Return only the selected signals
        return df[selected_columns].copy()


    
    def _fix_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Make sure the measurements are sorted chronologically
        df = df.sort_index().copy()

        # Fill missing values with the last valid measurement
        df = df.ffill()

        # Check if there are still missing values
        missing_values = df.isna().sum()

        if missing_values.sum() > 0:
            print("Remaining missing values:")
            print(missing_values[missing_values > 0])

            raise ValueError("Some missing values could not be filled.")

        return df


    
    
    def _assign_maneuver_labels(
        self, df: pd.DataFrame, annotations: pd.Series
    ) -> pd.DataFrame:

        df = df.copy()

        # Initialize all measurements as No Maneuver
        df["Maneuver_Label"] = self._config.label2id["no maneuver"]

        # Assign labels from annotation intervals
        intervals = self._annotation_intervals(annotations)

        for start, end, label_id in intervals:

            mask = (df.index >= start) & (df.index < end)

            df.loc[mask, "Maneuver_Label"] = label_id

        return df





    def _annotation_intervals(self, annotations: pd.Series) -> list:
        intervals = []

        for maneuver in annotations:

            # Normalize maneuver name
            name = maneuver["comment"].strip().lower()

            if name not in self._config.label2id:
                raise ValueError(f"Unknown maneuver: {name}")

            label_id = self._config.label2id[name]

            # Convert timestamps to UTC and remove timezone information
            start = pd.to_datetime(
                maneuver["startTimestamp"], utc=True
            ).tz_localize(None)

            end = pd.to_datetime(
                maneuver["endTimeStamp"], utc=True
            ).tz_localize(None)

            if end <= start:
                raise ValueError(f"Invalid interval: {name}")

            intervals.append((start, end, label_id))

        # Sort annotations chronologically
        intervals.sort(key=lambda interval: interval[0])

        # Check if two annotations overlap
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise ValueError("Overlapping annotations found")

        return intervals




    
    
    def _summarize_features_in_blocks(
        self,
        df: pd.DataFrame,
        annotations: pd.Series,
        data_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        # Window parameters
        block_seconds = self._config.preprocess.block_size
        stride_seconds = self._config.preprocess.stride_size

        block_size = pd.Timedelta(seconds=block_seconds)
        stride_size = pd.Timedelta(seconds=stride_seconds)

        # Prepare maneuver intervals
        intervals = self._annotation_intervals(annotations)

        # Select signals, excluding the target label
        signal_columns = df.columns.drop("Maneuver_Label")

        # Start and end of the recording
        time_index = df.index
        start = time_index[0]
        last_start = time_index[-1] - block_size

        feature_rows = []
        metadata_rows = []

        # Move through the recording in overlapping windows
        while start <= last_start:

            end = start + block_size

            # Find rows belonging to the current time window
            first_row = time_index.searchsorted(start, side="left")
            last_row = time_index.searchsorted(end, side="left")

            block = df.iloc[first_row:last_row]

            if len(block) >= 2:

                features = _window_features(block[signal_columns])

                # Calculate how long each maneuver occurs in the window
                durations = {
                    label: 0.0
                    for label in self._config.all_labels
                }

                annotated_seconds = 0.0

                for annotation_start, annotation_end, label_id in intervals:

                    # Intersection of window and annotation
                    overlap_start = max(start, annotation_start)
                    overlap_end = min(end, annotation_end)

                    if overlap_end > overlap_start:

                        seconds = (
                            overlap_end - overlap_start
                        ).total_seconds()

                        durations[label_id] += seconds
                        annotated_seconds += seconds

                # Time without annotation = No Maneuver
                no_maneuver_id = self._config.label2id["no maneuver"]

                durations[no_maneuver_id] = max(
                    0.0,
                    block_seconds - annotated_seconds
                )

                # Select the class with the longest duration
                window_label = max(durations, key=durations.get)

                features["Maneuver_Label"] = window_label

                feature_rows.append(features)

                # Save time information separately
                metadata_rows.append({
                    "Flight_ID": data_id,
                    "Window_Start": start.tz_localize("UTC"),
                    "Window_End": end.tz_localize("UTC"),
                    "Label_Coverage": (
                        durations[window_label] / block_seconds
                    )
                })

            # Move the window
            start += stride_size

        if not feature_rows:
            raise ValueError(f"No valid windows in flight {data_id}")

        df_features = pd.DataFrame(feature_rows)
        df_metadata = pd.DataFrame(metadata_rows)

        return df_features, df_metadata

    

    


    



def _window_features(block: pd.DataFrame) -> dict[str, float]:
    """Compute the features of one window.

    For most signals: mean, standard deviation and slope (change per second
    from the first to the last sample).

    Height: its absolute value differs strongly between flights (5 m to 500 m
    median), so only its slope, its std, the mean of log(1 + height) and the
    fraction of samples on the ground are used. The on-ground fraction
    separates hover (a few metres) from standing on the ground (0 m).

    Heading: absolute heading carries no maneuver information, so only the
    turn rate (slope of the unwrapped heading) is used.
    """
    duration = (block.index[-1] - block.index[0]).total_seconds()
    features = {}

    for column in block.columns:
        values = block[column].to_numpy(dtype=float)
        slope = (values[-1] - values[0]) / duration if duration > 0 else 0.0

        if column == HEADING_COLUMN:
            features[f"{column}_Rate"] = float(slope)
        elif column == HEIGHT_COLUMN:
            features[f"{column}_LogMean"] = float(np.log1p(np.clip(values, 0, None)).mean())
            features[f"{column}_Std"] = float(values.std())
            features[f"{column}_Slope"] = float(slope)
            features["OnGround_Fraction"] = float((values < GROUND_HEIGHT_THRESHOLD).mean())
        else:
            features[f"{column}_Mean"] = float(values.mean())
            features[f"{column}_Std"] = float(values.std())
            features[f"{column}_Slope"] = float(slope)

    return features


def _unwrap_heading(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the jump between 2*pi and 0 so the heading change is continuous
    df[HEADING_COLUMN] = np.unwrap(df[HEADING_COLUMN].to_numpy(dtype=float))
    return df


def _fix_index(df: pd.DataFrame) -> pd.DataFrame:
    # Drop FrameCounter level
    index = df.index.droplevel("FrameCounter")

    # Remove timezone information
    index = pd.to_datetime(index).tz_localize(None)
    df.index = index
    return df


def _convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return df.convert_dtypes()


def _expand_vectors(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if hasattr(df[column].iloc[0], "__len__"):
            vector_length = len(df[column].iloc[0])
            dtype = df[column].dtype
            if vector_length > 1 and dtype != "string":
                expanded = pd.DataFrame(
                    df[column].tolist(),
                    columns=[f"{column}_{i}" for i in range(vector_length)],
                    index=df.index,
                )

                df = pd.concat([df, expanded], axis="columns")
                df = df.drop(columns=[column])
    return df


if __name__ == "__main__":

    from maneuver_detection.config import get_default_config

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also preprocess the reserved test flights"
    )

    args = parser.parse_args()

    config = get_default_config()
    preprocessor = Preprocessor(config)

    # Always generate training data
    preprocessor.create_train_set()

    # Only process test data when explicitly requested
    if args.include_test:
        preprocessor.create_test_set()
    else:
        print("Test flights were not processed.")

