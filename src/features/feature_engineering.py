import heapq
import json

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.config import (
    TRAIN_DATA_FILE,
    VALIDATION_DATA_FILE,
    TEST_DATA_FILE,
    FEATURES_DIR,
    TRAIN_FEATURES_FILE,
    VALIDATION_FEATURES_FILE,
    TEST_FEATURES_FILE,
    HISTORICAL_FEATURE_REPORT,
    METADATA_FILE,
    DEFAULT_TEST_DURATION,
    MIN_ESTIMATED_DURATION,
    FAILURE_PRIOR_ALPHA,
    FAILURE_PRIOR_BETA,
    UNKNOWN_HISTORY_DAYS,
)


HISTORICAL_COLUMNS = [
    "historical_executions",
    "historical_failures",
    "historical_failure_rate",
    "smoothed_failure_rate",
    "historical_mean_duration",
    "historical_median_duration",
    "historical_std_duration",
    "last_failure",
    "last_duration",
    "recent_failure_rate_5",
    "recent_failure_rate_10",
    "consecutive_failures",
    "days_since_last_execution",
    "days_since_last_failure",
    "is_new_test",
    "estimated_duration_seconds",
]


class TestHistory:

    def __init__(self):
        self.executions = 0
        self.failures = 0
        self.durations = []
        self.recent_outcomes = deque(maxlen=10)
        self.last_failure = 0
        self.last_duration = DEFAULT_TEST_DURATION
        self.last_execution_time = None
        self.last_failure_time = None
        self.consecutive_failures = 0

    def update(self, failed, duration, timestamp):
        self.executions += 1
        self.failures += failed

        self.durations.append(duration)
        self.recent_outcomes.append(failed)

        self.last_failure = failed
        self.last_duration = duration
        self.last_execution_time = timestamp

        if failed == 1:
            self.last_failure_time = timestamp
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

    def get_features(self, timestamp, fallback_duration):
        if self.executions > 0:
            failure_rate = (
                self.failures / self.executions
            )

            mean_duration = float(
                np.mean(self.durations)
            )

            median_duration = float(
                np.median(self.durations)
            )

            std_duration = float(
                np.std(self.durations)
            )

            recent = list(self.recent_outcomes)

            recent_rate_5 = float(
                np.mean(recent[-5:])
            )

            recent_rate_10 = float(
                np.mean(recent[-10:])
            )

            last_duration = self.last_duration

        else:
            failure_rate = 0.0
            mean_duration = fallback_duration
            median_duration = fallback_duration
            std_duration = 0.0
            recent_rate_5 = 0.0
            recent_rate_10 = 0.0
            last_duration = fallback_duration

        smoothed_rate = (
            self.failures + FAILURE_PRIOR_ALPHA
        ) / (
            self.executions
            + FAILURE_PRIOR_ALPHA
            + FAILURE_PRIOR_BETA
        )

        if self.last_execution_time is None:
            days_since_execution = UNKNOWN_HISTORY_DAYS
        else:
            days_since_execution = (
                timestamp - self.last_execution_time
            ).total_seconds() / 86400

        if self.last_failure_time is None:
            days_since_failure = UNKNOWN_HISTORY_DAYS
        else:
            days_since_failure = (
                timestamp - self.last_failure_time
            ).total_seconds() / 86400

        return {
            "historical_executions": self.executions,
            "historical_failures": self.failures,
            "historical_failure_rate": failure_rate,
            "smoothed_failure_rate": smoothed_rate,
            "historical_mean_duration": mean_duration,
            "historical_median_duration": median_duration,
            "historical_std_duration": std_duration,
            "last_failure": self.last_failure,
            "last_duration": last_duration,
            "recent_failure_rate_5": recent_rate_5,
            "recent_failure_rate_10": recent_rate_10,
            "consecutive_failures": self.consecutive_failures,
            "days_since_last_execution": days_since_execution,
            "days_since_last_failure": days_since_failure,
            "is_new_test": int(self.executions == 0),
            "estimated_duration_seconds": max(
                median_duration,
                MIN_ESTIMATED_DURATION,
            ),
        }


class HistoricalFeatureGenerator:

    def __init__(self):
        self.histories = defaultdict(TestHistory)
        self.global_duration_sum = 0.0
        self.global_execution_count = 0
        self.pending = []
        self.sequence = 0

    def get_fallback_duration(self):
        if self.global_execution_count == 0:
            return DEFAULT_TEST_DURATION

        return max(
            self.global_duration_sum
            / self.global_execution_count,
            MIN_ESTIMATED_DURATION,
        )

    def release_completed_builds(self, timestamp):
        while self.pending and self.pending[0][0] < timestamp:
            completion_time, _, records = heapq.heappop(
                self.pending
            )

            for test_id, failed, duration in records:
                self.histories[test_id].update(
                    failed,
                    duration,
                    completion_time,
                )

                self.global_duration_sum += duration
                self.global_execution_count += 1

    def generate(self, dataframe, partition_name):
        dataframe = dataframe.sort_values(
            ["timestamp", "build_key", "run_id", "test_id"]
        ).reset_index(drop=True)

        feature_frames = []

        total_builds = dataframe["build_key"].nunique()
        processed_builds = 0

        for timestamp, batch in dataframe.groupby(
            "timestamp",
            sort=True,
        ):
            self.release_completed_builds(timestamp)

            fallback_duration = self.get_fallback_duration()

            features = [
                self.histories[test_id].get_features(
                    timestamp,
                    fallback_duration,
                )
                for test_id in batch["test_id"]
            ]

            feature_frame = pd.DataFrame(
                features,
                index=batch.index,
            )

            combined = pd.concat(
                [batch, feature_frame],
                axis=1,
            )

            feature_frames.append(combined)

            for _, build in batch.groupby("build_key"):
                completion_time = build[
                    "completion_timestamp"
                ].iloc[0]

                records = list(
                    build[
                        [
                            "test_id",
                            "failed",
                            "duration_seconds",
                        ]
                    ].itertuples(
                        index=False,
                        name=None,
                    )
                )

                heapq.heappush(
                    self.pending,
                    (
                        completion_time,
                        self.sequence,
                        records,
                    ),
                )

                self.sequence += 1
                processed_builds += 1

            if (
                processed_builds % 20 == 0
                or processed_builds == total_builds
            ):
                print(
                    f"{partition_name}: "
                    f"{processed_builds}/{total_builds} builds"
                )

        return pd.concat(
            feature_frames,
            ignore_index=True,
        )


def load_completion_metadata():
    metadata = pd.read_csv(
        METADATA_FILE,
        usecols=[
            "project",
            "pr_name",
            "build_id",
            "build_duration",
        ],
    )

    metadata["build_key"] = (
        metadata["project"].astype(str)
        + ":"
        + metadata["pr_name"].astype(str)
        + ":"
        + metadata["build_id"].astype(str)
    )

    metadata["build_duration"] = pd.to_numeric(
        metadata["build_duration"],
        errors="raise",
    )

    if not np.isfinite(
        metadata["build_duration"]
    ).all():
        raise ValueError("Invalid build durations.")

    if metadata["build_duration"].lt(0).any():
        raise ValueError("Negative build durations.")

    return (
        metadata.groupby("build_key")["build_duration"]
        .max()
    )


def load_partition(path, completion_metadata):
    dataframe = pd.read_csv(
        path,
        low_memory=False,
        keep_default_na=False,
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
        errors="raise",
    )

    dataframe["build_duration_raw"] = (
        dataframe["build_key"].map(completion_metadata)
    )

    if dataframe["build_duration_raw"].isna().any():
        raise ValueError("Missing build duration metadata.")

    dataframe["completion_timestamp"] = (
        dataframe["timestamp"]
        + pd.to_timedelta(
            dataframe["build_duration_raw"],
            unit="s",
        )
    )

    return dataframe


def validate_features(dataframe, name):
    if dataframe.empty:
        raise ValueError(f"{name} is empty.")

    if dataframe[HISTORICAL_COLUMNS].isna().any().any():
        raise ValueError(
            f"{name} contains missing historical features."
        )

    if not np.isfinite(
        dataframe[HISTORICAL_COLUMNS].to_numpy()
    ).all():
        raise ValueError(
            f"{name} contains non-finite features."
        )

    if (
        dataframe["historical_failures"]
        > dataframe["historical_executions"]
    ).any():
        raise ValueError("Invalid historical failure counts.")

    probability_columns = [
        "historical_failure_rate",
        "smoothed_failure_rate",
        "recent_failure_rate_5",
        "recent_failure_rate_10",
    ]

    for column in probability_columns:
        if not dataframe[column].between(0, 1).all():
            raise ValueError(
                f"Invalid probability in {column}"
            )

    if dataframe["estimated_duration_seconds"].le(0).any():
        raise ValueError("Invalid estimated duration.")

    new_tests = dataframe["is_new_test"].eq(1)

    if dataframe.loc[
        new_tests,
        "historical_executions",
    ].ne(0).any():
        raise ValueError("Invalid new-test history.")

    print(f"{name} feature validation passed.")


def generate_statistics(dataframe):
    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "failed_tests": int(
            dataframe["failed"].sum()
        ),
        "historical_features": len(HISTORICAL_COLUMNS),
        "cold_start_rows": int(
            dataframe["is_new_test"].sum()
        ),
        "mean_historical_executions": float(
            dataframe["historical_executions"].mean()
        ),
        "mean_historical_failure_rate": float(
            dataframe["historical_failure_rate"].mean()
        ),
        "mean_estimated_duration": float(
            dataframe["estimated_duration_seconds"].mean()
        ),
        "median_estimated_duration": float(
            dataframe["estimated_duration_seconds"].median()
        ),
        "missing_historical_values": int(
            dataframe[HISTORICAL_COLUMNS]
            .isna()
            .sum()
            .sum()
        ),
    }


def main():
    print("=" * 60)
    print("QuantCI - Historical Feature Engineering")
    print("=" * 60)

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    completion_metadata = load_completion_metadata()

    generator = HistoricalFeatureGenerator()

    partitions = [
        ("Training", TRAIN_DATA_FILE, TRAIN_FEATURES_FILE),
        (
            "Validation",
            VALIDATION_DATA_FILE,
            VALIDATION_FEATURES_FILE,
        ),
        ("Testing", TEST_DATA_FILE, TEST_FEATURES_FILE),
    ]

    report = {}

    total_rows = 0
    total_failures = 0

    for name, input_path, output_path in partitions:
        print(f"\nProcessing {name} dataset")

        dataframe = load_partition(
            input_path,
            completion_metadata,
        )

        features = generator.generate(
            dataframe,
            name,
        )

        validate_features(
            features,
            name,
        )

        features.to_csv(
            output_path,
            index=False,
        )

        report[name.lower()] = generate_statistics(
            features
        )

        total_rows += len(features)

        total_failures += int(
            features["failed"].sum()
        )

        print(f"Saved: {output_path}")

    report["total_rows"] = total_rows
    report["total_failures"] = total_failures
    report["feature_names"] = HISTORICAL_COLUMNS
    report["evaluation_mode"] = "walk_forward"

    HISTORICAL_FEATURE_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORICAL_FEATURE_REPORT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nFeature Engineering Summary:")
    print(json.dumps(report, indent=4))

    print("\n" + "=" * 60)
    print("HISTORICAL FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()