import heapq
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_repository"
)

SPLIT_DIR = DATA_DIR / "splits"

OUTPUT_DIR = DATA_DIR / "features"

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "multi_repository"
)

PROJECTS = ["karaf", "hbase"]

PARTITIONS = ["train", "validation", "test"]

DEFAULT_DURATION = 10.0

MIN_DURATION = 0.1

PRIOR_ALPHA = 1.0

PRIOR_BETA = 99.0

FEATURE_COLUMNS = [
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

REQUIRED_COLUMNS = [
    "project",
    "run_id",
    "build_key",
    "stage_id",
    "test_id",
    "timestamp",
    "completion_timestamp",
    "failed",
    "duration_seconds",
]


class TestHistory:
    def __init__(self):
        self.executions = 0
        self.failures = 0
        self.duration_sum = 0.0
        self.duration_squared_sum = 0.0
        self.lower_heap = []
        self.upper_heap = []
        self.recent_outcomes = deque(maxlen=10)
        self.last_failure = 0
        self.last_duration = DEFAULT_DURATION
        self.last_execution_time = None
        self.last_failure_time = None
        self.consecutive_failures = 0

    def update_median(self, duration):
        if (
            not self.lower_heap
            or duration <= -self.lower_heap[0]
        ):
            heapq.heappush(
                self.lower_heap,
                -duration,
            )
        else:
            heapq.heappush(
                self.upper_heap,
                duration,
            )

        if len(self.lower_heap) > len(
            self.upper_heap
        ) + 1:
            heapq.heappush(
                self.upper_heap,
                -heapq.heappop(self.lower_heap),
            )

        if len(self.upper_heap) > len(
            self.lower_heap
        ):
            heapq.heappush(
                self.lower_heap,
                -heapq.heappop(self.upper_heap),
            )

    def median_duration(self):
        if not self.lower_heap:
            return DEFAULT_DURATION

        if len(self.lower_heap) == len(
            self.upper_heap
        ):
            return (
                -self.lower_heap[0]
                + self.upper_heap[0]
            ) / 2.0

        return -self.lower_heap[0]

    def update(
        self,
        failed,
        duration,
        completion_time,
    ):
        if failed not in (0, 1):
            raise ValueError("Invalid outcome")

        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid duration")

        if (
            self.last_execution_time is not None
            and completion_time < self.last_execution_time
        ):
            raise ValueError(
                "Out-of-order historical update"
            )

        self.executions += 1
        self.failures += failed

        self.duration_sum += duration

        self.duration_squared_sum += (
            duration * duration
        )

        self.update_median(duration)

        self.recent_outcomes.append(failed)

        self.last_failure = failed

        self.last_duration = duration

        self.last_execution_time = completion_time

        if failed:
            self.last_failure_time = completion_time

            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

    def features(self, timestamp):
        if (
            self.last_execution_time is not None
            and self.last_execution_time >= timestamp
        ):
            raise ValueError(
                "Historical information leakage"
            )

        count = self.executions

        if count:
            mean = self.duration_sum / count

            variance = max(
                0.0,
                self.duration_squared_sum / count
                - mean * mean,
            )

            standard_deviation = math.sqrt(
                variance
            )

            failure_rate = self.failures / count
        else:
            mean = DEFAULT_DURATION
            standard_deviation = 0.0
            failure_rate = 0.0

        smoothed_rate = (
            self.failures + PRIOR_ALPHA
        ) / (
            count + PRIOR_ALPHA + PRIOR_BETA
        )

        recent = list(
            self.recent_outcomes
        )

        recent_five = recent[-5:]

        rate_five = (
            sum(recent_five) / len(recent_five)
            if recent_five
            else 0.0
        )

        rate_ten = (
            sum(recent) / len(recent)
            if recent
            else 0.0
        )

        days_since_execution = (
            (
                timestamp - self.last_execution_time
            ).total_seconds() / 86400
            if self.last_execution_time is not None
            else -1.0
        )

        days_since_failure = (
            (
                timestamp - self.last_failure_time
            ).total_seconds() / 86400
            if self.last_failure_time is not None
            else -1.0
        )

        median = self.median_duration()

        return [
            count,
            self.failures,
            failure_rate,
            smoothed_rate,
            mean,
            median,
            standard_deviation,
            self.last_failure,
            self.last_duration,
            rate_five,
            rate_ten,
            self.consecutive_failures,
            days_since_execution,
            days_since_failure,
            int(count == 0),
            max(MIN_DURATION, median),
        ]


def validate_data(dataframe):
    missing = set(REQUIRED_COLUMNS).difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if dataframe.empty:
        raise ValueError("Empty dataset")

    if dataframe[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Missing required values")

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )

    dataframe["completion_timestamp"] = pd.to_datetime(
        dataframe["completion_timestamp"],
        utc=True,
    )

    if not dataframe["failed"].isin([0, 1]).all():
        raise ValueError("Invalid failure labels")

    if not np.isfinite(
        dataframe["duration_seconds"]
    ).all():
        raise ValueError("Invalid durations")

    if dataframe["duration_seconds"].lt(0).any():
        raise ValueError("Negative duration")

    if (
        dataframe["completion_timestamp"]
        < dataframe["timestamp"]
    ).any():
        raise ValueError(
            "Invalid completion timestamp"
        )

    if dataframe.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError(
            "Duplicate test executions"
        )

    return dataframe


def history_key(row):
    return (
        row.project,
        row.stage_id,
        row.test_id,
    )


def generate_features(dataframe):
    dataframe = validate_data(dataframe)

    dataframe = dataframe.sort_values(
        [
            "timestamp",
            "build_key",
            "stage_id",
            "test_id",
        ]
    ).reset_index(drop=True)

    build_times = (
        dataframe.groupby("build_key")
        .agg(
            start=("timestamp", "min"),
            completion=(
                "completion_timestamp",
                "max",
            ),
        )
    )

    histories = {}

    pending = []

    sequence = 0

    features = np.empty(
        (
            len(dataframe),
            len(FEATURE_COLUMNS),
        ),
        dtype=np.float64,
    )

    processed = 0

    build_order = build_times.sort_values(
        "start"
    ).index

    groups = dataframe.groupby(
        "build_key",
        sort=False,
    )

    for build_key in build_order:
        timestamp = build_times.loc[
            build_key,
            "start",
        ]

        completion = build_times.loc[
            build_key,
            "completion",
        ]

        while pending and pending[0][0] < timestamp:
            (
                completed_at,
                _,
                key,
                failed,
                duration,
            ) = heapq.heappop(pending)

            if key not in histories:
                histories[key] = TestHistory()

            histories[key].update(
                failed,
                duration,
                completed_at,
            )

        group = groups.get_group(build_key)

        for row in group.itertuples():
            key = history_key(row)

            if key not in histories:
                histories[key] = TestHistory()

            features[row.Index] = histories[
                key
            ].features(timestamp)

        for row in group.itertuples():
            key = history_key(row)

            heapq.heappush(
                pending,
                (
                    completion,
                    sequence,
                    key,
                    int(row.failed),
                    float(row.duration_seconds),
                ),
            )

            sequence += 1

        processed += len(group)

        if processed % 50000 < len(group):
            print(
                f"Processed {processed:,}"
                f"/{len(dataframe):,} executions"
            )

    feature_frame = pd.DataFrame(
        features,
        columns=FEATURE_COLUMNS,
    )

    if not np.isfinite(features).all():
        raise ValueError(
            "Non-finite historical features"
        )

    return pd.concat(
        [
            dataframe,
            feature_frame,
        ],
        axis=1,
    )


def load_partition_assignments(project):
    assignments = {}

    for partition in PARTITIONS:
        path = (
            SPLIT_DIR
            / project
            / f"{partition}.csv"
        )

        dataframe = pd.read_csv(
            path,
            usecols=["build_key"],
        )

        for build_key in dataframe[
            "build_key"
        ].unique():
            if build_key in assignments:
                raise ValueError(
                    "Build assigned to multiple partitions"
                )

            assignments[build_key] = partition

    return assignments


def process_project(project):
    started = time.perf_counter()

    source = (
        DATA_DIR
        / f"{project}_normalized.csv"
    )

    dataframe = pd.read_csv(
        source,
        usecols=REQUIRED_COLUMNS,
    )

    if not dataframe["project"].eq(project).all():
        raise ValueError(
            "Unexpected project identity"
        )

    assignments = load_partition_assignments(
        project
    )

    dataframe = generate_features(
        dataframe
    )

    dataframe["partition"] = dataframe[
        "build_key"
    ].map(assignments)

    project_dir = OUTPUT_DIR / project

    project_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = []

    for partition in PARTITIONS:
        subset = dataframe[
            dataframe["partition"] == partition
        ].copy()

        if subset.empty:
            raise ValueError(
                f"Empty partition: {partition}"
            )

        output_file = (
            project_dir
            / f"{partition}_history.csv"
        )

        subset.to_csv(
            output_file,
            index=False,
        )

        reports.append({
            "project": project,
            "partition": partition,
            "rows": len(subset),
            "failures": int(
                subset["failed"].sum()
            ),
            "historical_features": len(
                FEATURE_COLUMNS
            ),
            "new_test_executions": int(
                subset["is_new_test"].sum()
            ),
            "mean_historical_executions": float(
                subset[
                    "historical_executions"
                ].mean()
            ),
            "mean_estimated_duration": float(
                subset[
                    "estimated_duration_seconds"
                ].mean()
            ),
            "missing_features": int(
                subset[
                    FEATURE_COLUMNS
                ].isna().sum().sum()
            ),
            "output_file": str(
                output_file.relative_to(BASE_DIR)
            ),
        })

    return {
        "project": project,
        "source_rows": len(dataframe),
        "retained_rows": sum(
            report["rows"]
            for report in reports
        ),
        "feature_count": len(FEATURE_COLUMNS),
        "history_identity": [
            "project",
            "stage_id",
            "test_id",
        ],
        "availability_rule": (
            "build_completion_strictly_before_current_build_start"
        ),
        "evaluation_mode": "walk_forward",
        "partitions": reports,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = []

    print("=" * 70)
    print("QUANTCI MULTI-REPOSITORY HISTORY")
    print("=" * 70)

    for project in PROJECTS:
        print(f"\nProcessing {project}")

        report = process_project(project)

        reports.append(report)

        print(
            json.dumps(report, indent=4)
        )

    summary = pd.DataFrame([
        partition
        for report in reports
        for partition in report["partitions"]
    ])

    summary.to_csv(
        RESULTS_DIR / "history_summary.csv",
        index=False,
    )

    report = {
        "projects": PROJECTS,
        "feature_columns": FEATURE_COLUMNS,
        "total_retained_rows": sum(
            item["retained_rows"]
            for item in reports
        ),
        "repositories": reports,
    }

    report_path = (
        RESULTS_DIR
        / "history_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nHistorical Feature Summary\n")

    print(
        summary[
            [
                "project",
                "partition",
                "rows",
                "failures",
                "historical_features",
                "new_test_executions",
                "missing_features",
            ]
        ].to_string(index=False)
    )

    print("\nReport:", report_path)

    print(
        "\nHISTORICAL FEATURE ENGINEERING COMPLETED"
    )


if __name__ == "__main__":
    main()