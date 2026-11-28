import json
import time
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

OUTPUT_DIR = DATA_DIR / "splits"

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "multi_repository"
)

PROJECTS = ["karaf", "hbase"]

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15

CHUNK_SIZE = 25000

REQUIRED_COLUMNS = [
    "project",
    "run_id",
    "build_key",
    "test_id",
    "timestamp",
    "completion_timestamp",
    "commit_sha",
    "failed",
    "duration_seconds",
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

    if dataframe[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Missing required values")

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
        errors="raise",
    )

    dataframe["completion_timestamp"] = pd.to_datetime(
        dataframe["completion_timestamp"],
        utc=True,
        errors="raise",
    )

    if (
        dataframe["completion_timestamp"]
        < dataframe["timestamp"]
    ).any():
        raise ValueError("Invalid completion timestamp")

    if dataframe.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError("Duplicate executions")

    if not dataframe["failed"].isin([0, 1]).all():
        raise ValueError("Invalid failure labels")

    if not np.isfinite(
        dataframe["duration_seconds"]
    ).all():
        raise ValueError("Invalid durations")

    if (
        dataframe["duration_seconds"] < 0
    ).any():
        raise ValueError("Negative durations")

    return dataframe


def create_build_table(dataframe):
    builds = (
        dataframe.groupby(
            "build_key",
            as_index=False,
        )
        .agg(
            timestamp=("timestamp", "min"),
            completion_timestamp=(
                "completion_timestamp",
                "max",
            ),
            executions=("test_id", "size"),
            failures=("failed", "sum"),
            stages=("run_id", "nunique"),
            commits=("commit_sha", "nunique"),
        )
        .sort_values(
            ["timestamp", "build_key"]
        )
        .reset_index(drop=True)
    )

    if (builds["commits"] != 1).any():
        raise ValueError(
            "Multiple commit identities within a build"
        )

    return builds


def create_partitions(builds):
    total = len(builds)

    if total < 10:
        raise ValueError("Insufficient builds")

    train_end = int(total * TRAIN_RATIO)

    validation_end = int(
        total * (
            TRAIN_RATIO + VALIDATION_RATIO
        )
    )

    validation_start = builds.iloc[
        train_end
    ]["timestamp"]

    test_start = builds.iloc[
        validation_end
    ]["timestamp"]

    if validation_start >= test_start:
        raise ValueError(
            "Invalid chronological boundaries"
        )

    train = builds[
        builds["timestamp"] < validation_start
    ].copy()

    validation = builds[
        (builds["timestamp"] >= validation_start)
        & (builds["timestamp"] < test_start)
    ].copy()

    test = builds[
        builds["timestamp"] >= test_start
    ].copy()

    excluded_train = train[
        train["completion_timestamp"]
        >= validation_start
    ].copy()

    excluded_validation = validation[
        validation["completion_timestamp"]
        >= test_start
    ].copy()

    train = train[
        train["completion_timestamp"]
        < validation_start
    ].copy()

    validation = validation[
        validation["completion_timestamp"]
        < test_start
    ].copy()

    excluded_train["original_partition"] = "train"

    excluded_validation[
        "original_partition"
    ] = "validation"

    excluded = pd.concat(
        [
            excluded_train,
            excluded_validation,
        ],
        ignore_index=True,
    )

    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, partition in partitions.items():
        if partition.empty:
            raise ValueError(
                f"Empty partition: {name}"
            )

    return partitions, excluded


def verify_partitions(partitions):
    train = partitions["train"]

    validation = partitions["validation"]

    test = partitions["test"]

    train_ids = set(train["build_key"])

    validation_ids = set(
        validation["build_key"]
    )

    test_ids = set(test["build_key"])

    if train_ids & validation_ids:
        raise ValueError(
            "Train-validation build overlap"
        )

    if train_ids & test_ids:
        raise ValueError(
            "Train-test build overlap"
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation-test build overlap"
        )

    if (
        train["completion_timestamp"].max()
        >= validation["timestamp"].min()
    ):
        raise ValueError(
            "Train-validation temporal overlap"
        )

    if (
        validation["completion_timestamp"].max()
        >= test["timestamp"].min()
    ):
        raise ValueError(
            "Validation-test temporal overlap"
        )

    return True


def summarize_partition(
    project,
    name,
    dataframe,
):
    failures = int(
        dataframe["failed"].sum()
    )

    return {
        "project": project,
        "partition": name,
        "rows": len(dataframe),
        "failures": failures,
        "passes": len(dataframe) - failures,
        "failure_rate": (
            failures / len(dataframe)
        ),
        "unique_builds": int(
            dataframe["build_key"].nunique()
        ),
        "unique_runs": int(
            dataframe["run_id"].nunique()
        ),
        "unique_tests": int(
            dataframe["test_id"].nunique()
        ),
        "unique_commits": int(
            dataframe["commit_sha"].nunique()
        ),
        "earliest_timestamp": str(
            dataframe["timestamp"].min()
        ),
        "latest_timestamp": str(
            dataframe["timestamp"].max()
        ),
        "latest_completion_timestamp": str(
            dataframe[
                "completion_timestamp"
            ].max()
        ),
    }


def export_partitions(
    source_file,
    project,
    partitions,
):
    project_dir = OUTPUT_DIR / project

    project_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments = {}

    for name, builds in partitions.items():
        for build_key in builds["build_key"]:
            assignments[build_key] = name

    output_files = {
        name: project_dir / f"{name}.csv"
        for name in partitions
    }

    temporary_files = {
        name: path.with_suffix(".tmp")
        for name, path in output_files.items()
    }

    for path in temporary_files.values():
        path.unlink(missing_ok=True)

    counts = {
        name: 0
        for name in partitions
    }

    for chunk in pd.read_csv(
        source_file,
        chunksize=CHUNK_SIZE,
    ):
        assignment = chunk[
            "build_key"
        ].map(assignments)

        for name, path in temporary_files.items():
            subset = chunk[
                assignment == name
            ]

            if subset.empty:
                continue

            subset.to_csv(
                path,
                mode="a",
                header=not path.exists(),
                index=False,
            )

            counts[name] += len(subset)

    for name, builds in partitions.items():
        expected = int(
            builds["executions"].sum()
        )

        if counts[name] != expected:
            raise ValueError(
                f"Export mismatch for {name}"
            )

    for name, path in temporary_files.items():
        path.replace(output_files[name])

    return output_files


def split_project(project):
    started = time.perf_counter()

    source_file = (
        DATA_DIR
        / f"{project}_normalized.csv"
    )

    if not source_file.exists():
        raise FileNotFoundError(source_file)

    dataframe = pd.read_csv(
        source_file,
        usecols=REQUIRED_COLUMNS,
    )

    dataframe = validate_data(
        dataframe
    )

    if not dataframe["project"].eq(project).all():
        raise ValueError(
            "Unexpected project identity"
        )

    builds = create_build_table(
        dataframe
    )

    partitions, excluded = create_partitions(
        builds
    )

    verify_partitions(
        partitions
    )

    output_files = export_partitions(
        source_file,
        project,
        partitions,
    )

    records = []

    commit_sets = {}

    for name, partition in partitions.items():
        subset = dataframe[
            dataframe["build_key"].isin(
                partition["build_key"]
            )
        ]

        records.append(
            summarize_partition(
                project,
                name,
                subset,
            )
        )

        commit_sets[name] = set(
            subset["commit_sha"]
        )

        if subset["failed"].nunique() < 2:
            print(
                f"WARNING: {project}/{name} "
                "does not contain both outcome classes"
            )

    excluded.to_csv(
        RESULTS_DIR
        / f"{project}_embargoed_builds.csv",
        index=False,
    )

    excluded_rows = int(
        excluded["executions"].sum()
    )

    retained_rows = sum(
        record["rows"]
        for record in records
    )

    if retained_rows + excluded_rows != len(dataframe):
        raise ValueError(
            "Partition accounting mismatch"
        )

    report = {
        "project": project,
        "original_rows": len(dataframe),
        "original_builds": len(builds),
        "retained_rows": retained_rows,
        "excluded_builds": len(excluded),
        "excluded_rows": excluded_rows,
        "excluded_failures": int(
            excluded["failures"].sum()
        ),
        "build_overlap": 0,
        "temporal_overlap": False,
        "commit_overlap": {
            "train_validation": len(
                commit_sets["train"]
                & commit_sets["validation"]
            ),
            "train_test": len(
                commit_sets["train"]
                & commit_sets["test"]
            ),
            "validation_test": len(
                commit_sets["validation"]
                & commit_sets["test"]
            ),
        },
        "partitions": records,
        "output_files": {
            name: str(
                path.relative_to(BASE_DIR)
            )
            for name, path in output_files.items()
        },
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    print(
        json.dumps(report, indent=4)
    )

    return report


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
    print("QUANTCI MULTI-REPOSITORY SPLITTING")
    print("=" * 70)

    for project in PROJECTS:
        print(f"\nProcessing {project}")

        reports.append(
            split_project(project)
        )

    summary = pd.DataFrame([
        partition
        for report in reports
        for partition in report["partitions"]
    ])

    summary.to_csv(
        RESULTS_DIR
        / "split_summary.csv",
        index=False,
    )

    report = {
        "dataset": "LRTS",
        "projects": PROJECTS,
        "split_strategy": "chronological_build_grouped",
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "temporal_embargo": True,
        "repositories": reports,
    }

    report_path = (
        RESULTS_DIR
        / "split_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nPartition Summary\n")

    print(
        summary[
            [
                "project",
                "partition",
                "rows",
                "failures",
                "failure_rate",
                "unique_builds",
                "unique_runs",
            ]
        ].to_string(index=False)
    )

    print("\nReport:", report_path)

    print(
        "\nMULTI-REPOSITORY SPLITTING COMPLETED"
    )


if __name__ == "__main__":
    main()