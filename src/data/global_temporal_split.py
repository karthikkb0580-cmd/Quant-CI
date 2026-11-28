import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "multi_repository"

OUTPUT_DIR = DATA_DIR / "global_splits"

RESULTS_DIR = BASE_DIR / "results" / "multi_repository" / "combined"

PROJECTS = ["karaf", "hbase"]

PARTITIONS = ["train", "validation", "test"]

TRAIN_PERCENT = 70

VALIDATION_PERCENT = 15

CHUNK_SIZE = 25000

REQUIRED_COLUMNS = [
    "project",
    "run_id",
    "build_key",
    "test_id",
    "timestamp",
    "completion_timestamp",
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

    if dataframe.duplicated(
        ["project", "run_id", "test_id"]
    ).any():
        raise ValueError("Duplicate executions")

    if not dataframe["failed"].isin([0, 1]).all():
        raise ValueError("Invalid failure labels")

    if not np.isfinite(
        dataframe["duration_seconds"]
    ).all():
        raise ValueError("Invalid durations")

    if dataframe["duration_seconds"].lt(0).any():
        raise ValueError("Negative durations")

    if (
        dataframe["completion_timestamp"]
        < dataframe["timestamp"]
    ).any():
        raise ValueError("Invalid completion timestamp")

    return dataframe


def load_repository(project):
    path = DATA_DIR / f"{project}_normalized.csv"

    dataframe = pd.read_csv(
        path,
        usecols=REQUIRED_COLUMNS,
    )

    dataframe = validate_data(dataframe)

    if not dataframe["project"].eq(project).all():
        raise ValueError("Unexpected project identity")

    return dataframe


def create_build_table(dataframe):
    dataframe = validate_data(dataframe)

    builds = (
        dataframe.groupby(
            ["project", "build_key"],
            as_index=False,
        )
        .agg(
            timestamp=("timestamp", "min"),
            completion_timestamp=(
                "completion_timestamp",
                "max",
            ),
            executions=("failed", "size"),
            failures=("failed", "sum"),
            runs=("run_id", "nunique"),
        )
    )

    return builds.sort_values(
        ["timestamp", "project", "build_key"]
    ).reset_index(drop=True)


def calculate_boundaries(builds):
    total = len(builds)

    if total < 10:
        raise ValueError("Insufficient builds")

    validation_index = (
        total * TRAIN_PERCENT // 100
    )

    test_index = (
        total
        * (TRAIN_PERCENT + VALIDATION_PERCENT)
        // 100
    )

    ordered = builds.sort_values(
        ["timestamp", "project", "build_key"]
    ).reset_index(drop=True)

    validation_start = ordered.iloc[
        validation_index
    ]["timestamp"]

    test_start = ordered.iloc[
        test_index
    ]["timestamp"]

    if validation_start >= test_start:
        raise ValueError(
            "Invalid chronological boundaries"
        )

    return validation_start, test_start


def assign_partitions(
    builds,
    validation_start,
    test_start,
):
    builds = builds.copy()

    builds["partition"] = np.select(
        [
            builds["timestamp"] < validation_start,
            builds["timestamp"] < test_start,
        ],
        [
            "train",
            "validation",
        ],
        default="test",
    )

    builds["original_partition"] = (
        builds["partition"]
    )

    train_overlap = (
        builds["partition"].eq("train")
        & builds["completion_timestamp"].ge(
            validation_start
        )
    )

    validation_overlap = (
        builds["partition"].eq("validation")
        & builds["completion_timestamp"].ge(
            test_start
        )
    )

    builds.loc[
        train_overlap | validation_overlap,
        "partition",
    ] = "embargoed"

    return builds


def verify_partitions(builds):
    partitions = {
        name: builds[
            builds["partition"] == name
        ]
        for name in PARTITIONS
    }

    for name, subset in partitions.items():
        if subset.empty:
            raise ValueError(
                f"Empty partition: {name}"
            )

    train = partitions["train"]

    validation = partitions["validation"]

    test = partitions["test"]

    if (
        train["completion_timestamp"].max()
        >= validation["timestamp"].min()
    ):
        raise ValueError(
            "Global train-validation overlap"
        )

    if (
        validation["completion_timestamp"].max()
        >= test["timestamp"].min()
    ):
        raise ValueError(
            "Global validation-test overlap"
        )

    if builds.duplicated(
        ["project", "build_key"]
    ).any():
        raise ValueError(
            "Duplicate build assignments"
        )

    return True


def summarize_partitions(builds):
    records = []

    for project in sorted(
        builds["project"].unique()
    ):
        for partition in PARTITIONS:
            subset = builds[
                builds["project"].eq(project)
                & builds["partition"].eq(partition)
            ]

            rows = int(
                subset["executions"].sum()
            )

            failures = int(
                subset["failures"].sum()
            )

            records.append({
                "project": project,
                "partition": partition,
                "rows": rows,
                "failures": failures,
                "passes": rows - failures,
                "failure_rate": (
                    failures / rows
                    if rows
                    else 0.0
                ),
                "builds": len(subset),
                "runs": int(
                    subset["runs"].sum()
                ),
                "earliest_start": (
                    str(subset["timestamp"].min())
                    if rows
                    else None
                ),
                "latest_start": (
                    str(subset["timestamp"].max())
                    if rows
                    else None
                ),
                "latest_completion": (
                    str(
                        subset[
                            "completion_timestamp"
                        ].max()
                    )
                    if rows
                    else None
                ),
            })

    return pd.DataFrame(records)


def export_repository(project, builds):
    project_builds = builds[
        builds["project"] == project
    ]

    assignments = dict(zip(
        project_builds["build_key"],
        project_builds["partition"],
    ))

    directory = OUTPUT_DIR / project

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = DATA_DIR / f"{project}_normalized.csv"

    temporary_paths = {
        partition: directory / f"{partition}.csv.tmp"
        for partition in PARTITIONS
    }

    counts = {
        partition: 0
        for partition in PARTITIONS
    }

    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    for chunk in pd.read_csv(
        source,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        labels = chunk["build_key"].map(
            assignments
        )

        if labels.isna().any():
            raise ValueError(
                "Unassigned build"
            )

        for partition in PARTITIONS:
            subset = chunk[
                labels == partition
            ]

            if subset.empty:
                continue

            path = temporary_paths[
                partition
            ]

            subset.to_csv(
                path,
                mode="a",
                header=not path.exists(),
                index=False,
            )

            counts[partition] += len(subset)

    for partition in PARTITIONS:
        expected = int(
            project_builds.loc[
                project_builds["partition"]
                == partition,
                "executions",
            ].sum()
        )

        if counts[partition] != expected:
            raise ValueError(
                "Export row count mismatch"
            )

        if expected == 0:
            raise ValueError(
                f"Empty {project}/{partition}"
            )

    for partition in PARTITIONS:
        temporary_paths[partition].replace(
            directory / f"{partition}.csv"
        )

    return counts


def main():
    started = time.perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("QUANTCI GLOBAL TEMPORAL SPLITTING")
    print("=" * 70)

    build_tables = []

    for project in PROJECTS:
        print(f"\nLoading {project}")

        dataframe = load_repository(
            project
        )

        builds = create_build_table(
            dataframe
        )

        build_tables.append(builds)

        print(
            f"Executions: {len(dataframe):,}"
        )

        print(
            f"Builds: {len(builds):,}"
        )

        del dataframe

    builds = pd.concat(
        build_tables,
        ignore_index=True,
    )

    validation_start, test_start = (
        calculate_boundaries(builds)
    )

    builds = assign_partitions(
        builds,
        validation_start,
        test_start,
    )

    verify_partitions(builds)

    summary = summarize_partitions(
        builds
    )

    if summary["rows"].eq(0).any():
        raise ValueError(
            "Repository missing from a partition"
        )

    if summary["failures"].eq(0).any():
        raise ValueError(
            "Repository partition has no failures"
        )

    if summary["passes"].eq(0).any():
        raise ValueError(
            "Repository partition has no passing executions"
        )

    embargoed = builds[
        builds["partition"] == "embargoed"
    ]

    original_rows = int(
        builds["executions"].sum()
    )

    retained_rows = int(
        summary["rows"].sum()
    )

    excluded_rows = int(
        embargoed["executions"].sum()
    )

    if (
        retained_rows + excluded_rows
        != original_rows
    ):
        raise ValueError(
            "Dataset accounting mismatch"
        )

    print(
        "\nValidation starts:",
        validation_start,
    )

    print(
        "Testing starts:",
        test_start,
    )

    print("\nGlobal Partition Summary\n")

    print(
        summary[
            [
                "project",
                "partition",
                "rows",
                "failures",
                "builds",
            ]
        ].to_string(index=False)
    )

    for project in PROJECTS:
        print(f"\nExporting {project}")

        counts = export_repository(
            project,
            builds,
        )

        print(counts)

    builds.to_csv(
        RESULTS_DIR / "global_build_assignments.csv",
        index=False,
    )

    embargoed.to_csv(
        RESULTS_DIR / "global_embargoed_builds.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS_DIR / "global_split_summary.csv",
        index=False,
    )

    report = {
        "projects": PROJECTS,
        "split_strategy": "global_chronological_build_grouped",
        "validation_start": str(validation_start),
        "test_start": str(test_start),
        "original_builds": len(builds),
        "original_rows": original_rows,
        "original_failures": int(
            builds["failures"].sum()
        ),
        "retained_rows": retained_rows,
        "retained_failures": int(
            summary["failures"].sum()
        ),
        "embargoed_builds": len(embargoed),
        "embargoed_rows": excluded_rows,
        "embargoed_failures": int(
            embargoed["failures"].sum()
        ),
        "train_validation_isolation": True,
        "validation_test_isolation": True,
        "partitions": summary.to_dict(
            orient="records"
        ),
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    (
        RESULTS_DIR / "global_split_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nGlobal Split Report\n")

    print(
        json.dumps(report, indent=4)
    )

    print(
        "\nGLOBAL TEMPORAL SPLITTING COMPLETED"
    )


if __name__ == "__main__":
    main()