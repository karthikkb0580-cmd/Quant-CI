import json

import pandas as pd

from src.config import (
    NORMALIZED_DATA_FILE,
    SPLITS_DIR,
    TRAIN_DATA_FILE,
    VALIDATION_DATA_FILE,
    TEST_DATA_FILE,
    SPLIT_REPORT_FILE,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
)


def load_dataset():
    if not NORMALIZED_DATA_FILE.exists():
        raise FileNotFoundError(NORMALIZED_DATA_FILE)

    dataframe = pd.read_csv(
        NORMALIZED_DATA_FILE,
        low_memory=False,
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
        errors="raise",
    )

    dataframe["build_key"] = (
        dataframe["project"].astype(str)
        + ":"
        + dataframe["pr_name"].astype(str)
        + ":"
        + dataframe["build_id"].astype(str)
    )

    if dataframe["timestamp"].isna().any():
        raise ValueError("Missing timestamps found.")

    if not dataframe["failed"].isin([0, 1]).all():
        raise ValueError("Invalid failure labels.")

    if dataframe.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError("Duplicate test executions.")

    return dataframe.sort_values(
        ["timestamp", "build_key", "run_id", "test_id"]
    ).reset_index(drop=True)


def get_builds(dataframe):
    timestamp_counts = (
        dataframe.groupby("build_key")["timestamp"]
        .nunique()
    )

    if timestamp_counts.gt(1).any():
        raise ValueError(
            "Inconsistent timestamps within builds."
        )

    builds = (
        dataframe.groupby("build_key", as_index=False)
        .agg(
            timestamp=("timestamp", "first"),
            commit_sha=("commit_sha", "first"),
        )
        .sort_values(["timestamp", "build_key"])
        .reset_index(drop=True)
    )

    return builds


def split_builds(builds):
    if abs(
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
        - 1.0
    ) > 1e-9:
        raise ValueError("Invalid split ratios.")

    total_builds = len(builds)

    train_end = int(
        total_builds * TRAIN_RATIO
    )

    validation_end = int(
        total_builds
        * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    if not 0 < train_end < validation_end < total_builds:
        raise ValueError("Insufficient builds for splitting.")

    validation_start = builds.iloc[
        train_end
    ]["timestamp"]

    test_start = builds.iloc[
        validation_end
    ]["timestamp"]

    train_builds = builds[
        builds["timestamp"] < validation_start
    ]

    validation_builds = builds[
        (builds["timestamp"] >= validation_start)
        & (builds["timestamp"] < test_start)
    ]

    test_builds = builds[
        builds["timestamp"] >= test_start
    ]

    return (
        train_builds,
        validation_builds,
        test_builds,
    )


def create_partitions(dataframe, build_partitions):
    train_builds, validation_builds, test_builds = (
        build_partitions
    )

    train = dataframe[
        dataframe["build_key"].isin(
            train_builds["build_key"]
        )
    ].copy()

    validation = dataframe[
        dataframe["build_key"].isin(
            validation_builds["build_key"]
        )
    ].copy()

    test = dataframe[
        dataframe["build_key"].isin(
            test_builds["build_key"]
        )
    ].copy()

    return train, validation, test


def validate_partitions(
    original,
    train,
    validation,
    test,
):
    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, dataframe in partitions.items():
        if dataframe.empty:
            raise ValueError(
                f"{name} partition is empty."
            )

        if dataframe["failed"].nunique() != 2:
            raise ValueError(
                f"{name} does not contain both classes."
            )

    train_keys = set(train["build_key"])
    validation_keys = set(validation["build_key"])
    test_keys = set(test["build_key"])

    if train_keys & validation_keys:
        raise ValueError(
            "Train-validation build overlap."
        )

    if train_keys & test_keys:
        raise ValueError(
            "Train-test build overlap."
        )

    if validation_keys & test_keys:
        raise ValueError(
            "Validation-test build overlap."
        )

    if not (
        train["timestamp"].max()
        < validation["timestamp"].min()
    ):
        raise ValueError(
            "Train-validation chronological overlap."
        )

    if not (
        validation["timestamp"].max()
        < test["timestamp"].min()
    ):
        raise ValueError(
            "Validation-test chronological overlap."
        )

    total_rows = (
        len(train)
        + len(validation)
        + len(test)
    )

    if total_rows != len(original):
        raise ValueError(
            "Dataset row count mismatch."
        )

    total_failures = (
        train["failed"].sum()
        + validation["failed"].sum()
        + test["failed"].sum()
    )

    if total_failures != original["failed"].sum():
        raise ValueError(
            "Failure count mismatch."
        )

    print("Partition validation passed.")
    print("Build overlap: None")
    print("Chronological overlap: None")
    print("Missing records: None")


def partition_statistics(dataframe):
    failed = int(dataframe["failed"].sum())

    passed = len(dataframe) - failed

    return {
        "rows": len(dataframe),
        "builds": int(
            dataframe["build_key"].nunique()
        ),
        "suite_runs": int(
            dataframe["run_id"].nunique()
        ),
        "unique_tests": int(
            dataframe["test_id"].nunique()
        ),
        "unique_commits": int(
            dataframe["commit_sha"].nunique()
        ),
        "passed": passed,
        "failed": failed,
        "failure_rate_percent": round(
            failed / len(dataframe) * 100,
            6,
        ),
        "start_date": str(
            dataframe["timestamp"].min()
        ),
        "end_date": str(
            dataframe["timestamp"].max()
        ),
    }


def save_partitions(
    original,
    train,
    validation,
    test,
):
    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SPLIT_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        TRAIN_DATA_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_DATA_FILE,
        index=False,
    )

    test.to_csv(
        TEST_DATA_FILE,
        index=False,
    )

    train_commits = set(train["commit_sha"])
    validation_commits = set(validation["commit_sha"])
    test_commits = set(test["commit_sha"])

    report = {
        "strategy": "chronological_build_grouped",
        "total_rows": len(original),
        "total_builds": int(
            original["build_key"].nunique()
        ),
        "requested_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "train": partition_statistics(train),
        "validation": partition_statistics(validation),
        "test": partition_statistics(test),
        "commit_overlap": {
            "train_validation": len(
                train_commits & validation_commits
            ),
            "train_test": len(
                train_commits & test_commits
            ),
            "validation_test": len(
                validation_commits & test_commits
            ),
        },
        "build_overlap": 0,
        "chronological_overlap": False,
    }

    SPLIT_REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nSplit Summary:")
    print(json.dumps(report, indent=4))

    print(f"\nTraining data: {TRAIN_DATA_FILE}")
    print(f"Validation data: {VALIDATION_DATA_FILE}")
    print(f"Testing data: {TEST_DATA_FILE}")
    print(f"Split report: {SPLIT_REPORT_FILE}")


def main():
    print("=" * 60)
    print("QuantCI - Chronological Dataset Splitting")
    print("=" * 60)

    dataframe = load_dataset()

    builds = get_builds(dataframe)

    print(f"\nTotal records: {len(dataframe):,}")
    print(f"Total builds: {len(builds)}")

    build_partitions = split_builds(builds)

    train, validation, test = create_partitions(
        dataframe,
        build_partitions,
    )

    validate_partitions(
        dataframe,
        train,
        validation,
        test,
    )

    save_partitions(
        dataframe,
        train,
        validation,
        test,
    )

    print("\n" + "=" * 60)
    print("DATASET SPLITTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()