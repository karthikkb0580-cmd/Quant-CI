import pandas as pd
import pytest

from src.data.multi_repository_split import (
    validate_data,
    create_build_table,
    create_partitions,
    verify_partitions,
)


@pytest.fixture
def sample_data():
    records = []

    timestamps = pd.date_range(
        "2023-01-01",
        periods=40,
        freq="D",
        tz="UTC",
    )

    for index, timestamp in enumerate(
        timestamps
    ):
        for stage in ["stage1", "stage2"]:
            for test in ["A", "B"]:
                records.append({
                    "project": "sample",
                    "run_id": f"build{index}:{stage}",
                    "build_key": f"build{index}",
                    "test_id": test,
                    "timestamp": timestamp,
                    "completion_timestamp": (
                        timestamp
                        + pd.Timedelta(hours=1)
                    ),
                    "commit_sha": f"commit{index}",
                    "failed": int(
                        test == "A"
                        and index % 5 == 0
                    ),
                    "duration_seconds": 10.0,
                })

    return pd.DataFrame(records)


def test_valid_data(sample_data):
    assert len(
        validate_data(sample_data)
    ) == 160


def test_empty_dataset(sample_data):
    with pytest.raises(ValueError):
        validate_data(
            sample_data.iloc[:0]
        )


def test_missing_columns(sample_data):
    with pytest.raises(ValueError):
        validate_data(
            sample_data.drop(
                columns=["failed"]
            )
        )


def test_duplicate_executions(sample_data):
    dataframe = pd.concat([
        sample_data,
        sample_data.iloc[[0]],
    ])

    with pytest.raises(ValueError):
        validate_data(dataframe)


def test_invalid_labels(sample_data):
    sample_data.loc[0, "failed"] = 2

    with pytest.raises(ValueError):
        validate_data(sample_data)


def test_negative_duration(sample_data):
    sample_data.loc[
        0,
        "duration_seconds",
    ] = -1

    with pytest.raises(ValueError):
        validate_data(sample_data)


def test_invalid_completion(sample_data):
    sample_data.loc[
        0,
        "completion_timestamp",
    ] = (
        sample_data.loc[0, "timestamp"]
        - pd.Timedelta(hours=1)
    )

    with pytest.raises(ValueError):
        validate_data(sample_data)


def test_build_grouping(sample_data):
    builds = create_build_table(
        sample_data
    )

    assert len(builds) == 40

    assert builds["stages"].eq(2).all()

    assert builds["executions"].eq(4).all()


def test_partition_counts(sample_data):
    builds = create_build_table(
        sample_data
    )

    partitions, excluded = create_partitions(
        builds
    )

    assert len(partitions["train"]) == 28

    assert len(partitions["validation"]) == 6

    assert len(partitions["test"]) == 6

    assert excluded.empty


def test_partition_verification(sample_data):
    builds = create_build_table(
        sample_data
    )

    partitions, _ = create_partitions(
        builds
    )

    assert verify_partitions(partitions)


def test_temporal_embargo(sample_data):
    mask = sample_data[
        "build_key"
    ].eq("build27")

    sample_data.loc[
        mask,
        "completion_timestamp",
    ] += pd.Timedelta(days=2)

    builds = create_build_table(
        sample_data
    )

    partitions, excluded = create_partitions(
        builds
    )

    assert "build27" in set(
        excluded["build_key"]
    )

    assert "build27" not in set(
        partitions["train"]["build_key"]
    )

    assert verify_partitions(partitions)


def test_insufficient_builds(sample_data):
    builds = create_build_table(
        sample_data
    )

    with pytest.raises(ValueError):
        create_partitions(
            builds.head(5)
        )


def test_detect_build_overlap(sample_data):
    builds = create_build_table(
        sample_data
    )

    partitions, _ = create_partitions(
        builds
    )

    partitions["test"] = pd.concat([
        partitions["test"],
        partitions["train"].iloc[[0]],
    ])

    with pytest.raises(ValueError):
        verify_partitions(partitions)


def test_partition_accounting(sample_data):
    builds = create_build_table(
        sample_data
    )

    partitions, excluded = create_partitions(
        builds
    )

    total = sum(
        int(partition["executions"].sum())
        for partition in partitions.values()
    )

    total += int(
        excluded["executions"].sum()
    )

    assert total == len(sample_data)