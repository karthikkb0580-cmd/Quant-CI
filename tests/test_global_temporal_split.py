import pandas as pd
import pytest

from src.data.global_temporal_split import (
    validate_data,
    create_build_table,
    calculate_boundaries,
    assign_partitions,
    verify_partitions,
    summarize_partitions,
)


@pytest.fixture
def sample_data():
    records = []

    start = pd.Timestamp(
        "2023-01-01",
        tz="UTC",
    )

    for index in range(40):
        project = (
            "karaf"
            if index % 2 == 0
            else "hbase"
        )

        timestamp = start + pd.Timedelta(
            days=index
        )

        for stage in range(2):
            for test in range(2):
                records.append({
                    "project": project,
                    "run_id": f"{project}:{index}:{stage}",
                    "build_key": f"{project}:{index}",
                    "test_id": f"Test{test}",
                    "timestamp": timestamp,
                    "completion_timestamp": (
                        timestamp
                        + pd.Timedelta(hours=2)
                    ),
                    "failed": test % 2,
                    "duration_seconds": 10.0,
                })

    return pd.DataFrame(records)


@pytest.fixture
def assigned_builds(sample_data):
    builds = create_build_table(
        sample_data
    )

    validation_start, test_start = (
        calculate_boundaries(builds)
    )

    return assign_partitions(
        builds,
        validation_start,
        test_start,
    )


def test_valid_data(sample_data):
    assert len(
        validate_data(sample_data)
    ) == 160


def test_empty_data(sample_data):
    with pytest.raises(ValueError):
        validate_data(
            sample_data.iloc[:0]
        )


def test_missing_column(sample_data):
    with pytest.raises(ValueError):
        validate_data(
            sample_data.drop(
                columns=["failed"]
            )
        )


def test_duplicate_execution(sample_data):
    dataframe = pd.concat([
        sample_data,
        sample_data.iloc[[0]],
    ])

    with pytest.raises(ValueError):
        validate_data(dataframe)


def test_invalid_label(sample_data):
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

    assert builds["executions"].eq(4).all()

    assert builds["runs"].eq(2).all()


def test_boundaries(sample_data):
    builds = create_build_table(
        sample_data
    )

    validation_start, test_start = (
        calculate_boundaries(builds)
    )

    assert validation_start == (
        builds.iloc[28]["timestamp"]
    )

    assert test_start == (
        builds.iloc[34]["timestamp"]
    )


def test_partition_counts(assigned_builds):
    counts = assigned_builds[
        "partition"
    ].value_counts()

    assert counts["train"] == 28

    assert counts["validation"] == 6

    assert counts["test"] == 6


def test_global_isolation(assigned_builds):
    assert verify_partitions(
        assigned_builds
    )


def test_training_embargo(sample_data):
    builds = create_build_table(
        sample_data
    )

    validation_start, test_start = (
        calculate_boundaries(builds)
    )

    builds.loc[
        27,
        "completion_timestamp",
    ] = validation_start

    result = assign_partitions(
        builds,
        validation_start,
        test_start,
    )

    assert result.loc[
        27,
        "partition",
    ] == "embargoed"

    assert verify_partitions(result)


def test_validation_embargo(sample_data):
    builds = create_build_table(
        sample_data
    )

    validation_start, test_start = (
        calculate_boundaries(builds)
    )

    builds.loc[
        33,
        "completion_timestamp",
    ] = test_start

    result = assign_partitions(
        builds,
        validation_start,
        test_start,
    )

    assert result.loc[
        33,
        "partition",
    ] == "embargoed"

    assert verify_partitions(result)


def test_partition_accounting(assigned_builds):
    summary = summarize_partitions(
        assigned_builds
    )

    assert summary["rows"].sum() == 160

    assert summary["failures"].sum() == 80


def test_repository_coverage(assigned_builds):
    summary = summarize_partitions(
        assigned_builds
    )

    assert len(summary) == 6

    assert summary["rows"].gt(0).all()


def test_insufficient_builds(sample_data):
    builds = create_build_table(
        sample_data
    )

    with pytest.raises(ValueError):
        calculate_boundaries(
            builds.iloc[:5]
        )


def test_duplicate_build_assignment(
    assigned_builds,
):
    builds = pd.concat([
        assigned_builds,
        assigned_builds.iloc[[0]],
    ])

    with pytest.raises(ValueError):
        verify_partitions(builds)


def test_temporal_overlap(assigned_builds):
    validation_start = assigned_builds.loc[
        assigned_builds["partition"]
        == "validation",
        "timestamp",
    ].min()

    assigned_builds.loc[
        0,
        "completion_timestamp",
    ] = validation_start

    with pytest.raises(ValueError):
        verify_partitions(assigned_builds)