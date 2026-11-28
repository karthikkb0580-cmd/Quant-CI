import pandas as pd
import pytest

from src.features.multi_repository_history import (
    TestHistory as ExecutionHistory,
    FEATURE_COLUMNS,
    generate_features,
    validate_data,
)


@pytest.fixture
def timestamp():
    return pd.Timestamp(
        "2023-01-01",
        tz="UTC",
    )


@pytest.fixture
def sample_data(timestamp):
    records = []

    for index in range(5):
        start = timestamp + pd.Timedelta(
            days=index
        )

        records.append({
            "project": "sample",
            "run_id": f"run{index}",
            "build_key": f"build{index}",
            "stage_id": "single",
            "test_id": "TestA",
            "timestamp": start,
            "completion_timestamp": (
                start + pd.Timedelta(hours=1)
            ),
            "failed": int(index % 2 == 0),
            "duration_seconds": float(
                10 + index
            ),
        })

    return pd.DataFrame(records)


def test_feature_count(timestamp):
    history = ExecutionHistory()

    assert len(
        history.features(timestamp)
    ) == 16


def test_new_test(timestamp):
    history = ExecutionHistory()

    features = dict(zip(
        FEATURE_COLUMNS,
        history.features(timestamp),
    ))

    assert features["is_new_test"] == 1

    assert features["historical_executions"] == 0


def test_default_duration(timestamp):
    history = ExecutionHistory()

    assert history.features(timestamp)[-1] == 10.0


def test_history_update(timestamp):
    history = ExecutionHistory()

    history.update(
        1,
        20.0,
        timestamp,
    )

    assert history.executions == 1

    assert history.failures == 1


def test_median_duration(timestamp):
    history = ExecutionHistory()

    for duration in [10.0, 30.0, 20.0]:
        history.update(
            0,
            duration,
            timestamp,
        )

    assert history.median_duration() == 20.0


def test_even_median(timestamp):
    history = ExecutionHistory()

    history.update(0, 10.0, timestamp)

    history.update(0, 30.0, timestamp)

    assert history.median_duration() == 20.0


def test_smoothed_rate(timestamp):
    history = ExecutionHistory()

    features = dict(zip(
        FEATURE_COLUMNS,
        history.features(timestamp),
    ))

    assert features[
        "smoothed_failure_rate"
    ] == pytest.approx(0.01)


def test_consecutive_failures(timestamp):
    history = ExecutionHistory()

    history.update(1, 10.0, timestamp)

    history.update(1, 10.0, timestamp)

    assert history.consecutive_failures == 2

    history.update(0, 10.0, timestamp)

    assert history.consecutive_failures == 0


def test_invalid_outcome(timestamp):
    history = ExecutionHistory()

    with pytest.raises(ValueError):
        history.update(
            2,
            10.0,
            timestamp,
        )


def test_invalid_duration(timestamp):
    history = ExecutionHistory()

    with pytest.raises(ValueError):
        history.update(
            0,
            -10.0,
            timestamp,
        )


def test_future_information(timestamp):
    history = ExecutionHistory()

    history.update(
        1,
        10.0,
        timestamp,
    )

    with pytest.raises(ValueError):
        history.features(timestamp)


def test_walk_forward(sample_data):
    result = generate_features(
        sample_data
    )

    assert result[
        "historical_executions"
    ].tolist() == [0, 1, 2, 3, 4]


def test_current_outcome_not_used(sample_data):
    result = generate_features(
        sample_data
    )

    assert result.iloc[0][
        "historical_failures"
    ] == 0

    assert result.iloc[1][
        "historical_failures"
    ] == 1


def test_overlapping_builds(sample_data):
    sample_data.loc[
        0,
        "completion_timestamp",
    ] += pd.Timedelta(days=2)

    result = generate_features(
        sample_data
    )

    assert result.iloc[1][
        "historical_executions"
    ] == 0


def test_same_build_isolation(sample_data):
    extra = sample_data.iloc[[0]].copy()

    extra["run_id"] = "other_stage"

    extra["stage_id"] = "other_stage"

    dataframe = pd.concat([
        sample_data,
        extra,
    ])

    result = generate_features(
        dataframe
    )

    first_build = result[
        result["build_key"] == "build0"
    ]

    assert first_build[
        "historical_executions"
    ].eq(0).all()


def test_stage_isolation(sample_data):
    sample_data.loc[
        1,
        "stage_id",
    ] = "different_stage"

    result = generate_features(
        sample_data
    )

    assert result.iloc[1][
        "historical_executions"
    ] == 0


def test_missing_columns(sample_data):
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


def test_no_missing_features(sample_data):
    result = generate_features(
        sample_data
    )

    assert not result[
        FEATURE_COLUMNS
    ].isna().any().any()


def test_outcome_mutation(sample_data):
    original = generate_features(
        sample_data
    )

    sample_data.loc[2, "failed"] = (
        1 - sample_data.loc[2, "failed"]
    )

    modified = generate_features(
        sample_data
    )

    pd.testing.assert_series_equal(
        original.loc[2, FEATURE_COLUMNS],
        modified.loc[2, FEATURE_COLUMNS],
    )
