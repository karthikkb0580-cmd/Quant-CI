import numpy as np
import pandas as pd
import pytest

from scipy import sparse

from src.features.global_feature_generation import (
    validate_assignments,
    assign_history_partitions,
    verify_expected_counts,
    transform_in_batches,
    verify_global_contexts,
    EXPECTED_COUNTS,
)


class DummyPreprocessor:
    def transform(self, dataframe):
        return sparse.csr_matrix(
            dataframe[["a", "b"]]
            .to_numpy(dtype=np.float32)
        )


@pytest.fixture
def assignments():
    return pd.DataFrame({
        "project": [
            "karaf",
            "karaf",
            "hbase",
            "hbase",
        ],
        "build_key": [
            "k1",
            "k2",
            "h1",
            "h2",
        ],
        "partition": [
            "train",
            "validation",
            "test",
            "embargoed",
        ],
    })


@pytest.fixture
def history():
    return pd.DataFrame({
        "project": [
            "karaf",
            "karaf",
            "hbase",
            "hbase",
        ],
        "build_key": [
            "k1",
            "k2",
            "h1",
            "h2",
        ],
        "test_id": [
            "A",
            "B",
            "C",
            "D",
        ],
    })


@pytest.fixture
def contexts():
    result = {}

    for index, partition in enumerate(
        ["train", "validation", "test"]
    ):
        timestamp = (
            pd.Timestamp(
                "2023-01-01",
                tz="UTC",
            )
            + pd.Timedelta(days=index)
        )

        result[partition] = pd.DataFrame({
            "project": ["karaf"],
            "build_key": [f"b{index}"],
            "timestamp": [timestamp],
            "completion_timestamp": [
                timestamp
                + pd.Timedelta(hours=1)
            ],
        })

    return result


def test_valid_assignments(assignments):
    assert len(
        validate_assignments(assignments)
    ) == 4


def test_empty_assignments(assignments):
    with pytest.raises(ValueError):
        validate_assignments(
            assignments.iloc[:0]
        )


def test_missing_columns(assignments):
    with pytest.raises(ValueError):
        validate_assignments(
            assignments.drop(
                columns=["partition"]
            )
        )


def test_duplicate_assignments(assignments):
    dataframe = pd.concat([
        assignments,
        assignments.iloc[[0]],
    ])

    with pytest.raises(ValueError):
        validate_assignments(dataframe)


def test_invalid_partition(assignments):
    assignments.loc[
        0,
        "partition",
    ] = "unknown"

    with pytest.raises(ValueError):
        validate_assignments(assignments)


def test_missing_assignment_value(assignments):
    assignments.loc[
        0,
        "partition",
    ] = None

    with pytest.raises(ValueError):
        validate_assignments(assignments)


def test_history_assignment(
    history,
    assignments,
):
    result = assign_history_partitions(
        history,
        assignments,
    )

    assert len(result) == 4

    assert result["partition"].tolist() == [
        "train",
        "validation",
        "test",
        "embargoed",
    ]


def test_unassigned_history(
    history,
    assignments,
):
    history.loc[
        0,
        "build_key",
    ] = "unknown"

    with pytest.raises(ValueError):
        assign_history_partitions(
            history,
            assignments,
        )


def test_expected_counts():
    rows, failures = EXPECTED_COUNTS[
        "train"
    ]

    labels = np.zeros(
        rows,
        dtype=np.int8,
    )

    labels[:failures] = 1

    dataframe = pd.DataFrame({
        "failed": labels
    })

    assert verify_expected_counts(
        dataframe,
        "train",
    )


def test_invalid_row_count():
    dataframe = pd.DataFrame({
        "failed": [0, 1]
    })

    with pytest.raises(ValueError):
        verify_expected_counts(
            dataframe,
            "train",
        )


def test_invalid_failure_count():
    rows, _ = EXPECTED_COUNTS[
        "train"
    ]

    dataframe = pd.DataFrame({
        "failed": np.zeros(
            rows,
            dtype=np.int8,
        )
    })

    with pytest.raises(ValueError):
        verify_expected_counts(
            dataframe,
            "train",
        )


def test_batch_transformation():
    dataframe = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [5, 4, 3, 2, 1],
    })

    result = transform_in_batches(
        DummyPreprocessor(),
        dataframe,
        batch_size=2,
    )

    assert result.shape == (5, 2)

    assert sparse.isspmatrix_csr(
        result
    )


def test_batch_order():
    dataframe = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })

    result = transform_in_batches(
        DummyPreprocessor(),
        dataframe,
        batch_size=1,
    )

    assert np.array_equal(
        result.toarray(),
        dataframe.to_numpy(),
    )


def test_invalid_batch_size():
    dataframe = pd.DataFrame({
        "a": [1],
        "b": [2],
    })

    with pytest.raises(ValueError):
        transform_in_batches(
            DummyPreprocessor(),
            dataframe,
            batch_size=0,
        )


def test_empty_transformation():
    dataframe = pd.DataFrame({
        "a": [],
        "b": [],
    })

    with pytest.raises(ValueError):
        transform_in_batches(
            DummyPreprocessor(),
            dataframe,
        )


def test_global_isolation(contexts):
    assert verify_global_contexts(
        contexts
    )


def test_build_overlap(contexts):
    contexts["test"]["build_key"] = (
        contexts["train"]["build_key"]
    )

    with pytest.raises(ValueError):
        verify_global_contexts(
            contexts
        )


def test_temporal_overlap(contexts):
    contexts["train"][
        "completion_timestamp"
    ] = contexts["validation"][
        "timestamp"
    ]

    with pytest.raises(ValueError):
        verify_global_contexts(
            contexts
        )


def test_missing_partition(contexts):
    del contexts["validation"]

    with pytest.raises(ValueError):
        verify_global_contexts(
            contexts
        )


def test_empty_partition(contexts):
    contexts["test"] = (
        contexts["test"].iloc[:0]
    )

    with pytest.raises(ValueError):
        verify_global_contexts(
            contexts
        )