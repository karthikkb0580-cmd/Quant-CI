import numpy as np
import pandas as pd
import pytest

from scipy.sparse import csr_matrix

from src.data.temporal_embargo import (
    find_excluded_runs,
    create_eligibility_mask,
    validate_alignment,
)


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "run_id": ["A", "A", "B", "B", "C"],
            "test_id": ["T1", "T2", "T1", "T2", "T1"],
            "completion_timestamp": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-03",
                    "2023-01-03",
                    "2023-01-02",
                ],
                utc=True,
            ),
            "failed": [0, 1, 0, 0, 1],
        }
    )


def test_excluded_runs(dataframe):
    cutoff = pd.Timestamp(
        "2023-01-02",
        tz="UTC",
    )

    excluded = find_excluded_runs(
        dataframe,
        cutoff,
    )

    assert set(excluded) == {"B", "C"}


def test_eligibility_mask(dataframe):
    cutoff = pd.Timestamp(
        "2023-01-02",
        tz="UTC",
    )

    mask = create_eligibility_mask(
        dataframe,
        cutoff,
    )

    assert mask.tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]


def test_strict_cutoff(dataframe):
    cutoff = pd.Timestamp(
        "2023-01-01",
        tz="UTC",
    )

    mask = create_eligibility_mask(
        dataframe,
        cutoff,
    )

    assert mask.sum() == 0


def test_complete_run_exclusion(dataframe):
    dataframe.loc[
        1,
        "completion_timestamp"
    ] = pd.Timestamp(
        "2023-01-04",
        tz="UTC",
    )

    cutoff = pd.Timestamp(
        "2023-01-02",
        tz="UTC",
    )

    mask = create_eligibility_mask(
        dataframe,
        cutoff,
    )

    assert not mask[0]
    assert not mask[1]


def test_no_exclusions(dataframe):
    cutoff = pd.Timestamp(
        "2023-02-01",
        tz="UTC",
    )

    mask = create_eligibility_mask(
        dataframe,
        cutoff,
    )

    assert mask.all()


def test_failure_preservation(dataframe):
    cutoff = pd.Timestamp(
        "2023-01-03",
        tz="UTC",
    )

    mask = create_eligibility_mask(
        dataframe,
        cutoff,
    )

    assert dataframe.loc[
        mask,
        "failed"
    ].sum() == 2


def test_valid_alignment(dataframe):
    matrix = csr_matrix(
        np.ones((5, 3))
    )

    labels = dataframe[
        "failed"
    ].to_numpy()

    validate_alignment(
        dataframe,
        dataframe.copy(),
        matrix,
        labels,
    )


def test_invalid_row_count(dataframe):
    matrix = csr_matrix(
        np.ones((4, 3))
    )

    labels = dataframe[
        "failed"
    ].to_numpy()

    with pytest.raises(ValueError):
        validate_alignment(
            dataframe,
            dataframe.copy(),
            matrix,
            labels,
        )


def test_invalid_row_order(dataframe):
    matrix = csr_matrix(
        np.ones((5, 3))
    )

    labels = dataframe[
        "failed"
    ].to_numpy()

    context = dataframe.iloc[
        ::-1
    ].reset_index(drop=True)

    with pytest.raises(ValueError):
        validate_alignment(
            dataframe,
            context,
            matrix,
            labels,
        )


def test_invalid_labels(dataframe):
    matrix = csr_matrix(
        np.ones((5, 3))
    )

    labels = np.zeros(5)

    with pytest.raises(ValueError):
        validate_alignment(
            dataframe,
            dataframe.copy(),
            matrix,
            labels,
        )