import numpy as np
import pytest

from src.quantum.hybrid_solver import (
    encode_slack,
    repair_state,
)


@pytest.fixture
def report():
    return {
        "retained_test_variables": 2,
        "reduced_qubits": 5,
        "duration_units": [2, 3],
        "slack_weights": [1, 2, 2],
        "budget_units": 5,
    }


def test_zero_slack():
    state = encode_slack(
        0,
        [1, 2, 4],
    )

    assert state.tolist() == [0, 0, 0]


def test_positive_slack():
    weights = np.array(
        [1, 2, 4]
    )

    state = encode_slack(
        6,
        weights,
    )

    assert state @ weights == 6


def test_nonstandard_slack_weights():
    weights = np.array(
        [1, 2, 4, 8, 9]
    )

    for value in range(25):
        state = encode_slack(
            value,
            weights,
        )

        assert state @ weights == value


def test_negative_slack():
    with pytest.raises(ValueError):
        encode_slack(
            -1,
            [1, 2, 4],
        )


def test_unrepresentable_slack():
    with pytest.raises(ValueError):
        encode_slack(
            8,
            [1, 2, 4],
        )


def test_invalid_weights():
    with pytest.raises(ValueError):
        encode_slack(
            2,
            [1, -2, 4],
        )


def test_repair_preserves_selection(report):
    original = np.array(
        [1, 0, 0, 0, 0]
    )

    repaired = repair_state(
        original,
        report,
    )

    assert np.array_equal(
        original[:2],
        repaired[:2],
    )


def test_repair_satisfies_constraint(report):
    repaired = repair_state(
        [1, 0, 0, 0, 0],
        report,
    )

    duration = (
        repaired[:2]
        @ np.array([2, 3])
    )

    slack = (
        repaired[2:]
        @ np.array([1, 2, 2])
    )

    assert duration + slack == 5


def test_repair_empty_selection(report):
    repaired = repair_state(
        [0, 0, 0, 0, 0],
        report,
    )

    assert repaired[:2].sum() == 0

    assert (
        repaired[2:]
        @ np.array([1, 2, 2])
    ) == 5


def test_repair_full_budget(report):
    repaired = repair_state(
        [1, 1, 1, 1, 1],
        report,
    )

    assert repaired.tolist() == [
        1, 1, 0, 0, 0
    ]


def test_reject_over_budget(report):
    report["budget_units"] = 4

    repaired = repair_state(
        [1, 1, 0, 0, 0],
        report,
    )

    assert repaired is None


def test_invalid_binary_state(report):
    with pytest.raises(ValueError):
        repair_state(
            [2, 0, 0, 0, 0],
            report,
        )


def test_invalid_state_size(report):
    with pytest.raises(ValueError):
        repair_state(
            [1, 0],
            report,
        )


def test_already_valid_state(report):
    state = np.array(
        [1, 1, 0, 0, 0]
    )

    repaired = repair_state(
        state,
        report,
    )

    assert np.array_equal(
        repaired,
        state,
    )


def test_repair_idempotence(report):
    first = repair_state(
        [1, 0, 0, 0, 0],
        report,
    )

    second = repair_state(
        first,
        report,
    )

    assert np.array_equal(
        first,
        second,
    )