import numpy as np
import pandas as pd
import pytest

from src.quantum.qaoa_experiments import (
    create_initial_point,
    decode_state,
    summarize_measurements,
    aggregate_results,
)


@pytest.fixture
def pruning_report():
    return {
        "retained_test_variables": 2,
        "reduced_qubits": 4,
        "duration_units": [2, 3],
        "slack_weights": [1, 2],
        "budget_units": 5,
    }


@pytest.fixture
def measurements():
    return pd.DataFrame(
        {
            "bitstring": [
                "0001",
                "0011",
                "1111",
            ],
            "count": [
                20,
                30,
                50,
            ],
            "feasible": [
                True,
                True,
                False,
            ],
            "optimal": [
                False,
                True,
                False,
            ],
            "objective": [
                1.0,
                2.0,
                3.0,
            ],
            "energy": [
                -1.0,
                -2.0,
                10.0,
            ],
            "selected_count": [
                1,
                2,
                2,
            ],
            "estimated_duration_seconds": [
                10.0,
                20.0,
                20.0,
            ],
        }
    )


def test_initial_point_size():
    point = create_initial_point(
        2,
        42,
    )

    assert len(point) == 4


def test_initial_point_reproducibility():
    first = create_initial_point(
        2,
        123,
    )

    second = create_initial_point(
        2,
        123,
    )

    assert np.array_equal(
        first,
        second,
    )


def test_different_initializations():
    first = create_initial_point(
        2,
        123,
    )

    second = create_initial_point(
        2,
        2026,
    )

    assert not np.array_equal(
        first,
        second,
    )


def test_invalid_depth():
    with pytest.raises(ValueError):
        create_initial_point(
            0,
            42,
        )


def test_decode_state(pruning_report):
    decoded = decode_state(
        "0011",
        pruning_report,
    )

    assert decoded[
        "selection"
    ].tolist() == [1, 1]

    assert decoded[
        "constraint_satisfied"
    ]


def test_invalid_state(pruning_report):
    with pytest.raises(ValueError):
        decode_state(
            "101",
            pruning_report,
        )


def test_feasible_rate(measurements):
    result = summarize_measurements(
        measurements,
        2.0,
    )

    assert result[
        "feasible_measurement_rate"
    ] == pytest.approx(0.5)


def test_optimal_rate(measurements):
    result = summarize_measurements(
        measurements,
        2.0,
    )

    assert result[
        "optimal_measurement_rate"
    ] == pytest.approx(0.3)


def test_best_objective(measurements):
    result = summarize_measurements(
        measurements,
        2.0,
    )

    assert result[
        "best_objective"
    ] == 2.0


def test_approximation_ratio(measurements):
    result = summarize_measurements(
        measurements,
        2.0,
    )

    assert result[
        "approximation_ratio"
    ] == 1.0


def test_no_feasible_solution(measurements):
    measurements["feasible"] = False
    measurements["optimal"] = False

    result = summarize_measurements(
        measurements,
        2.0,
    )

    assert not result[
        "feasible_solution_found"
    ]

    assert result[
        "approximation_ratio"
    ] is None


def test_aggregation():
    dataframe = pd.DataFrame(
        {
            "experiment": ["E1", "E1"],
            "approximation_ratio": [
                0.5,
                1.0,
            ],
            "feasible_solution_found": [
                True,
                True,
            ],
            "optimal_shots": [0, 1],
            "feasible_measurement_rate": [
                0.1,
                0.2,
            ],
            "optimal_measurement_rate": [
                0.0,
                0.01,
            ],
            "total_seconds": [
                2.0,
                4.0,
            ],
            "circuit_depth": [
                20,
                20,
            ],
        }
    )

    result = aggregate_results(
        dataframe
    )

    assert result.iloc[0][
        "mean_approximation_ratio"
    ] == pytest.approx(0.75)

    assert result.iloc[0][
        "mean_runtime_seconds"
    ] == pytest.approx(3.0)