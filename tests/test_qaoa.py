import numpy as np
import pandas as pd
import pytest

from qiskit.quantum_info import SparsePauliOp

from src.quantum.qaoa_solver import (
    normalize_hamiltonian,
    calculate_qubo_energy,
    decode_measurement,
    approximation_ratio,
    find_best_feasible_measurement,
)


@pytest.fixture
def report():
    return {
        "test_variables": 2,
        "total_binary_variables": 4,
        "duration_units": [2, 3],
        "slack_weights": [1, 2],
        "budget_units": 5,
    }


def test_hamiltonian_normalization():
    operator = SparsePauliOp.from_list(
        [
            ("ZI", 10.0),
            ("IZ", -5.0),
        ]
    )

    normalized, scale = (
        normalize_hamiltonian(
            operator
        )
    )

    assert scale == 10.0

    assert np.max(
        np.abs(normalized.coeffs)
    ) == pytest.approx(1.0)


def test_zero_hamiltonian():
    operator = SparsePauliOp.from_list(
        [("II", 0.0)]
    )

    normalized, scale = (
        normalize_hamiltonian(
            operator
        )
    )

    assert scale == 1.0

    assert normalized.coeffs[0] == 0


def test_qubo_energy():
    matrix = np.array(
        [
            [-2.0, 3.0],
            [0.0, -1.0],
        ]
    )

    energy = calculate_qubo_energy(
        matrix,
        4.0,
        [1, 1],
    )

    assert energy == pytest.approx(
        4.0
    )


def test_invalid_binary_state():
    with pytest.raises(ValueError):
        calculate_qubo_energy(
            np.eye(2),
            0,
            [1, 2],
        )


def test_invalid_state_size():
    with pytest.raises(ValueError):
        calculate_qubo_energy(
            np.eye(2),
            0,
            [1, 0, 1],
        )


def test_measurement_decoding(report):
    decoded = decode_measurement(
        "0011",
        report,
    )

    assert decoded[
        "selection"
    ].tolist() == [1, 1]

    assert decoded[
        "used_duration_units"
    ] == 5

    assert decoded[
        "constraint_satisfied"
    ]


def test_slack_decoding(report):
    decoded = decode_measurement(
        "1101",
        report,
    )

    assert decoded[
        "used_duration_units"
    ] == 2

    assert decoded[
        "slack_units"
    ] == 3

    assert decoded[
        "constraint_satisfied"
    ]


def test_invalid_measurement(report):
    with pytest.raises(ValueError):
        decode_measurement(
            "101",
            report,
        )


def test_unsatisfied_constraint(report):
    decoded = decode_measurement(
        "0001",
        report,
    )

    assert decoded[
        "budget_feasible"
    ]

    assert not decoded[
        "constraint_satisfied"
    ]


def test_approximation_ratio():
    assert approximation_ratio(
        8.0,
        10.0,
    ) == pytest.approx(0.8)


def test_zero_optimal_objective():
    assert approximation_ratio(
        0.0,
        0.0,
    ) is None


def test_best_feasible_measurement():
    measurements = pd.DataFrame(
        {
            "bitstring": [
                "00",
                "01",
                "10",
            ],
            "objective_value": [
                0.0,
                5.0,
                3.0,
            ],
            "count": [
                10,
                20,
                30,
            ],
            "budget_feasible": [
                True,
                False,
                True,
            ],
            "constraint_satisfied": [
                True,
                False,
                True,
            ],
            "original_budget_feasible": [
                True,
                False,
                True,
            ],
        }
    )

    best = find_best_feasible_measurement(
        measurements
    )

    assert best[
        "bitstring"
    ] == "10"


def test_no_feasible_measurement():
    measurements = pd.DataFrame(
        {
            "bitstring": ["11"],
            "objective_value": [5.0],
            "count": [10],
            "budget_feasible": [False],
            "constraint_satisfied": [False],
            "original_budget_feasible": [False],
        }
    )

    assert find_best_feasible_measurement(
        measurements
    ) is None