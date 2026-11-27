import itertools

import numpy as np
import pytest

from qiskit.quantum_info import Statevector

from src.quantum.ising import (
    qubo_to_ising,
    binary_to_spin,
    spin_to_binary,
    ising_energy,
    operator_energy,
    create_pauli_label,
    state_to_bitstring,
    bitstring_to_state,
    verify_all_states,
)


@pytest.fixture
def matrix():
    return np.array(
        [
            [-2.0, 3.0, 1.0],
            [0.0, -1.0, 2.0],
            [0.0, 0.0, -4.0],
        ]
    )


def test_binary_to_spin():
    result = binary_to_spin(
        [0, 1, 0, 1]
    )

    assert result.tolist() == [
        1, -1, 1, -1
    ]


def test_spin_to_binary():
    result = spin_to_binary(
        [1, -1, 1, -1]
    )

    assert result.tolist() == [
        0, 1, 0, 1
    ]


def test_invalid_binary():
    with pytest.raises(ValueError):
        binary_to_spin(
            [0, 2, 1]
        )


def test_invalid_spin():
    with pytest.raises(ValueError):
        spin_to_binary(
            [1, 0, -1]
        )


def test_pauli_label():
    assert create_pauli_label(
        3,
        [0],
    ) == "IIZ"


def test_quadratic_pauli_label():
    assert create_pauli_label(
        3,
        [0, 2],
    ) == "ZIZ"


def test_single_variable_conversion():
    ising = qubo_to_ising(
        np.array([[-2.0]]),
        3.0,
    )

    assert ising.linear[0] == pytest.approx(
        1.0
    )

    assert ising.offset == pytest.approx(
        2.0
    )


def test_quadratic_conversion():
    matrix = np.array(
        [
            [0.0, 4.0],
            [0.0, 0.0],
        ]
    )

    ising = qubo_to_ising(
        matrix
    )

    assert ising.linear.tolist() == [
        -1.0,
        -1.0,
    ]

    assert ising.quadratic[0, 1] == (
        pytest.approx(1.0)
    )

    assert ising.offset == pytest.approx(
        1.0
    )


def test_energy_equivalence(matrix):
    offset = 5.0

    ising = qubo_to_ising(
        matrix,
        offset,
    )

    for bits in itertools.product(
        [0, 1],
        repeat=3,
    ):
        state = np.array(bits)

        expected = (
            state @ matrix @ state
            + offset
        )

        assert ising_energy(
            ising,
            state,
        ) == pytest.approx(
            expected
        )


def test_operator_energy(matrix):
    ising = qubo_to_ising(
        matrix,
        5.0,
    )

    for bits in itertools.product(
        [0, 1],
        repeat=3,
    ):
        assert operator_energy(
            ising,
            bits,
        ) == pytest.approx(
            ising_energy(
                ising,
                bits,
            )
        )


def test_qiskit_statevector_energy(matrix):
    ising = qubo_to_ising(
        matrix,
        5.0,
    )

    for bits in itertools.product(
        [0, 1],
        repeat=3,
    ):
        bitstring = state_to_bitstring(
            bits
        )

        statevector = Statevector.from_label(
            bitstring
        )

        energy = float(
            np.real(
                statevector.expectation_value(
                    ising.operator
                )
            )
        ) + ising.offset

        assert energy == pytest.approx(
            ising_energy(
                ising,
                bits,
            )
        )


def test_bitstring_conversion():
    state = np.array(
        [1, 0, 0, 1, 1]
    )

    bitstring = state_to_bitstring(
        state
    )

    assert bitstring == "11001"

    assert np.array_equal(
        bitstring_to_state(
            bitstring
        ),
        state,
    )


def test_invalid_bitstring():
    with pytest.raises(ValueError):
        bitstring_to_state(
            "10201"
        )


def test_invalid_matrix():
    with pytest.raises(ValueError):
        qubo_to_ising(
            np.ones((2, 3))
        )


def test_lower_triangular_matrix():
    with pytest.raises(ValueError):
        qubo_to_ising(
            np.array(
                [
                    [1.0, 0.0],
                    [1.0, 2.0],
                ]
            )
        )


def test_zero_matrix():
    ising = qubo_to_ising(
        np.zeros((3, 3))
    )

    assert ising_energy(
        ising,
        [1, 0, 1],
    ) == 0.0


def test_exhaustive_verification(matrix):
    ising = qubo_to_ising(
        matrix,
        5.0,
    )

    result, _, _ = verify_all_states(
        matrix,
        5.0,
        ising,
    )

    assert result[
        "states_verified"
    ] == 8

    assert result[
        "verification_passed"
    ]

    assert result[
        "maximum_energy_error"
    ] < 1e-7


def test_minimum_energy_equivalence(matrix):
    ising = qubo_to_ising(
        matrix,
        5.0,
    )

    result, _, _ = verify_all_states(
        matrix,
        5.0,
        ising,
    )

    assert result[
        "minimum_qubo_energy"
    ] == pytest.approx(
        result[
            "minimum_ising_energy"
        ]
    )