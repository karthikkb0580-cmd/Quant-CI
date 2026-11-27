import json
import time

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp

from src.config import (
    QUBO_RESULTS_DIR,
    QUBO_REPORT_FILE,
    QUBO_MATRIX_FILE,
    QUBO_EXACT_SOLUTION_FILE,
    ISING_REPORT_FILE,
    ISING_TERMS_FILE,
    ISING_ENERGY_PLOT,
    ISING_VERIFICATION_TOLERANCE,
    ISING_MAX_VERIFICATION_VARIABLES,
    ISING_VERIFICATION_BATCH_SIZE,
)


@dataclass
class IsingProblem:
    linear: np.ndarray
    quadratic: np.ndarray
    offset: float
    operator: SparsePauliOp

    @property
    def size(self):
        return len(self.linear)


def validate_qubo_matrix(matrix):
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "QUBO matrix must be two-dimensional."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "QUBO matrix must be square."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "QUBO matrix cannot be empty."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "QUBO matrix contains invalid values."
        )

    if not np.allclose(
        np.tril(matrix, -1),
        0,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError(
            "Expected upper-triangular QUBO matrix."
        )

    return matrix


def create_pauli_label(size, indices):
    label = ["I"] * size

    for index in indices:
        if index < 0 or index >= size:
            raise ValueError(
                "Invalid qubit index."
            )

        label[size - index - 1] = "Z"

    return "".join(label)


def create_ising_operator(linear, quadratic):
    size = len(linear)

    terms = []

    for i in range(size):
        if linear[i] != 0:
            terms.append(
                (
                    create_pauli_label(
                        size,
                        [i],
                    ),
                    float(linear[i]),
                )
            )

    for i in range(size):
        for j in range(i + 1, size):
            if quadratic[i, j] != 0:
                terms.append(
                    (
                        create_pauli_label(
                            size,
                            [i, j],
                        ),
                        float(
                            quadratic[i, j]
                        ),
                    )
                )

    if not terms:
        terms.append(
            (
                "I" * size,
                0.0,
            )
        )

    return SparsePauliOp.from_list(
        terms
    )


def qubo_to_ising(matrix, offset=0.0):
    matrix = validate_qubo_matrix(
        matrix
    )

    if not np.isfinite(offset):
        raise ValueError(
            "Invalid QUBO offset."
        )

    size = matrix.shape[0]

    linear = np.zeros(
        size,
        dtype=float,
    )

    quadratic = np.zeros(
        (size, size),
        dtype=float,
    )

    ising_offset = float(offset)

    for i in range(size):
        coefficient = matrix[i, i]

        ising_offset += (
            coefficient / 2
        )

        linear[i] -= (
            coefficient / 2
        )

    for i in range(size):
        for j in range(i + 1, size):
            coefficient = matrix[i, j]

            ising_offset += (
                coefficient / 4
            )

            linear[i] -= (
                coefficient / 4
            )

            linear[j] -= (
                coefficient / 4
            )

            quadratic[i, j] += (
                coefficient / 4
            )

    operator = create_ising_operator(
        linear,
        quadratic,
    )

    return IsingProblem(
        linear=linear,
        quadratic=quadratic,
        offset=ising_offset,
        operator=operator,
    )


def validate_state(state, size):
    state = np.asarray(
        state,
        dtype=float,
    )

    if state.shape != (size,):
        raise ValueError(
            "Invalid binary state size."
        )

    if not np.isin(state, [0, 1]).all():
        raise ValueError(
            "State must be binary."
        )

    return state


def binary_to_spin(state):
    state = np.asarray(
        state,
        dtype=float,
    )

    if not np.isin(state, [0, 1]).all():
        raise ValueError(
            "State must be binary."
        )

    return 1 - 2 * state


def spin_to_binary(spins):
    spins = np.asarray(
        spins,
        dtype=float,
    )

    if not np.isin(spins, [-1, 1]).all():
        raise ValueError(
            "Spins must be -1 or 1."
        )

    return (
        (1 - spins) / 2
    ).astype(np.int8)


def ising_energy(ising, state):
    state = validate_state(
        state,
        ising.size,
    )

    spins = binary_to_spin(
        state
    )

    return float(
        ising.offset
        + ising.linear @ spins
        + spins @ ising.quadratic @ spins
    )


def state_to_bitstring(state):
    state = np.asarray(state)

    if state.ndim != 1:
        raise ValueError(
            "Expected one-dimensional state."
        )

    if not np.isin(state, [0, 1]).all():
        raise ValueError(
            "State must be binary."
        )

    return "".join(
        str(int(value))
        for value in state[::-1]
    )


def bitstring_to_state(bitstring):
    if not bitstring:
        raise ValueError(
            "Bitstring cannot be empty."
        )

    if any(
        value not in "01"
        for value in bitstring
    ):
        raise ValueError(
            "Invalid bitstring."
        )

    return np.array(
        [
            int(value)
            for value in bitstring[::-1]
        ],
        dtype=np.int8,
    )


def operator_energy(ising, state):
    state = validate_state(
        state,
        ising.size,
    )

    spins = binary_to_spin(
        state
    )

    energy = 0.0

    for label, coefficient in (
        ising.operator.to_list()
    ):
        value = 1.0

        for index, symbol in enumerate(
            reversed(label)
        ):
            if symbol == "Z":
                value *= spins[index]

            elif symbol != "I":
                raise ValueError(
                    "Unexpected Pauli operator."
                )

        energy += (
            float(np.real(coefficient))
            * value
        )

    return float(
        energy + ising.offset
    )


def verify_all_states(
    matrix,
    qubo_offset,
    ising,
):
    matrix = validate_qubo_matrix(
        matrix
    )

    size = matrix.shape[0]

    if size != ising.size:
        raise ValueError(
            "QUBO and Ising dimensions differ."
        )

    if size > ISING_MAX_VERIFICATION_VARIABLES:
        raise ValueError(
            "Verification variable limit exceeded."
        )

    total_states = 2 ** size

    shifts = np.arange(
        size,
        dtype=np.uint64,
    )

    maximum_error = 0.0

    minimum_qubo_energy = float("inf")

    minimum_ising_energy = float("inf")

    sample_qubo = []

    sample_ising = []

    start = time.perf_counter()

    for lower in range(
        0,
        total_states,
        ISING_VERIFICATION_BATCH_SIZE,
    ):
        upper = min(
            lower
            + ISING_VERIFICATION_BATCH_SIZE,
            total_states,
        )

        integers = np.arange(
            lower,
            upper,
            dtype=np.uint64,
        )

        states = (
            (
                integers[:, None]
                >> shifts
            ) & 1
        ).astype(float)

        spins = 1 - 2 * states

        qubo_energies = np.einsum(
            "bi,ij,bj->b",
            states,
            matrix,
            states,
            optimize=True,
        ) + qubo_offset

        ising_energies = (
            ising.offset
            + spins @ ising.linear
            + np.einsum(
                "bi,ij,bj->b",
                spins,
                ising.quadratic,
                spins,
                optimize=True,
            )
        )

        error = np.max(
            np.abs(
                qubo_energies
                - ising_energies
            )
        )

        maximum_error = max(
            maximum_error,
            float(error),
        )

        minimum_qubo_energy = min(
            minimum_qubo_energy,
            float(
                qubo_energies.min()
            ),
        )

        minimum_ising_energy = min(
            minimum_ising_energy,
            float(
                ising_energies.min()
            ),
        )

        sample_qubo.extend(
            qubo_energies[::32].tolist()
        )

        sample_ising.extend(
            ising_energies[::32].tolist()
        )

    result = {
        "states_verified": total_states,
        "maximum_energy_error": maximum_error,
        "minimum_qubo_energy": minimum_qubo_energy,
        "minimum_ising_energy": minimum_ising_energy,
        "verification_seconds": (
            time.perf_counter() - start
        ),
        "verification_passed": (
            maximum_error
            <= ISING_VERIFICATION_TOLERANCE
        ),
    }

    return (
        result,
        sample_qubo,
        sample_ising,
    )


def generate_energy_plot(
    qubo_energies,
    ising_energies,
):
    ISING_ENERGY_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lower = min(
        min(qubo_energies),
        min(ising_energies),
    )

    upper = max(
        max(qubo_energies),
        max(ising_energies),
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.scatter(
        qubo_energies,
        ising_energies,
        alpha=0.5,
        s=12,
        label="Verified states",
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Perfect equivalence",
    )

    plt.xlabel("QUBO energy")

    plt.ylabel("Ising energy")

    plt.title(
        "QUBO-Ising Energy Equivalence"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ISING_ENERGY_PLOT,
        dpi=200,
    )

    plt.close()


def main():
    QUBO_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    qubo_report = json.loads(
        QUBO_REPORT_FILE.read_text(
            encoding="utf-8"
        )
    )

    exact_report = json.loads(
        QUBO_EXACT_SOLUTION_FILE.read_text(
            encoding="utf-8"
        )
    )

    matrix = pd.read_csv(
        QUBO_MATRIX_FILE,
        index_col=0,
    ).to_numpy(
        dtype=float
    )

    qubo_offset = qubo_report[
        "constant_offset"
    ]

    ising = qubo_to_ising(
        matrix,
        qubo_offset,
    )

    verification, qubo_values, ising_values = (
        verify_all_states(
            matrix,
            qubo_offset,
            ising,
        )
    )

    if not verification[
        "verification_passed"
    ]:
        raise RuntimeError(
            "Ising energy verification failed."
        )

    exact_state = np.array(
        exact_report["binary_state"],
        dtype=np.int8,
    )

    exact_ising_energy = ising_energy(
        ising,
        exact_state,
    )

    exact_operator_energy = operator_energy(
        ising,
        exact_state,
    )

    exact_qubo_energy = exact_report[
        "energy"
    ]

    if not np.isclose(
        exact_ising_energy,
        exact_qubo_energy,
        atol=ISING_VERIFICATION_TOLERANCE,
        rtol=0,
    ):
        raise RuntimeError(
            "Exact state energy mismatch."
        )

    if not np.isclose(
        exact_operator_energy,
        exact_qubo_energy,
        atol=ISING_VERIFICATION_TOLERANCE,
        rtol=0,
    ):
        raise RuntimeError(
            "Pauli operator energy mismatch."
        )

    if not np.isclose(
        verification["minimum_ising_energy"],
        exact_qubo_energy,
        atol=ISING_VERIFICATION_TOLERANCE,
        rtol=0,
    ):
        raise RuntimeError(
            "Minimum energy mismatch."
        )

    terms = pd.DataFrame(
        [
            {
                "pauli": label,
                "coefficient": float(
                    np.real(coefficient)
                ),
            }
            for label, coefficient
            in ising.operator.to_list()
        ]
    )

    terms.to_csv(
        ISING_TERMS_FILE,
        index=False,
    )

    report = {
        "run_id": qubo_report["run_id"],
        "qubits": ising.size,
        "test_qubits": qubo_report[
            "test_variables"
        ],
        "slack_qubits": qubo_report[
            "slack_variables"
        ],
        "linear_terms": int(
            np.count_nonzero(
                ising.linear
            )
        ),
        "quadratic_terms": int(
            np.count_nonzero(
                ising.quadratic
            )
        ),
        "pauli_terms": len(
            ising.operator
        ),
        "ising_offset": ising.offset,
        "exact_qubo_energy": exact_qubo_energy,
        "exact_ising_energy": exact_ising_energy,
        "exact_operator_energy": exact_operator_energy,
        "exact_operator_energy_without_offset": (
            exact_operator_energy
            - ising.offset
        ),
        "optimal_bitstring": (
            state_to_bitstring(
                exact_state
            )
        ),
        "optimal_binary_state": (
            exact_state.tolist()
        ),
        "bit_order": "qiskit_little_endian",
        "identity_offset_in_operator": False,
        "verification": verification,
    }

    ISING_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    generate_energy_plot(
        qubo_values,
        ising_values,
    )

    print("=" * 70)

    print(
        "QuantCI - Ising Hamiltonian Conversion"
    )

    print("=" * 70)

    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    print("=" * 70)

    print(
        "ISING HAMILTONIAN VERIFICATION COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()