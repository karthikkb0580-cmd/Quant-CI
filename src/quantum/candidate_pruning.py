import json
import time

import numpy as np
import pandas as pd

from src.config import (
    QUBO_RESULTS_DIR,
    QUBO_REPORT_FILE,
    QUBO_MATRIX_FILE,
    QUBO_EXACT_SOLUTION_FILE,
    PRUNED_RESULTS_DIR,
    PRUNED_CANDIDATES_FILE,
    PRUNED_REMOVED_FILE,
    PRUNED_MATRIX_FILE,
    PRUNED_REPORT_FILE,
    PRUNED_EXACT_FILE,
    PRUNED_ISING_TERMS_FILE,
    PRUNING_TOLERANCE,
)

from src.quantum.ising import (
    qubo_to_ising,
    verify_all_states,
    state_to_bitstring,
)

from src.optimization.objective import (
    create_problem,
    evaluate_selection,
)


def find_retained_indices(
    duration_units,
    budget_units,
):
    durations = np.asarray(
        duration_units,
        dtype=float,
    )

    if durations.ndim != 1:
        raise ValueError(
            "Expected one-dimensional durations."
        )

    if not np.isfinite(durations).all():
        raise ValueError(
            "Invalid durations."
        )

    if np.any(durations <= 0):
        raise ValueError(
            "Durations must be positive."
        )

    if not np.isfinite(budget_units):
        raise ValueError(
            "Invalid budget."
        )

    if budget_units < 0:
        raise ValueError(
            "Budget cannot be negative."
        )

    retained = np.flatnonzero(
        durations <= budget_units
    )

    removed = np.flatnonzero(
        durations > budget_units
    )

    return retained, removed


def prune_qubo_matrix(
    matrix,
    retained_indices,
    test_count,
):
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Invalid matrix dimensions."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Matrix must be square."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Invalid matrix values."
        )

    if test_count < 0 or test_count > matrix.shape[0]:
        raise ValueError(
            "Invalid test count."
        )

    retained_indices = np.asarray(
        retained_indices,
        dtype=int,
    )

    if np.any(retained_indices < 0):
        raise ValueError(
            "Invalid retained index."
        )

    if np.any(retained_indices >= test_count):
        raise ValueError(
            "Invalid retained index."
        )

    if len(np.unique(retained_indices)) != len(retained_indices):
        raise ValueError(
            "Duplicate retained indices."
        )

    slack_indices = np.arange(
        test_count,
        matrix.shape[0],
    )

    retained_variables = np.concatenate(
        [
            retained_indices,
            slack_indices,
        ]
    )

    reduced_matrix = matrix[
        np.ix_(
            retained_variables,
            retained_variables,
        )
    ]

    return (
        reduced_matrix,
        retained_variables,
    )


def restore_original_state(
    reduced_state,
    retained_variables,
    original_size,
):
    reduced_state = np.asarray(
        reduced_state,
        dtype=np.int8,
    )

    retained_variables = np.asarray(
        retained_variables,
        dtype=int,
    )

    if len(reduced_state) != len(retained_variables):
        raise ValueError(
            "State size mismatch."
        )

    if not np.isin(
        reduced_state,
        [0, 1],
    ).all():
        raise ValueError(
            "State must be binary."
        )

    original_state = np.zeros(
        original_size,
        dtype=np.int8,
    )

    original_state[
        retained_variables
    ] = reduced_state

    return original_state


def solve_reduced_exact(
    matrix,
    offset,
):
    size = matrix.shape[0]

    if size > 20:
        raise ValueError(
            "Exact enumeration limit exceeded."
        )

    start = time.perf_counter()

    integers = np.arange(
        2 ** size,
        dtype=np.uint64,
    )

    shifts = np.arange(
        size,
        dtype=np.uint64,
    )

    states = (
        (
            integers[:, None]
            >> shifts
        ) & 1
    ).astype(np.int8)

    energies = np.einsum(
        "bi,ij,bj->b",
        states,
        matrix,
        states,
        optimize=True,
    ) + offset

    best_index = int(
        np.argmin(energies)
    )

    return {
        "state": states[
            best_index
        ],
        "energy": float(
            energies[best_index]
        ),
        "states_evaluated": len(
            states
        ),
        "solver_seconds": (
            time.perf_counter() - start
        ),
    }


def main():
    PRUNED_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    qubo_report = json.loads(
        QUBO_REPORT_FILE.read_text(
            encoding="utf-8"
        )
    )

    original_exact = json.loads(
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

    candidates = pd.read_csv(
        QUBO_RESULTS_DIR
        / "qubo_candidates.csv"
    )

    test_count = int(
        qubo_report["test_variables"]
    )

    duration_units = np.asarray(
        qubo_report["duration_units"],
        dtype=int,
    )

    budget_units = int(
        qubo_report["budget_units"]
    )

    offset = float(
        qubo_report["constant_offset"]
    )

    if len(candidates) != test_count:
        raise RuntimeError(
            "Candidate count mismatch."
        )

    if len(duration_units) != test_count:
        raise RuntimeError(
            "Duration count mismatch."
        )

    if matrix.shape[0] != qubo_report["total_binary_variables"]:
        raise RuntimeError(
            "QUBO dimension mismatch."
        )

    retained, removed = (
        find_retained_indices(
            duration_units,
            budget_units,
        )
    )

    if len(retained) == 0:
        raise RuntimeError(
            "No individually feasible candidates."
        )

    reduced_matrix, retained_variables = (
        prune_qubo_matrix(
            matrix,
            retained,
            test_count,
        )
    )

    retained_candidates = candidates.iloc[
        retained
    ].copy().reset_index(
        drop=True
    )

    removed_candidates = candidates.iloc[
        removed
    ].copy()

    removed_candidates[
        "duration_units"
    ] = duration_units[
        removed
    ]

    removed_candidates[
        "removal_reason"
    ] = "individually_infeasible"

    retained_candidates.to_csv(
        PRUNED_CANDIDATES_FILE,
        index=False,
    )

    removed_candidates.to_csv(
        PRUNED_REMOVED_FILE,
        index=False,
    )

    pd.DataFrame(
        reduced_matrix
    ).to_csv(
        PRUNED_MATRIX_FILE
    )

    exact = solve_reduced_exact(
        reduced_matrix,
        offset,
    )

    restored_state = (
        restore_original_state(
            exact["state"],
            retained_variables,
            matrix.shape[0],
        )
    )

    restored_energy = float(
        restored_state
        @ matrix
        @ restored_state
        + offset
    )

    original_energy = float(
        original_exact["energy"]
    )

    if not np.isclose(
        exact["energy"],
        original_energy,
        atol=PRUNING_TOLERANCE,
        rtol=0,
    ):
        raise RuntimeError(
            "Pruning changed the exact optimum."
        )

    if not np.isclose(
        restored_energy,
        exact["energy"],
        atol=PRUNING_TOLERANCE,
        rtol=0,
    ):
        raise RuntimeError(
            "Restored state energy mismatch."
        )

    problem = create_problem(
        retained_candidates,
        budget=qubo_report[
            "budget_seconds"
        ],
        redundancy_penalty=qubo_report[
            "redundancy_penalty"
        ],
    )

    selection = exact[
        "state"
    ][:len(retained)]

    slack_bits = exact[
        "state"
    ][len(retained):]

    used_units = int(
        selection
        @ duration_units[retained]
    )

    slack_units = int(
        slack_bits
        @ np.asarray(
            qubo_report["slack_weights"],
            dtype=int,
        )
    )

    if used_units + slack_units != budget_units:
        raise RuntimeError(
            "Reduced exact constraint violated."
        )

    evaluation = evaluate_selection(
        problem,
        selection,
    )

    if not evaluation["feasible"]:
        raise RuntimeError(
            "Reduced exact solution is infeasible."
        )

    if not np.isclose(
        evaluation["objective_value"],
        original_exact["objective_value"],
        atol=PRUNING_TOLERANCE,
        rtol=0,
    ):
        raise RuntimeError(
            "Reduced objective mismatch."
        )

    original_ising = qubo_to_ising(
        matrix,
        offset,
    )

    reduced_ising = qubo_to_ising(
        reduced_matrix,
        offset,
    )

    verification, _, _ = (
        verify_all_states(
            reduced_matrix,
            offset,
            reduced_ising,
        )
    )

    if not verification[
        "verification_passed"
    ]:
        raise RuntimeError(
            "Reduced Ising verification failed."
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
            in reduced_ising.operator.to_list()
        ]
    )

    terms.to_csv(
        PRUNED_ISING_TERMS_FILE,
        index=False,
    )

    original_scale = float(
        np.max(
            np.abs(
                original_ising.operator.coeffs
            )
        )
    )

    reduced_scale = float(
        np.max(
            np.abs(
                reduced_ising.operator.coeffs
            )
        )
    )

    exact_report = {
        "energy": exact["energy"],
        "binary_state": exact[
            "state"
        ].tolist(),
        "restored_binary_state": (
            restored_state.tolist()
        ),
        "optimal_bitstring": (
            state_to_bitstring(
                exact["state"]
            )
        ),
        "states_evaluated": exact[
            "states_evaluated"
        ],
        "solver_seconds": exact[
            "solver_seconds"
        ],
        "objective_value": evaluation[
            "objective_value"
        ],
        "used_duration_units": used_units,
        "slack_units": slack_units,
        "constraint_satisfied": True,
        "selection_evaluation": evaluation,
    }

    PRUNED_EXACT_FILE.write_text(
        json.dumps(
            exact_report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    report = {
        "experiment": "candidate_pruning",
        "partition": qubo_report[
            "partition"
        ],
        "run_id": qubo_report[
            "run_id"
        ],
        "original_test_variables": test_count,
        "retained_test_variables": len(
            retained
        ),
        "removed_test_variables": len(
            removed
        ),
        "original_qubits": matrix.shape[0],
        "reduced_qubits": reduced_matrix.shape[0],
        "qubits_saved": (
            matrix.shape[0]
            - reduced_matrix.shape[0]
        ),
        "original_state_space": (
            2 ** matrix.shape[0]
        ),
        "reduced_state_space": (
            2 ** reduced_matrix.shape[0]
        ),
        "state_space_reduction_percent": (
            100
            * (
                1
                - (
                    2 ** reduced_matrix.shape[0]
                )
                / (
                    2 ** matrix.shape[0]
                )
            )
        ),
        "original_pauli_terms": len(
            original_ising.operator
        ),
        "reduced_pauli_terms": len(
            reduced_ising.operator
        ),
        "original_hamiltonian_scale": original_scale,
        "reduced_hamiltonian_scale": reduced_scale,
        "original_exact_energy": original_energy,
        "reduced_exact_energy": exact[
            "energy"
        ],
        "restored_exact_energy": restored_energy,
        "original_exact_objective": original_exact[
            "objective_value"
        ],
        "reduced_exact_objective": evaluation[
            "objective_value"
        ],
        "optimality_preserved": True,
        "removed_indices": removed.tolist(),
        "retained_indices": retained.tolist(),
        "retained_variables": retained_variables.tolist(),
        "duration_units": duration_units[
            retained
        ].tolist(),
        "slack_weights": qubo_report[
            "slack_weights"
        ],
        "budget_units": budget_units,
        "budget_seconds": qubo_report[
            "budget_seconds"
        ],
        "constant_offset": offset,
        "ising_offset": reduced_ising.offset,
        "redundancy_penalty": qubo_report[
            "redundancy_penalty"
        ],
        "verification": verification,
    }

    PRUNED_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("QuantCI - Quantum Candidate Pruning")
    print("=" * 70)

    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    print("=" * 70)
    print("CANDIDATE PRUNING VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    main()