import json
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from src.config import (
    RANDOM_SEED,
    QAOA_REPS,
    QAOA_SHOTS,
    QAOA_MAX_ITERATIONS,
    QAOA_INITIAL_POINT,
    QAOA_MAX_PARALLEL_THREADS,
    QAOA_TRANSPILER_OPTIMIZATION_LEVEL,
    QAOA_ENERGY_TOLERANCE,
    QUBO_RESULTS_DIR,
    QUBO_REPORT_FILE,
    QUBO_MATRIX_FILE,
    QUBO_EXACT_SOLUTION_FILE,
    ISING_REPORT_FILE,
    ISING_TERMS_FILE,
    QAOA_REPORT_FILE,
    QAOA_MEASUREMENTS_FILE,
    QAOA_SELECTION_FILE,
    QAOA_HISTORY_FILE,
    QAOA_CONVERGENCE_PLOT,
    QAOA_DISTRIBUTION_PLOT,
)

from src.quantum.ising import (
    bitstring_to_state,
    qubo_to_ising,
)

from src.optimization.objective import (
    create_problem,
    evaluate_selection,
)


def normalize_hamiltonian(operator):
    coefficients = np.asarray(
        operator.coeffs
    )

    if not np.isfinite(coefficients).all():
        raise ValueError(
            "Invalid Hamiltonian coefficients."
        )

    scale = float(
        np.max(
            np.abs(coefficients)
        )
    )

    if scale == 0:
        scale = 1.0

    normalized = operator / scale

    return normalized, scale


def calculate_qubo_energy(
    matrix,
    offset,
    state,
):
    state = np.asarray(
        state,
        dtype=float,
    )

    if state.shape != (
        matrix.shape[0],
    ):
        raise ValueError(
            "Invalid state dimensions."
        )

    if not np.isin(
        state,
        [0, 1],
    ).all():
        raise ValueError(
            "State must be binary."
        )

    return float(
        state @ matrix @ state
        + offset
    )


def decode_measurement(
    bitstring,
    qubo_report,
):
    state = bitstring_to_state(
        bitstring
    )

    test_count = qubo_report[
        "test_variables"
    ]

    total_count = qubo_report[
        "total_binary_variables"
    ]

    if len(state) != total_count:
        raise ValueError(
            "Unexpected measurement length."
        )

    selection = state[:test_count]

    slack_bits = state[test_count:]

    duration_units = np.asarray(
        qubo_report["duration_units"],
        dtype=int,
    )

    slack_weights = np.asarray(
        qubo_report["slack_weights"],
        dtype=int,
    )

    used_units = int(
        selection @ duration_units
    )

    slack_units = int(
        slack_bits @ slack_weights
    )

    budget_units = int(
        qubo_report["budget_units"]
    )

    return {
        "state": state,
        "selection": selection,
        "slack_bits": slack_bits,
        "used_duration_units": used_units,
        "slack_units": slack_units,
        "budget_units": budget_units,
        "budget_feasible": (
            used_units <= budget_units
        ),
        "constraint_satisfied": (
            used_units + slack_units
            == budget_units
        ),
    }


def approximation_ratio(
    objective,
    optimal_objective,
):
    if optimal_objective <= 0:
        return None

    return float(
        objective / optimal_objective
    )


def create_backend():
    return AerSimulator(
        method="statevector",
        max_parallel_threads=(
            QAOA_MAX_PARALLEL_THREADS
        ),
    )


def create_sampler():
    return SamplerV2(
        default_shots=QAOA_SHOTS,
        seed=RANDOM_SEED,
        options={
            "backend_options": {
                "method": "statevector",
                "max_parallel_threads": (
                    QAOA_MAX_PARALLEL_THREADS
                ),
            }
        },
    )


def optimize_hamiltonian(
    operator,
    backend,
):
    algorithm_globals.random_seed = (
        RANDOM_SEED
    )

    history = []

    def callback(
        evaluation,
        parameters,
        energy,
        metadata,
    ):
        history.append(
            {
                "evaluation": int(
                    evaluation
                ),
                "normalized_energy": float(
                    np.real(energy)
                ),
                "parameters": json.dumps(
                    np.asarray(
                        parameters
                    ).tolist()
                ),
            }
        )

    pass_manager = (
        generate_preset_pass_manager(
            backend=backend,
            optimization_level=(
                QAOA_TRANSPILER_OPTIMIZATION_LEVEL
            ),
            seed_transpiler=RANDOM_SEED,
        )
    )

    optimizer = COBYLA(
        maxiter=QAOA_MAX_ITERATIONS
    )

    algorithm = QAOA(
        sampler=create_sampler(),
        optimizer=optimizer,
        reps=QAOA_REPS,
        initial_point=np.asarray(
            QAOA_INITIAL_POINT,
            dtype=float,
        ),
        callback=callback,
        transpiler=pass_manager,
    )

    start = time.perf_counter()

    result = (
        algorithm.compute_minimum_eigenvalue(
            operator
        )
    )

    elapsed = (
        time.perf_counter() - start
    )

    return (
        result,
        history,
        elapsed,
    )


def sample_optimized_circuit(
    result,
    backend,
):
    circuit = result.optimal_circuit.copy()

    if circuit.num_parameters:
        circuit = circuit.assign_parameters(
            result.optimal_parameters
        )

    circuit = (
        circuit.remove_final_measurements(
            inplace=False
        )
    )

    circuit.measure_all()

    compiled = transpile(
        circuit,
        backend,
        optimization_level=(
            QAOA_TRANSPILER_OPTIMIZATION_LEVEL
        ),
        seed_transpiler=RANDOM_SEED,
    )

    start = time.perf_counter()

    execution = backend.run(
        compiled,
        shots=QAOA_SHOTS,
        seed_simulator=RANDOM_SEED,
    ).result()

    elapsed = (
        time.perf_counter() - start
    )

    counts = execution.get_counts()

    if sum(counts.values()) != QAOA_SHOTS:
        raise RuntimeError(
            "Unexpected measurement count."
        )

    return (
        counts,
        compiled,
        elapsed,
    )


def evaluate_measurements(
    counts,
    matrix,
    qubo_report,
    problem,
):
    records = []

    for raw_bitstring, count in counts.items():
        bitstring = raw_bitstring.replace(
            " ",
            "",
        )

        decoded = decode_measurement(
            bitstring,
            qubo_report,
        )

        evaluation = evaluate_selection(
            problem,
            decoded["selection"],
        )

        energy = calculate_qubo_energy(
            matrix,
            qubo_report["constant_offset"],
            decoded["state"],
        )

        records.append(
            {
                "bitstring": bitstring,
                "count": int(count),
                "probability": (
                    count / QAOA_SHOTS
                ),
                "qubo_energy": energy,
                "objective_value": evaluation[
                    "objective_value"
                ],
                "selected_count": evaluation[
                    "selected_count"
                ],
                "estimated_duration_seconds": evaluation[
                    "estimated_duration_seconds"
                ],
                "used_duration_units": decoded[
                    "used_duration_units"
                ],
                "slack_units": decoded[
                    "slack_units"
                ],
                "budget_feasible": decoded[
                    "budget_feasible"
                ],
                "constraint_satisfied": decoded[
                    "constraint_satisfied"
                ],
                "original_budget_feasible": evaluation[
                    "feasible"
                ],
            }
        )

    return pd.DataFrame(
        records
    ).sort_values(
        ["qubo_energy", "bitstring"]
    ).reset_index(
        drop=True
    )


def find_best_feasible_measurement(
    measurements,
):
    feasible = measurements[
        measurements["budget_feasible"]
        & measurements["constraint_satisfied"]
        & measurements["original_budget_feasible"]
    ]

    if feasible.empty:
        return None

    return feasible.sort_values(
        [
            "objective_value",
            "count",
            "bitstring",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).iloc[0]


def generate_plots(
    history,
    measurements,
    scale,
    offset,
    exact_energy,
):
    if not history.empty:
        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            history["evaluation"],
            history["normalized_energy"]
            * scale
            + offset,
            label="QAOA expected energy",
        )

        plt.axhline(
            exact_energy,
            linestyle="--",
            label="Exact minimum energy",
        )

        plt.xlabel(
            "Optimizer evaluation"
        )

        plt.ylabel(
            "Expected QUBO energy"
        )

        plt.title(
            "QAOA Optimization Convergence"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            QAOA_CONVERGENCE_PLOT,
            dpi=200,
        )

        plt.close()

    top = measurements.nlargest(
        15,
        "count",
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        top["bitstring"],
        top["probability"],
    )

    plt.xticks(
        rotation=60
    )

    plt.xlabel(
        "Measured bitstring"
    )

    plt.ylabel(
        "Measurement probability"
    )

    plt.title(
        "Most Frequent QAOA Measurements"
    )

    plt.tight_layout()

    plt.savefig(
        QAOA_DISTRIBUTION_PLOT,
        dpi=200,
    )

    plt.close()


def main():
    total_start = time.perf_counter()

    QUBO_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    qubo_report = json.loads(
        QUBO_REPORT_FILE.read_text(
            encoding="utf-8"
        )
    )

    ising_report = json.loads(
        ISING_REPORT_FILE.read_text(
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

    candidates = pd.read_csv(
        QUBO_RESULTS_DIR
        / "qubo_candidates.csv"
    )

    problem = create_problem(
        candidates,
        budget=qubo_report[
            "budget_seconds"
        ],
        redundancy_penalty=qubo_report[
            "redundancy_penalty"
        ],
    )

    if problem.size != qubo_report[
        "test_variables"
    ]:
        raise RuntimeError(
            "Candidate count mismatch."
        )

    terms = pd.read_csv(
        ISING_TERMS_FILE
    )

    operator = SparsePauliOp.from_list(
        list(
            zip(
                terms["pauli"],
                terms["coefficient"],
            )
        )
    )

    reconstructed = qubo_to_ising(
        matrix,
        qubo_report["constant_offset"],
    )

    if not operator.equiv(
        reconstructed.operator
    ):
        raise RuntimeError(
            "Hamiltonian mismatch."
        )

    normalized, scale = (
        normalize_hamiltonian(
            operator
        )
    )

    backend = create_backend()

    print(
        "Starting QAOA optimization..."
    )

    result, history, optimization_seconds = (
        optimize_hamiltonian(
            normalized,
            backend,
        )
    )

    counts, circuit, sampling_seconds = (
        sample_optimized_circuit(
            result,
            backend,
        )
    )

    measurements = evaluate_measurements(
        counts,
        matrix,
        qubo_report,
        problem,
    )

    best = find_best_feasible_measurement(
        measurements
    )

    history_frame = pd.DataFrame(
        history
    )

    history_frame.to_csv(
        QAOA_HISTORY_FILE,
        index=False,
    )

    measurements.to_csv(
        QAOA_MEASUREMENTS_FILE,
        index=False,
    )

    feasible_mask = (
        measurements["budget_feasible"]
        & measurements["constraint_satisfied"]
        & measurements["original_budget_feasible"]
    )

    feasible_shots = int(
        measurements.loc[
            feasible_mask,
            "count",
        ].sum()
    )

    optimal_mask = np.isclose(
        measurements["qubo_energy"],
        exact_report["energy"],
        atol=QAOA_ENERGY_TOLERANCE,
        rtol=0,
    )

    optimal_shots = int(
        measurements.loc[
            optimal_mask,
            "count",
        ].sum()
    )

    report = {
        "algorithm": "qaoa",
        "partition": qubo_report[
            "partition"
        ],
        "run_id": qubo_report[
            "run_id"
        ],
        "backend": backend.name,
        "simulation_method": "statevector",
        "qubits": operator.num_qubits,
        "reps": QAOA_REPS,
        "shots": QAOA_SHOTS,
        "random_seed": RANDOM_SEED,
        "optimizer": "COBYLA",
        "optimizer_evaluations": int(
            result.cost_function_evals
        ),
        "hamiltonian_scale": scale,
        "ising_offset": ising_report[
            "ising_offset"
        ],
        "normalized_expected_energy": float(
            np.real(
                result.eigenvalue
            )
        ),
        "expected_qubo_energy": float(
            np.real(
                result.eigenvalue
            )
            * scale
            + ising_report[
                "ising_offset"
            ]
        ),
        "optimal_parameters": (
            result.optimal_point.tolist()
        ),
        "unique_measurements": len(
            measurements
        ),
        "feasible_shots": feasible_shots,
        "feasible_measurement_rate": (
            feasible_shots / QAOA_SHOTS
        ),
        "optimal_shots": optimal_shots,
        "optimal_measurement_rate": (
            optimal_shots / QAOA_SHOTS
        ),
        "exact_energy": exact_report[
            "energy"
        ],
        "exact_objective": exact_report[
            "objective_value"
        ],
        "circuit_depth": circuit.depth(),
        "circuit_operations": {
            str(key): int(value)
            for key, value
            in circuit.count_ops().items()
        },
        "optimization_seconds": (
            optimization_seconds
        ),
        "sampling_seconds": (
            sampling_seconds
        ),
        "feasible_solution_found": (
            best is not None
        ),
    }

    if best is not None:
        decoded = decode_measurement(
            best["bitstring"],
            qubo_report,
        )

        selection = decoded[
            "selection"
        ]

        evaluation = evaluate_selection(
            problem,
            selection,
        )

        if not evaluation["feasible"]:
            raise RuntimeError(
                "Infeasible QAOA solution."
            )

        objective = evaluation[
            "objective_value"
        ]

        optimal_objective = exact_report[
            "objective_value"
        ]

        if objective > (
            optimal_objective
            + QAOA_ENERGY_TOLERANCE
        ):
            raise RuntimeError(
                "Objective exceeds exact optimum."
            )

        report.update(
            {
                "best_feasible_bitstring": (
                    best["bitstring"]
                ),
                "best_feasible_energy": float(
                    best["qubo_energy"]
                ),
                "best_feasible_objective": (
                    objective
                ),
                "approximation_ratio": (
                    approximation_ratio(
                        objective,
                        optimal_objective,
                    )
                ),
                "objective_gap": (
                    optimal_objective
                    - objective
                ),
                "selection": selection.tolist(),
                "selection_evaluation": evaluation,
            }
        )

        selected = candidates.loc[
            selection.astype(bool)
        ].copy()

        selected.to_csv(
            QAOA_SELECTION_FILE,
            index=False,
        )

    else:
        report.update(
            {
                "best_feasible_bitstring": None,
                "best_feasible_energy": None,
                "best_feasible_objective": None,
                "approximation_ratio": None,
                "objective_gap": None,
                "selection": None,
                "selection_evaluation": None,
            }
        )

        candidates.iloc[:0].to_csv(
            QAOA_SELECTION_FILE,
            index=False,
        )

    generate_plots(
        history_frame,
        measurements,
        scale,
        ising_report["ising_offset"],
        exact_report["energy"],
    )

    report["total_seconds"] = (
        time.perf_counter()
        - total_start
    )

    QAOA_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("QuantCI - QAOA Optimization")
    print("=" * 70)

    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    print("=" * 70)
    print("QAOA EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()