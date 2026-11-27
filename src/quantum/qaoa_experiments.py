import json
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from src.config import (
    QUANTUM_EXPERIMENTS_DIR,
    PRUNED_CANDIDATES_FILE,
    PRUNED_MATRIX_FILE,
    PRUNED_REPORT_FILE,
    PRUNED_EXACT_FILE,
    QAOA_REPORT_FILE,
    QAOA_EXPERIMENT_CONFIGURATIONS,
    QUANTUM_EXPERIMENT_SEEDS,
    QAOA_EXPERIMENT_REPORT_FILE,
    QAOA_EXPERIMENT_SUMMARY_FILE,
    QAOA_EXPERIMENT_AGGREGATE_FILE,
    QAOA_EXPERIMENT_RATIO_PLOT,
    QAOA_EXPERIMENT_FEASIBILITY_PLOT,
    QAOA_EXPERIMENT_RUNTIME_PLOT,
    QAOA_MAX_PARALLEL_THREADS,
)

from src.quantum.ising import (
    qubo_to_ising,
    bitstring_to_state,
)

from src.quantum.qaoa_solver import (
    normalize_hamiltonian,
    calculate_qubo_energy,
    approximation_ratio,
)

from src.optimization.objective import (
    create_problem,
    evaluate_selection,
)


def read_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def create_initial_point(reps, seed):
    if reps <= 0:
        raise ValueError(
            "QAOA depth must be positive."
        )

    if seed == 42:
        return np.concatenate(
            [
                np.full(reps, 0.4),
                np.full(reps, 0.8),
            ]
        )

    rng = np.random.default_rng(seed)

    return rng.uniform(
        0.1,
        1.0,
        size=2 * reps,
    )


def decode_state(bitstring, report):
    state = bitstring_to_state(
        bitstring
    )

    test_count = report[
        "retained_test_variables"
    ]

    if len(state) != report["reduced_qubits"]:
        raise ValueError(
            "Invalid measurement size."
        )

    selection = state[:test_count]

    slack = state[test_count:]

    used_units = int(
        selection @ np.asarray(
            report["duration_units"],
            dtype=int,
        )
    )

    slack_units = int(
        slack @ np.asarray(
            report["slack_weights"],
            dtype=int,
        )
    )

    budget = report["budget_units"]

    return {
        "state": state,
        "selection": selection,
        "used_units": used_units,
        "slack_units": slack_units,
        "budget_feasible": used_units <= budget,
        "constraint_satisfied": (
            used_units + slack_units == budget
        ),
    }


def evaluate_counts(
    counts,
    matrix,
    report,
    problem,
    exact_energy,
):
    total_shots = sum(
        counts.values()
    )

    if total_shots <= 0:
        raise ValueError(
            "No measurements received."
        )

    records = []

    for bitstring, count in counts.items():
        bitstring = bitstring.replace(
            " ",
            "",
        )

        decoded = decode_state(
            bitstring,
            report,
        )

        evaluation = evaluate_selection(
            problem,
            decoded["selection"],
        )

        energy = calculate_qubo_energy(
            matrix,
            report["constant_offset"],
            decoded["state"],
        )

        feasible = bool(
            decoded["budget_feasible"]
            and decoded["constraint_satisfied"]
            and evaluation["feasible"]
        )

        optimal = bool(
            feasible
            and np.isclose(
                energy,
                exact_energy,
                atol=1e-7,
                rtol=0,
            )
        )

        records.append(
            {
                "bitstring": bitstring,
                "count": int(count),
                "probability": count / total_shots,
                "energy": energy,
                "objective": evaluation[
                    "objective_value"
                ],
                "selected_count": evaluation[
                    "selected_count"
                ],
                "estimated_duration_seconds": evaluation[
                    "estimated_duration_seconds"
                ],
                "budget_feasible": decoded[
                    "budget_feasible"
                ],
                "constraint_satisfied": decoded[
                    "constraint_satisfied"
                ],
                "feasible": feasible,
                "optimal": optimal,
            }
        )

    return pd.DataFrame(records)


def summarize_measurements(
    measurements,
    exact_objective,
):
    shots = int(
        measurements["count"].sum()
    )

    feasible = measurements[
        measurements["feasible"]
    ]

    feasible_shots = int(
        feasible["count"].sum()
    )

    optimal_shots = int(
        measurements.loc[
            measurements["optimal"],
            "count",
        ].sum()
    )

    result = {
        "shots": shots,
        "unique_measurements": len(
            measurements
        ),
        "feasible_shots": feasible_shots,
        "feasible_measurement_rate": (
            feasible_shots / shots
        ),
        "optimal_shots": optimal_shots,
        "optimal_measurement_rate": (
            optimal_shots / shots
        ),
        "feasible_solution_found": (
            not feasible.empty
        ),
        "best_bitstring": None,
        "best_objective": None,
        "best_energy": None,
        "approximation_ratio": None,
        "objective_gap": None,
        "selected_count": None,
        "estimated_duration_seconds": None,
    }

    if not feasible.empty:
        best = feasible.sort_values(
            [
                "objective",
                "count",
                "bitstring",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).iloc[0]

        objective = float(
            best["objective"]
        )

        if objective > exact_objective + 1e-7:
            raise RuntimeError(
                "Objective exceeds exact optimum."
            )

        result.update(
            {
                "best_bitstring": best[
                    "bitstring"
                ],
                "best_objective": objective,
                "best_energy": float(
                    best["energy"]
                ),
                "approximation_ratio": (
                    approximation_ratio(
                        objective,
                        exact_objective,
                    )
                ),
                "objective_gap": (
                    exact_objective - objective
                ),
                "selected_count": int(
                    best["selected_count"]
                ),
                "estimated_duration_seconds": float(
                    best[
                        "estimated_duration_seconds"
                    ]
                ),
            }
        )

    return result


def run_experiment(
    name,
    reps,
    shots,
    max_iterations,
    seed,
    matrix,
    pruning_report,
    exact_report,
    problem,
):
    total_start = time.perf_counter()

    algorithm_globals.random_seed = seed

    ising = qubo_to_ising(
        matrix,
        pruning_report["constant_offset"],
    )

    operator, scale = normalize_hamiltonian(
        ising.operator
    )

    backend = AerSimulator(
        method="statevector",
        max_parallel_threads=(
            QAOA_MAX_PARALLEL_THREADS
        ),
    )

    sampler = SamplerV2(
        default_shots=shots,
        seed=seed,
        options={
            "backend_options": {
                "method": "statevector",
                "max_parallel_threads": (
                    QAOA_MAX_PARALLEL_THREADS
                ),
            },
        },
    )

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1,
        seed_transpiler=seed,
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
                "evaluation": int(evaluation),
                "normalized_energy": float(
                    np.real(energy)
                ),
                "qubo_energy": float(
                    np.real(energy)
                    * scale
                    + ising.offset
                ),
                "parameters": json.dumps(
                    np.asarray(
                        parameters
                    ).tolist()
                ),
            }
        )

    initial_point = create_initial_point(
        reps,
        seed,
    )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(
            maxiter=max_iterations
        ),
        reps=reps,
        initial_point=initial_point,
        callback=callback,
        transpiler=pass_manager,
    )

    start = time.perf_counter()

    result = qaoa.compute_minimum_eigenvalue(
        operator
    )

    optimization_seconds = (
        time.perf_counter() - start
    )

    circuit = result.optimal_circuit.copy()

    if circuit.num_parameters:
        circuit = circuit.assign_parameters(
            result.optimal_parameters
        )

    circuit = circuit.remove_final_measurements(
        inplace=False
    )

    circuit.measure_all()

    compiled = transpile(
        circuit,
        backend,
        optimization_level=1,
        seed_transpiler=seed,
    )

    start = time.perf_counter()

    counts = backend.run(
        compiled,
        shots=shots,
        seed_simulator=seed,
    ).result().get_counts()

    sampling_seconds = (
        time.perf_counter() - start
    )

    if sum(counts.values()) != shots:
        raise RuntimeError(
            "Measurement count mismatch."
        )

    measurements = evaluate_counts(
        counts,
        matrix,
        pruning_report,
        problem,
        exact_report["energy"],
    )

    summary = summarize_measurements(
        measurements,
        exact_report["objective_value"],
    )

    experiment_dir = (
        QUANTUM_EXPERIMENTS_DIR
        / name
        / f"seed_{seed}"
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    measurements.to_csv(
        experiment_dir / "measurements.csv",
        index=False,
    )

    pd.DataFrame(history).to_csv(
        experiment_dir / "history.csv",
        index=False,
    )

    summary.update(
        {
            "experiment": name,
            "seed": seed,
            "qubits": operator.num_qubits,
            "reps": reps,
            "max_iterations": max_iterations,
            "optimizer_evaluations": int(
                result.cost_function_evals
            ),
            "hamiltonian_scale": scale,
            "exact_objective": exact_report[
                "objective_value"
            ],
            "expected_qubo_energy": float(
                np.real(result.eigenvalue)
                * scale
                + ising.offset
            ),
            "initial_point": initial_point.tolist(),
            "optimal_parameters": (
                result.optimal_point.tolist()
            ),
            "circuit_depth": compiled.depth(),
            "rzz_gates": int(
                compiled.count_ops().get(
                    "rzz",
                    0,
                )
            ),
            "optimization_seconds": optimization_seconds,
            "sampling_seconds": sampling_seconds,
            "total_seconds": (
                time.perf_counter() - total_start
            ),
        }
    )

    (
        experiment_dir / "report.json"
    ).write_text(
        json.dumps(
            summary,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return summary


def aggregate_results(dataframe):
    records = []

    for name, group in dataframe.groupby(
        "experiment",
        sort=False,
    ):
        ratios = group[
            "approximation_ratio"
        ].dropna()

        records.append(
            {
                "experiment": name,
                "runs": len(group),
                "successful_runs": int(
                    group[
                        "feasible_solution_found"
                    ].sum()
                ),
                "optimal_solution_runs": int(
                    (
                        group["optimal_shots"] > 0
                    ).sum()
                ),
                "mean_approximation_ratio": (
                    float(ratios.mean())
                    if len(ratios)
                    else None
                ),
                "std_approximation_ratio": (
                    float(ratios.std(ddof=0))
                    if len(ratios)
                    else None
                ),
                "best_approximation_ratio": (
                    float(ratios.max())
                    if len(ratios)
                    else None
                ),
                "mean_feasible_measurement_rate": float(
                    group[
                        "feasible_measurement_rate"
                    ].mean()
                ),
                "mean_optimal_measurement_rate": float(
                    group[
                        "optimal_measurement_rate"
                    ].mean()
                ),
                "mean_runtime_seconds": float(
                    group["total_seconds"].mean()
                ),
                "mean_circuit_depth": float(
                    group["circuit_depth"].mean()
                ),
            }
        )

    return pd.DataFrame(records)


def generate_plots(aggregate):
    configurations = aggregate[
        "experiment"
    ]

    plots = [
        (
            "mean_approximation_ratio",
            "Mean Approximation Ratio",
            QAOA_EXPERIMENT_RATIO_PLOT,
        ),
        (
            "mean_feasible_measurement_rate",
            "Mean Feasible Measurement Rate",
            QAOA_EXPERIMENT_FEASIBILITY_PLOT,
        ),
        (
            "mean_runtime_seconds",
            "Mean Runtime (Seconds)",
            QAOA_EXPERIMENT_RUNTIME_PLOT,
        ),
    ]

    for column, title, path in plots:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(
            figsize=(10, 6)
        )

        values = aggregate[column]

        plt.bar(
            configurations,
            values.fillna(0),
        )

        for index, value in enumerate(values):
            if pd.isna(value):
                plt.text(
                    index,
                    0,
                    "N/A",
                    ha="center",
                    va="bottom",
                )

        plt.ylabel(title)

        plt.title(
            f"QuantCI - {title}"
        )

        plt.xticks(
            rotation=25,
            ha="right",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=200,
        )

        plt.close()


def main():
    start = time.perf_counter()

    QUANTUM_EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pruning_report = read_json(
        PRUNED_REPORT_FILE
    )

    exact_report = read_json(
        PRUNED_EXACT_FILE
    )

    baseline = read_json(
        QAOA_REPORT_FILE
    )

    matrix = pd.read_csv(
        PRUNED_MATRIX_FILE,
        index_col=0,
    ).to_numpy(
        dtype=float
    )

    candidates = pd.read_csv(
        PRUNED_CANDIDATES_FILE
    )

    problem = create_problem(
        candidates,
        budget=pruning_report[
            "budget_seconds"
        ],
        redundancy_penalty=pruning_report[
            "redundancy_penalty"
        ],
    )

    if not pruning_report[
        "optimality_preserved"
    ]:
        raise RuntimeError(
            "Pruning verification failed."
        )

    results = []

    for configuration in (
        QAOA_EXPERIMENT_CONFIGURATIONS
    ):
        name, reps, shots, iterations = (
            configuration
        )

        for seed in QUANTUM_EXPERIMENT_SEEDS:
            print(
                f"Running {name} | "
                f"seed={seed} | "
                f"reps={reps} | "
                f"shots={shots}",
                flush=True,
            )

            result = run_experiment(
                name,
                reps,
                shots,
                iterations,
                seed,
                matrix,
                pruning_report,
                exact_report,
                problem,
            )

            results.append(result)

            pd.DataFrame(results).to_csv(
                QAOA_EXPERIMENT_SUMMARY_FILE,
                index=False,
            )

            print(
                "Approximation ratio:",
                result["approximation_ratio"],
            )

            print(
                "Feasible shots:",
                result["feasible_shots"],
            )

            print(
                "Optimal shots:",
                result["optimal_shots"],
            )

            print(
                "Runtime:",
                result["total_seconds"],
            )

    dataframe = pd.DataFrame(results)

    aggregate = aggregate_results(
        dataframe
    )

    aggregate.to_csv(
        QAOA_EXPERIMENT_AGGREGATE_FILE,
        index=False,
    )

    successful = dataframe[
        dataframe["feasible_solution_found"]
    ]

    best = None

    if not successful.empty:
        best_index = successful[
            "approximation_ratio"
        ].idxmax()

        best = results[
            int(best_index)
        ]

    report = {
        "partition": pruning_report[
            "partition"
        ],
        "run_id": pruning_report[
            "run_id"
        ],
        "total_experiments": len(results),
        "configurations": len(
            QAOA_EXPERIMENT_CONFIGURATIONS
        ),
        "seeds": QUANTUM_EXPERIMENT_SEEDS,
        "successful_experiments": int(
            dataframe[
                "feasible_solution_found"
            ].sum()
        ),
        "optimal_solution_experiments": int(
            (
                dataframe["optimal_shots"] > 0
            ).sum()
        ),
        "baseline_approximation_ratio": baseline[
            "approximation_ratio"
        ],
        "baseline_feasible_measurement_rate": baseline[
            "feasible_measurement_rate"
        ],
        "exact_objective": exact_report[
            "objective_value"
        ],
        "best_experiment": best,
        "total_experiment_seconds": (
            time.perf_counter() - start
        ),
        "evaluation_type": (
            "single_validation_instance"
        ),
        "classical_repair_used": False,
    }

    QAOA_EXPERIMENT_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    generate_plots(aggregate)

    print("=" * 70)
    print("QuantCI - QAOA Experiment Results")
    print("=" * 70)

    print(
        aggregate.to_string(
            index=False
        )
    )

    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    print("=" * 70)
    print("QAOA EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()