import json
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import (
    BENCHMARK_RESULTS_DIR,
    PRUNED_REPORT_FILE,
    PRUNED_CANDIDATES_FILE,
    PRUNED_EXACT_FILE,
    RANDOM_QAOA_COMPARISON_FILE,
    UNIFIED_BENCHMARK_REPORT,
    UNIFIED_BENCHMARK_SUMMARY,
    UNIFIED_BENCHMARK_SELECTIONS,
    UNIFIED_OBJECTIVE_PLOT,
    UNIFIED_RATIO_PLOT,
    UNIFIED_RUNTIME_PLOT,
    UNIFIED_SUCCESS_PLOT,
    BENCHMARK_CLASSICAL_REPETITIONS,
    BENCHMARK_TOLERANCE,
    QUBO_DURATION_UNIT_SECONDS,
)

from src.optimization.objective import (
    OptimizationProblem,
    create_problem,
    evaluate_selection,
)

from src.optimization.greedy import solve_greedy

from src.optimization.ortools_solver import solve_ortools


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def create_discrete_problem(problem, report):
    durations = np.asarray(
        report["duration_units"],
        dtype=float,
    )

    if durations.shape != problem.durations.shape:
        raise ValueError(
            "Duration size mismatch."
        )

    unit = float(
        report.get(
            "duration_unit_seconds",
            QUBO_DURATION_UNIT_SECONDS,
        )
    )

    if not np.isfinite(unit) or unit <= 0:
        raise ValueError(
            "Invalid duration unit."
        )

    expected = np.ceil(
        problem.durations / unit
    )

    if not np.array_equal(
        durations,
        expected,
    ):
        raise RuntimeError(
            "Discretization mismatch."
        )

    budget_units = int(
        report["budget_units"]
    )

    expected_budget_units = int(
        np.floor(problem.budget / unit)
    )

    if budget_units != expected_budget_units:
        raise RuntimeError(
            "Budget discretization mismatch."
        )

    budget = budget_units * unit

    return OptimizationProblem(
        test_ids=problem.test_ids,
        risks=problem.risks.copy(),
        durations=durations * unit,
        similarity=problem.similarity.copy(),
        budget=budget,
        redundancy_penalty=problem.redundancy_penalty,
    )


def calculate_ratio(objective, optimum):
    if objective is None:
        return None

    if optimum < -BENCHMARK_TOLERANCE:
        raise ValueError(
            "Invalid optimum."
        )

    if objective > optimum + BENCHMARK_TOLERANCE:
        raise RuntimeError(
            "Objective exceeds exact optimum."
        )

    if abs(optimum) <= BENCHMARK_TOLERANCE:
        return (
            1.0
            if abs(objective) <= BENCHMARK_TOLERANCE
            else None
        )

    return float(
        objective / optimum
    )


def solve_exact_selection(problem):
    start = time.perf_counter()

    if problem.size > 20:
        raise ValueError(
            "Exact enumeration limited to 20 variables."
        )

    total_states = 2 ** problem.size

    integers = np.arange(
        total_states,
        dtype=np.uint64,
    )

    shifts = np.arange(
        problem.size,
        dtype=np.uint64,
    )

    states = (
        (integers[:, None] >> shifts)
        & 1
    ).astype(np.float64)

    durations = states @ problem.durations

    feasible = (
        durations
        <= problem.budget + BENCHMARK_TOLERANCE
    )

    risks = states @ problem.risks

    redundancy = 0.5 * np.einsum(
        "bi,ij,bj->b",
        states,
        problem.similarity,
        states,
        optimize=True,
    )

    objectives = (
        risks
        - problem.redundancy_penalty
        * redundancy
    )

    objectives[~feasible] = -np.inf

    best_index = int(
        np.argmax(objectives)
    )

    selection = states[
        best_index
    ].astype(np.int8)

    result = evaluate_selection(
        problem,
        selection,
    )

    result.update(
        {
            "solver_seconds": (
                time.perf_counter() - start
            ),
            "states_evaluated": total_states,
            "feasible_states": int(
                feasible.sum()
            ),
        }
    )

    return selection, result


def validate_solution(
    original_problem,
    discrete_problem,
    selection,
    optimum,
):
    original = evaluate_selection(
        original_problem,
        selection,
    )

    discrete = evaluate_selection(
        discrete_problem,
        selection,
    )

    if not original["feasible"]:
        raise RuntimeError(
            "Original budget violated."
        )

    if not discrete["feasible"]:
        raise RuntimeError(
            "Discretized budget violated."
        )

    if not np.isclose(
        original["objective_value"],
        discrete["objective_value"],
        atol=BENCHMARK_TOLERANCE,
        rtol=0,
    ):
        raise RuntimeError(
            "Objective mismatch."
        )

    ratio = calculate_ratio(
        original["objective_value"],
        optimum,
    )

    return {
        "objective": float(
            original["objective_value"]
        ),
        "approximation_ratio": ratio,
        "selected_count": original[
            "selected_count"
        ],
        "estimated_duration_seconds": original[
            "estimated_duration_seconds"
        ],
        "discrete_duration_seconds": discrete[
            "estimated_duration_seconds"
        ],
        "selected_tests": original[
            "selected_tests"
        ],
        "feasible": True,
        "optimal": bool(
            np.isclose(
                original["objective_value"],
                optimum,
                atol=BENCHMARK_TOLERANCE,
                rtol=0,
            )
        ),
    }


def benchmark_classical(
    original_problem,
    discrete_problem,
    optimum,
):
    records = []
    selections = []

    for algorithm in [
        "greedy",
        "ortools_cp_sat",
        "exact_enumeration",
    ]:
        for repetition in range(
            BENCHMARK_CLASSICAL_REPETITIONS
        ):
            start = time.perf_counter()

            if algorithm == "greedy":
                selection, result, trace = solve_greedy(
                    discrete_problem
                )

            elif algorithm == "ortools_cp_sat":
                selection, result = solve_ortools(
                    discrete_problem
                )

            else:
                selection, result = solve_exact_selection(
                    discrete_problem
                )

            runtime = (
                time.perf_counter() - start
            )

            verified = validate_solution(
                original_problem,
                discrete_problem,
                selection,
                optimum,
            )

            records.append(
                {
                    "algorithm": algorithm,
                    "repetition": repetition + 1,
                    "objective": verified["objective"],
                    "approximation_ratio": verified[
                        "approximation_ratio"
                    ],
                    "runtime_seconds": runtime,
                    "feasible": verified["feasible"],
                    "optimal": verified["optimal"],
                    "selected_count": verified[
                        "selected_count"
                    ],
                    "estimated_duration_seconds": verified[
                        "estimated_duration_seconds"
                    ],
                    "discrete_duration_seconds": verified[
                        "discrete_duration_seconds"
                    ],
                }
            )

            for test_id in verified["selected_tests"]:
                selections.append(
                    {
                        "algorithm": algorithm,
                        "repetition": repetition + 1,
                        "test_id": test_id,
                    }
                )

        print(
            f"{algorithm}: "
            f"{BENCHMARK_CLASSICAL_REPETITIONS} "
            "repetitions completed",
            flush=True,
        )

    return (
        pd.DataFrame(records),
        pd.DataFrame(selections),
    )


def load_sampling_results(
    comparison,
    optimum,
):
    records = []

    algorithms = [
        "random_raw",
        "random_hybrid",
        "qaoa_raw",
        "qaoa_hybrid",
    ]

    for algorithm in algorithms:
        for index, row in comparison.iterrows():
            ratio = row[
                f"{algorithm}_ratio"
            ]

            feasible = bool(
                pd.notna(ratio)
            )

            objective = (
                float(ratio * optimum)
                if feasible
                else None
            )

            if feasible:
                calculate_ratio(
                    objective,
                    optimum,
                )

            records.append(
                {
                    "algorithm": algorithm,
                    "repetition": index + 1,
                    "experiment": row["experiment"],
                    "seed": int(row["seed"]),
                    "shots": int(row["shots"]),
                    "objective": objective,
                    "approximation_ratio": (
                        float(ratio)
                        if feasible
                        else None
                    ),
                    "runtime_seconds": float(
                        row[f"{algorithm}_seconds"]
                    ),
                    "feasible": feasible,
                    "optimal": bool(
                        row[
                            f"{algorithm}_optimal_shots"
                        ] > 0
                    ),
                    "measurement_feasibility_rate": float(
                        row[
                            f"{algorithm}_feasibility"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def aggregate_results(dataframe):
    return dataframe.groupby(
        "algorithm",
        sort=False,
    ).agg(
        runs=("repetition", "count"),
        feasible_runs=("feasible", "sum"),
        optimal_runs=("optimal", "sum"),
        mean_objective=("objective", "mean"),
        best_objective=("objective", "max"),
        mean_approximation_ratio=(
            "approximation_ratio",
            "mean",
        ),
        mean_runtime_seconds=(
            "runtime_seconds",
            "mean",
        ),
        median_runtime_seconds=(
            "runtime_seconds",
            "median",
        ),
        runtime_std_seconds=(
            "runtime_seconds",
            "std",
        ),
        feasibility_rate=("feasible", "mean"),
        optimal_success_rate=("optimal", "mean"),
    ).reset_index()


def generate_plot(
    dataframe,
    column,
    title,
    ylabel,
    path,
    logarithmic=False,
):
    labels = {
        "greedy": "Greedy",
        "ortools_cp_sat": "CP-SAT",
        "exact_enumeration": "Exact",
        "random_raw": "Random",
        "random_hybrid": "Random + Repair",
        "qaoa_raw": "QAOA",
        "qaoa_hybrid": "QAOA + Repair",
    }

    names = [
        labels[name]
        for name in dataframe["algorithm"]
    ]

    values = dataframe[
        column
    ].to_numpy(dtype=float)

    plt.figure(
        figsize=(12, 6)
    )

    bars = plt.bar(
        names,
        values,
    )

    if logarithmic:
        plt.yscale("log")

    for bar, value in zip(
        bars,
        values,
    ):
        if not np.isfinite(value):
            continue

        plt.annotate(
            f"{value:.4g}",
            (
                bar.get_x()
                + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
        )

    plt.title(title)

    plt.ylabel(ylabel)

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.margins(y=0.15)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


def generate_plots(summary):
    generate_plot(
        summary,
        "mean_objective",
        "QuantCI - Optimization Objective",
        "Mean Objective",
        UNIFIED_OBJECTIVE_PLOT,
    )

    generate_plot(
        summary,
        "mean_approximation_ratio",
        "QuantCI - Approximation Ratio",
        "Mean Approximation Ratio",
        UNIFIED_RATIO_PLOT,
    )

    generate_plot(
        summary,
        "mean_runtime_seconds",
        "QuantCI - Solver Runtime",
        "Runtime in Seconds (Log Scale)",
        UNIFIED_RUNTIME_PLOT,
        logarithmic=True,
    )

    generate_plot(
        summary,
        "optimal_success_rate",
        "QuantCI - Optimal Solution Success",
        "Optimal Solution Success Rate",
        UNIFIED_SUCCESS_PLOT,
    )


def main():
    start = time.perf_counter()

    BENCHMARK_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = read_json(
        PRUNED_REPORT_FILE
    )

    exact = read_json(
        PRUNED_EXACT_FILE
    )

    candidates = pd.read_csv(
        PRUNED_CANDIDATES_FILE
    )

    original_problem = create_problem(
        candidates,
        budget=report["budget_seconds"],
        redundancy_penalty=report[
            "redundancy_penalty"
        ],
    )

    discrete_problem = create_discrete_problem(
        original_problem,
        report,
    )

    optimum = float(
        exact["objective_value"]
    )

    selection, reference = solve_exact_selection(
        discrete_problem
    )

    if not np.isclose(
        reference["objective_value"],
        optimum,
        atol=BENCHMARK_TOLERANCE,
        rtol=0,
    ):
        raise RuntimeError(
            "Exact selection and QUBO optima differ."
        )

    print("=" * 70)
    print("QuantCI - Unified Optimization Benchmark")
    print("=" * 70)

    print(
        "Candidate tests:",
        discrete_problem.size,
    )

    print(
        "Budget:",
        discrete_problem.budget,
    )

    print(
        "Exact objective:",
        optimum,
    )

    classical, selections = benchmark_classical(
        original_problem,
        discrete_problem,
        optimum,
    )

    comparison = pd.read_csv(
        RANDOM_QAOA_COMPARISON_FILE
    )

    sampling = load_sampling_results(
        comparison,
        optimum,
    )

    dataframe = pd.concat(
        [
            classical,
            sampling,
        ],
        ignore_index=True,
    )

    summary = aggregate_results(
        dataframe
    )

    summary.to_csv(
        UNIFIED_BENCHMARK_SUMMARY,
        index=False,
    )

    selections.to_csv(
        UNIFIED_BENCHMARK_SELECTIONS,
        index=False,
    )

    dataframe.to_csv(
        BENCHMARK_RESULTS_DIR
        / "unified_benchmark_runs.csv",
        index=False,
    )

    generate_plots(
        summary
    )

    algorithms = json.loads(
        summary.to_json(
            orient="records"
        )
    )

    final_report = {
        "partition": report["partition"],
        "run_id": report["run_id"],
        "candidate_tests": discrete_problem.size,
        "qubo_variables": report["reduced_qubits"],
        "budget_seconds": discrete_problem.budget,
        "exact_objective": optimum,
        "exact_selection_states": reference[
            "states_evaluated"
        ],
        "exact_feasible_selections": reference[
            "feasible_states"
        ],
        "exact_reference_verified": True,
        "algorithms_compared": len(summary),
        "total_benchmark_records": len(dataframe),
        "same_candidate_set": True,
        "same_discrete_constraints": True,
        "same_objective": True,
        "classical_repetitions": (
            BENCHMARK_CLASSICAL_REPETITIONS
        ),
        "sampling_experiments_per_variant": len(
            comparison
        ),
        "quantum_results_reused": True,
        "quantum_advantage_established": False,
        "algorithms": algorithms,
        "total_benchmark_seconds": (
            time.perf_counter() - start
        ),
    }

    UNIFIED_BENCHMARK_REPORT.write_text(
        json.dumps(
            final_report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        json.dumps(
            final_report,
            indent=4,
        )
    )

    print("=" * 70)
    print("UNIFIED BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()