import json
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ortools.sat.python import cp_model

from src.config import (
    RANDOM_SEED,
    MAX_CLASSICAL_CANDIDATES,
    BENCHMARK_BUDGETS_SECONDS,
    OPTIMIZATION_TOLERANCE,
    OPTIMIZATION_RESULTS_DIR,
    ORTOOLS_REPORT_FILE,
    ORTOOLS_SUMMARY_FILE,
    ORTOOLS_SELECTION_FILE,
    ORTOOLS_COMPARISON_FILE,
    ORTOOLS_OBJECTIVE_PLOT,
    ORTOOLS_RUNTIME_PLOT,
    ORTOOLS_GAP_PLOT,
    ORTOOLS_TIME_LIMIT_SECONDS,
    ORTOOLS_DURATION_SCALE,
    ORTOOLS_OBJECTIVE_SCALE,
    ORTOOLS_WORKERS,
)

from src.optimization.candidates import (
    load_predictions,
    prepare_candidates,
    get_run_ids,
)

from src.optimization.objective import (
    create_problem,
    evaluate_selection,
    objective_value,
    is_feasible,
)

from src.optimization.greedy import solve_greedy


def scale_durations(problem):
    durations = np.ceil(
        problem.durations * ORTOOLS_DURATION_SCALE
    ).astype(np.int64)

    budget = int(
        np.floor(
            problem.budget * ORTOOLS_DURATION_SCALE
        )
    )

    return durations, budget


def scale_objective(problem):
    risks = np.rint(
        problem.risks * ORTOOLS_OBJECTIVE_SCALE
    ).astype(np.int64)

    penalties = np.rint(
        problem.similarity
        * problem.redundancy_penalty
        * ORTOOLS_OBJECTIVE_SCALE
    ).astype(np.int64)

    return risks, penalties


def solve_ortools(
    problem,
    time_limit=ORTOOLS_TIME_LIMIT_SECONDS,
    workers=ORTOOLS_WORKERS,
):
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Invalid solver time limit.")

    if not isinstance(workers, int) or workers <= 0:
        raise ValueError("Invalid worker count.")

    start_time = time.perf_counter()

    durations, budget = scale_durations(problem)

    risks, penalties = scale_objective(problem)

    model = cp_model.CpModel()

    variables = [
        model.new_bool_var(f"x_{i}")
        for i in range(problem.size)
    ]

    model.add(
        sum(
            int(durations[i]) * variables[i]
            for i in range(problem.size)
        ) <= budget
    )

    objective_terms = [
        int(risks[i]) * variables[i]
        for i in range(problem.size)
    ]

    auxiliary_variables = {}

    for i in range(problem.size):
        for j in range(i + 1, problem.size):
            penalty = int(penalties[i, j])

            if penalty <= 0:
                continue

            pair = model.new_bool_var(
                f"y_{i}_{j}"
            )

            model.add(pair <= variables[i])

            model.add(pair <= variables[j])

            model.add(
                pair >= variables[i] + variables[j] - 1
            )

            auxiliary_variables[(i, j)] = pair

            objective_terms.append(
                -penalty * pair
            )

    model.maximize(sum(objective_terms))

    greedy_selection, _, _ = solve_greedy(problem)

    hint = np.asarray(
        greedy_selection,
        dtype=np.int8,
    )

    if int(durations @ hint) > budget:
        hint = np.zeros(
            problem.size,
            dtype=np.int8,
        )

    for i, variable in enumerate(variables):
        model.add_hint(
            variable,
            int(hint[i]),
        )

    for (i, j), variable in auxiliary_variables.items():
        model.add_hint(
            variable,
            int(hint[i] * hint[j]),
        )

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = float(
        time_limit
    )

    solver.parameters.num_search_workers = workers

    solver.parameters.random_seed = RANDOM_SEED

    status = solver.solve(model)

    status_name = solver.status_name(status)

    scaled_objective = None
    scaled_best_bound = None
    optimality_gap = None

    if status in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        selection = np.array(
            [
                solver.value(variable)
                for variable in variables
            ],
            dtype=np.int8,
        )

        scaled_objective = float(
            solver.objective_value
        )

        scaled_best_bound = float(
            solver.best_objective_bound
        )

        optimality_gap = max(
            0.0,
            scaled_best_bound - scaled_objective,
        ) / max(
            abs(scaled_objective),
            1.0,
        )

        solution_source = "cp_sat"

    elif status == cp_model.UNKNOWN:
        selection = hint.copy()

        solution_source = "greedy_fallback"

    else:
        raise RuntimeError(
            f"CP-SAT failed: {status_name}"
        )

    if not is_feasible(problem, selection):
        raise RuntimeError(
            "Original budget constraint violated."
        )

    if int(durations @ selection) > budget:
        raise RuntimeError(
            "Scaled budget constraint violated."
        )

    result = evaluate_selection(
        problem,
        selection,
    )

    result.update(
        {
            "algorithm": "ortools_cp_sat",
            "solver_status": status_name,
            "solution_source": solution_source,
            "optimal_scaled_model": (
                status == cp_model.OPTIMAL
            ),
            "solver_seconds": (
                time.perf_counter() - start_time
            ),
            "cp_sat_wall_seconds": float(
                solver.wall_time
            ),
            "scaled_objective": scaled_objective,
            "scaled_best_bound": scaled_best_bound,
            "scaled_optimality_gap": optimality_gap,
            "binary_test_variables": problem.size,
            "auxiliary_variables": len(
                auxiliary_variables
            ),
            "branches": int(
                solver.num_branches
            ),
            "conflicts": int(
                solver.num_conflicts
            ),
            "budget_utilization": (
                result["estimated_duration_seconds"]
                / problem.budget
                if problem.budget > 0
                else 0.0
            ),
        }
    )

    return selection, result


def export_selection(
    candidates,
    selection,
    budget,
):
    selected = candidates.loc[
        np.asarray(selection) == 1
    ].copy()

    selected["selected"] = 1

    selected["budget_seconds"] = budget

    selected["algorithm"] = "ortools_cp_sat"

    return selected


def compare_solvers(problem):
    greedy_selection, greedy_result, _ = solve_greedy(
        problem
    )

    ortools_selection, ortools_result = solve_ortools(
        problem
    )

    greedy_objective = greedy_result[
        "objective_value"
    ]

    ortools_objective = ortools_result[
        "objective_value"
    ]

    improvement = (
        ortools_objective - greedy_objective
    )

    comparison = {
        "budget_seconds": problem.budget,
        "candidate_count": problem.size,
        "greedy_objective": greedy_objective,
        "ortools_objective": ortools_objective,
        "objective_improvement": improvement,
        "objective_improvement_percent": (
            improvement
            / abs(greedy_objective)
            * 100
            if abs(greedy_objective)
            > OPTIMIZATION_TOLERANCE
            else None
        ),
        "greedy_selected": greedy_result[
            "selected_count"
        ],
        "ortools_selected": ortools_result[
            "selected_count"
        ],
        "greedy_duration": greedy_result[
            "estimated_duration_seconds"
        ],
        "ortools_duration": ortools_result[
            "estimated_duration_seconds"
        ],
        "greedy_solver_seconds": greedy_result[
            "solver_seconds"
        ],
        "ortools_solver_seconds": ortools_result[
            "solver_seconds"
        ],
        "solver_status": ortools_result[
            "solver_status"
        ],
        "solution_source": ortools_result[
            "solution_source"
        ],
        "optimality_gap": ortools_result[
            "scaled_optimality_gap"
        ],
        "greedy_feasible": greedy_result[
            "feasible"
        ],
        "ortools_feasible": ortools_result[
            "feasible"
        ],
    }

    return (
        ortools_selection,
        ortools_result,
        comparison,
    )


def generate_plots(comparison):
    ORTOOLS_OBJECTIVE_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grouped = comparison.groupby(
        "budget_seconds"
    )

    objective = grouped[
        [
            "greedy_objective",
            "ortools_objective",
        ]
    ].mean()

    objective.columns = [
        "Greedy",
        "OR-Tools CP-SAT",
    ]

    objective.plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.title(
        "Classical Optimization Objective Comparison"
    )

    plt.xlabel("Budget in seconds")

    plt.ylabel("Mean objective value")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        ORTOOLS_OBJECTIVE_PLOT,
        dpi=200,
    )

    plt.close()

    runtime = grouped[
        [
            "greedy_solver_seconds",
            "ortools_solver_seconds",
        ]
    ].mean()

    runtime.columns = [
        "Greedy",
        "OR-Tools CP-SAT",
    ]

    runtime.plot(
        kind="bar",
        figsize=(10, 6),
        logy=True,
    )

    plt.title(
        "Classical Solver Runtime Comparison"
    )

    plt.xlabel("Budget in seconds")

    plt.ylabel("Mean solver time in seconds")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        ORTOOLS_RUNTIME_PLOT,
        dpi=200,
    )

    plt.close()

    gaps = comparison[
        "optimality_gap"
    ].dropna() * 100

    plt.figure(figsize=(10, 6))

    if not gaps.empty:
        plt.hist(
            gaps,
            bins=20,
            edgecolor="black",
        )

    plt.title("CP-SAT Scaled Optimality Gap")

    plt.xlabel("Optimality gap percentage")

    plt.ylabel("Number of problems")

    plt.tight_layout()

    plt.savefig(
        ORTOOLS_GAP_PLOT,
        dpi=200,
    )

    plt.close()


def main():
    print("=" * 70)
    print("QuantCI - OR-Tools CP-SAT Optimization")
    print("=" * 70)

    OPTIMIZATION_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = load_predictions(
        "validation"
    )

    run_ids = get_run_ids(
        predictions
    )

    summaries = []
    selections = []
    comparisons = []

    total_start = time.perf_counter()

    for index, run_id in enumerate(
        run_ids,
        start=1,
    ):
        candidates = prepare_candidates(
            predictions,
            run_id,
            MAX_CLASSICAL_CANDIDATES,
        )

        print(
            f"\nRun {index}/{len(run_ids)}: {run_id}"
        )

        for budget in BENCHMARK_BUDGETS_SECONDS:
            problem = create_problem(
                candidates,
                budget,
            )

            selection, result, comparison = compare_solvers(
                problem
            )

            result["run_id"] = run_id

            comparison["run_id"] = run_id

            summaries.append(
                {
                    key: value
                    for key, value in result.items()
                    if key != "selected_tests"
                }
            )

            selections.append(
                export_selection(
                    candidates,
                    selection,
                    budget,
                )
            )

            comparisons.append(
                comparison
            )

            print(
                f"Budget: {budget:4}s | "
                f"Greedy: {comparison['greedy_objective']:.4f} | "
                f"CP-SAT: {comparison['ortools_objective']:.4f} | "
                f"Status: {result['solver_status']} | "
                f"Time: {result['solver_seconds']:.3f}s"
            )

    summary = pd.DataFrame(
        summaries
    )

    comparison = pd.DataFrame(
        comparisons
    )

    selected_tests = pd.concat(
        selections,
        ignore_index=True,
    )

    summary.to_csv(
        ORTOOLS_SUMMARY_FILE,
        index=False,
    )

    comparison.to_csv(
        ORTOOLS_COMPARISON_FILE,
        index=False,
    )

    selected_tests.to_csv(
        ORTOOLS_SELECTION_FILE,
        index=False,
    )

    budget_summary = comparison.groupby(
        "budget_seconds"
    ).agg(
        problems=("run_id", "size"),
        mean_greedy_objective=(
            "greedy_objective",
            "mean",
        ),
        mean_ortools_objective=(
            "ortools_objective",
            "mean",
        ),
        mean_objective_improvement=(
            "objective_improvement",
            "mean",
        ),
        mean_greedy_seconds=(
            "greedy_solver_seconds",
            "mean",
        ),
        mean_ortools_seconds=(
            "ortools_solver_seconds",
            "mean",
        ),
        mean_optimality_gap=(
            "optimality_gap",
            "mean",
        ),
    ).reset_index()

    improvements = comparison[
        "objective_improvement"
    ]

    report = {
        "algorithm": "ortools_cp_sat",
        "partition": "validation",
        "total_runs": len(run_ids),
        "total_problems": len(summary),
        "feasible_solutions": int(
            summary["feasible"].sum()
        ),
        "infeasible_solutions": int(
            (~summary["feasible"]).sum()
        ),
        "solver_status_counts": {
            str(key): int(value)
            for key, value in summary[
                "solver_status"
            ].value_counts().items()
        },
        "fallback_solutions": int(
            (
                summary["solution_source"]
                == "greedy_fallback"
            ).sum()
        ),
        "better_than_greedy": int(
            (
                improvements
                > OPTIMIZATION_TOLERANCE
            ).sum()
        ),
        "equal_to_greedy": int(
            (
                improvements.abs()
                <= OPTIMIZATION_TOLERANCE
            ).sum()
        ),
        "worse_than_greedy": int(
            (
                improvements
                < -OPTIMIZATION_TOLERANCE
            ).sum()
        ),
        "mean_objective_improvement": float(
            improvements.mean()
        ),
        "mean_solver_seconds": float(
            summary["solver_seconds"].mean()
        ),
        "total_experiment_seconds": (
            time.perf_counter() - total_start
        ),
        "time_limit_seconds": ORTOOLS_TIME_LIMIT_SECONDS,
        "workers": ORTOOLS_WORKERS,
        "duration_scale": ORTOOLS_DURATION_SCALE,
        "objective_scale": ORTOOLS_OBJECTIVE_SCALE,
        "maximum_candidates": MAX_CLASSICAL_CANDIDATES,
        "actual_failure_evaluation": "deferred",
    }

    budget_records = json.loads(
        budget_summary.to_json(
            orient="records"
        )
    )

    report["budget_summary"] = budget_records

    ORTOOLS_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    generate_plots(
        comparison
    )

    print("\nBudget Comparison:")

    print(
        budget_summary.to_string(
            index=False
        )
    )

    print("\nFinal Report:")

    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    print("=" * 70)
    print("OR-TOOLS OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()