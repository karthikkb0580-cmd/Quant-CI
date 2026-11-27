import json
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import (
    DEFAULT_BUDGET_SECONDS,
    MAX_CLASSICAL_CANDIDATES,
    BENCHMARK_BUDGETS_SECONDS,
    OPTIMIZATION_TOLERANCE,
    GREEDY_REPORT_FILE,
    GREEDY_SELECTION_FILE,
    GREEDY_SUMMARY_FILE,
    GREEDY_TRACE_FILE,
    GREEDY_BUDGET_PLOT,
    GREEDY_OBJECTIVE_PLOT,
    GREEDY_RUNTIME_PLOT,
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


def solve_greedy(problem):
    start = time.perf_counter()

    selection = np.zeros(
        problem.size,
        dtype=np.int8,
    )

    remaining_budget = problem.budget

    marginal_gains = problem.risks.copy()

    trace = []

    iteration = 0

    while True:
        feasible = (
            (selection == 0)
            & (
                problem.durations
                <= remaining_budget
                + OPTIMIZATION_TOLERANCE
            )
            & (
                marginal_gains
                > OPTIMIZATION_TOLERANCE
            )
        )

        indices = np.flatnonzero(feasible)

        if len(indices) == 0:
            break

        priorities = (
            marginal_gains[indices]
            / problem.durations[indices]
        )

        best_index = indices[
            np.argmax(priorities)
        ]

        gain = float(
            marginal_gains[best_index]
        )

        priority = float(
            gain / problem.durations[best_index]
        )

        selection[best_index] = 1

        remaining_budget -= (
            problem.durations[best_index]
        )

        marginal_gains -= (
            problem.redundancy_penalty
            * problem.similarity[:, best_index]
        )

        iteration += 1

        trace.append(
            {
                "iteration": iteration,
                "candidate_index": int(best_index),
                "test_id": problem.test_ids[best_index],
                "failure_score": float(
                    problem.risks[best_index]
                ),
                "estimated_duration_seconds": float(
                    problem.durations[best_index]
                ),
                "marginal_gain": gain,
                "priority": priority,
                "remaining_budget_seconds": float(
                    remaining_budget
                ),
                "objective_value": objective_value(
                    problem,
                    selection,
                ),
            }
        )

    greedy_objective = objective_value(
        problem,
        selection,
    )

    feasible_singletons = np.flatnonzero(
        problem.durations
        <= problem.budget
        + OPTIMIZATION_TOLERANCE
    )

    solution_type = "greedy"

    if len(feasible_singletons):
        best_singleton = feasible_singletons[
            np.argmax(
                problem.risks[feasible_singletons]
            )
        ]

        singleton_selection = np.zeros(
            problem.size,
            dtype=np.int8,
        )

        singleton_selection[best_singleton] = 1

        singleton_objective = objective_value(
            problem,
            singleton_selection,
        )

        if (
            singleton_objective
            > greedy_objective
            + OPTIMIZATION_TOLERANCE
        ):
            selection = singleton_selection

            solution_type = "best_singleton"

            trace = [
                {
                    "iteration": 1,
                    "candidate_index": int(best_singleton),
                    "test_id": problem.test_ids[best_singleton],
                    "failure_score": float(
                        problem.risks[best_singleton]
                    ),
                    "estimated_duration_seconds": float(
                        problem.durations[best_singleton]
                    ),
                    "marginal_gain": float(
                        problem.risks[best_singleton]
                    ),
                    "priority": float(
                        problem.risks[best_singleton]
                        / problem.durations[best_singleton]
                    ),
                    "remaining_budget_seconds": float(
                        problem.budget
                        - problem.durations[best_singleton]
                    ),
                    "objective_value": singleton_objective,
                }
            ]

    if not is_feasible(
        problem,
        selection,
    ):
        raise RuntimeError(
            "Greedy generated an infeasible solution."
        )

    elapsed = time.perf_counter() - start

    result = evaluate_selection(
        problem,
        selection,
    )

    result["algorithm"] = "greedy"

    result["solution_type"] = solution_type

    result["solver_seconds"] = float(elapsed)

    result["iterations"] = len(trace)

    result["budget_utilization"] = (
        result["estimated_duration_seconds"]
        / problem.budget
        if problem.budget > 0
        else 0.0
    )

    return selection, result, trace


def export_selection(
    candidates,
    selection,
    trace,
):
    selected = candidates.loc[
        selection.astype(bool)
    ].copy()

    selected["selected"] = 1

    order = {
        item["test_id"]: item["iteration"]
        for item in trace
    }

    selected["selection_order"] = (
        selected["test_id"].map(order)
    )

    return selected.sort_values(
        "selection_order"
    ).reset_index(drop=True)


def evaluate_run(
    candidates,
    run_id,
    budget,
):
    problem = create_problem(
        candidates,
        budget=budget,
    )

    selection, result, trace = solve_greedy(
        problem
    )

    result["run_id"] = run_id

    result["full_candidate_runtime"] = float(
        problem.durations.sum()
    )

    result["estimated_runtime_reduction"] = (
        1
        - result["estimated_duration_seconds"]
        / problem.durations.sum()
    )

    selected = export_selection(
        candidates,
        selection,
        trace,
    )

    selected["budget_seconds"] = budget

    trace_dataframe = pd.DataFrame(trace)

    trace_dataframe["run_id"] = run_id

    trace_dataframe["budget_seconds"] = budget

    return result, selected, trace_dataframe


def generate_plots(
    summary,
    example_trace,
):
    grouped = summary.groupby(
        "budget_seconds"
    ).agg(
        mean_objective=("objective_value", "mean"),
        mean_selected=("selected_count", "mean"),
        mean_duration=(
            "estimated_duration_seconds",
            "mean",
        ),
        mean_solver_seconds=(
            "solver_seconds",
            "mean",
        ),
    ).reset_index()

    plt.figure(figsize=(10, 6))

    plt.plot(
        grouped["budget_seconds"],
        grouped["mean_objective"],
        marker="o",
    )

    plt.xlabel("Budget in Seconds")
    plt.ylabel("Mean Objective Value")
    plt.title("Greedy Optimization by Time Budget")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GREEDY_BUDGET_PLOT,
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    if not example_trace.empty:
        plt.plot(
            example_trace["iteration"],
            example_trace["objective_value"],
            marker="o",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title("Greedy Objective Progress")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GREEDY_OBJECTIVE_PLOT,
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(
        grouped["budget_seconds"],
        grouped["mean_duration"],
        marker="o",
        label="Selected Runtime",
    )

    plt.plot(
        grouped["budget_seconds"],
        grouped["budget_seconds"],
        linestyle="--",
        label="Maximum Budget",
    )

    plt.xlabel("Budget in Seconds")
    plt.ylabel("Estimated Runtime in Seconds")
    plt.title("Greedy Runtime and Budget")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GREEDY_RUNTIME_PLOT,
        dpi=200,
    )

    plt.close()


def main():
    print("=" * 60)
    print("QuantCI - Greedy Optimization")
    print("=" * 60)

    GREEDY_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GREEDY_BUDGET_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_predictions(
        "validation"
    )

    run_ids = get_run_ids(
        dataframe
    )

    results = []

    selections = []

    traces = []

    example_trace = pd.DataFrame()

    for run_number, run_id in enumerate(
        run_ids,
        start=1,
    ):
        candidates = prepare_candidates(
            dataframe,
            run_id,
            max_candidates=MAX_CLASSICAL_CANDIDATES,
        )

        print(
            f"\nRun {run_number}/{len(run_ids)}: "
            f"{run_id}"
        )

        for budget in BENCHMARK_BUDGETS_SECONDS:
            result, selected, trace = evaluate_run(
                candidates,
                run_id,
                budget,
            )

            results.append(result)

            selections.append(selected)

            traces.append(trace)

            if (
                run_number == 1
                and budget == DEFAULT_BUDGET_SECONDS
            ):
                example_trace = trace.copy()

            print(
                f"Budget: {budget:>4}s | "
                f"Selected: {result['selected_count']:>3} | "
                f"Runtime: "
                f"{result['estimated_duration_seconds']:.2f}s | "
                f"Objective: "
                f"{result['objective_value']:.4f} | "
                f"Feasible: {result['feasible']}"
            )

    summary = pd.DataFrame(results)

    summary = summary.drop(
        columns=["selected_tests"]
    )

    summary.to_csv(
        GREEDY_SUMMARY_FILE,
        index=False,
    )

    pd.concat(
        selections,
        ignore_index=True,
    ).to_csv(
        GREEDY_SELECTION_FILE,
        index=False,
    )

    pd.concat(
        traces,
        ignore_index=True,
    ).to_csv(
        GREEDY_TRACE_FILE,
        index=False,
    )

    budget_summary = []

    for budget, group in summary.groupby(
        "budget_seconds"
    ):
        budget_summary.append(
            {
                "budget_seconds": float(budget),
                "runs": len(group),
                "feasible_runs": int(
                    group["feasible"].sum()
                ),
                "mean_selected_tests": float(
                    group["selected_count"].mean()
                ),
                "mean_objective": float(
                    group["objective_value"].mean()
                ),
                "mean_estimated_duration": float(
                    group[
                        "estimated_duration_seconds"
                    ].mean()
                ),
                "mean_budget_utilization": float(
                    group[
                        "budget_utilization"
                    ].mean()
                ),
                "mean_solver_seconds": float(
                    group["solver_seconds"].mean()
                ),
                "mean_candidate_runtime_reduction": float(
                    group[
                        "estimated_runtime_reduction"
                    ].mean()
                ),
            }
        )

    report = {
        "algorithm": "greedy",
        "partition": "validation",
        "total_runs": len(run_ids),
        "budgets_seconds": BENCHMARK_BUDGETS_SECONDS,
        "total_optimization_problems": len(summary),
        "feasible_solutions": int(
            summary["feasible"].sum()
        ),
        "infeasible_solutions": int(
            (~summary["feasible"]).sum()
        ),
        "maximum_candidates": MAX_CLASSICAL_CANDIDATES,
        "budget_summary": budget_summary,
        "actual_failure_evaluation": "deferred",
    }

    GREEDY_REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    generate_plots(
        summary,
        example_trace,
    )

    print("\nBudget Summary:")

    print(
        pd.DataFrame(
            budget_summary
        ).to_string(index=False)
    )

    print(
        f"\nTotal problems: "
        f"{len(summary)}"
    )

    print(
        f"Feasible solutions: "
        f"{report['feasible_solutions']}"
    )

    print(
        f"Infeasible solutions: "
        f"{report['infeasible_solutions']}"
    )

    print(
        f"\nReport: {GREEDY_REPORT_FILE}"
    )

    print("=" * 60)
    print("GREEDY OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()