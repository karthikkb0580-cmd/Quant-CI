import json
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import (
    FINAL_TEST_PREDICTIONS,
    HELDOUT_RESULTS_DIR,
    HELDOUT_REPORT_FILE,
    HELDOUT_SUMMARY_FILE,
    HELDOUT_RUNS_FILE,
    HELDOUT_SELECTIONS_FILE,
    HELDOUT_RECALL_PLOT,
    HELDOUT_REDUCTION_PLOT,
    HELDOUT_RUNTIME_PLOT,
    HELDOUT_OVERRUN_PLOT,
    HELDOUT_BUDGETS,
    HELDOUT_RANDOM_SEED,
    HELDOUT_EXPECTED_ROWS,
    HELDOUT_EXPECTED_FAILURES,
    HELDOUT_EXPECTED_RUNS,
    MAX_CLASSICAL_CANDIDATES,
    MIN_ESTIMATED_DURATION,
    OPTIMIZATION_TOLERANCE,
)

from src.optimization.candidates import (
    prepare_candidates,
)

from src.optimization.objective import (
    create_problem,
)

from src.optimization.greedy import (
    solve_greedy,
)

from src.optimization.ortools_solver import (
    solve_ortools,
)


REQUIRED_COLUMNS = [
    "run_id",
    "test_id",
    "failed",
    "duration_seconds",
    "estimated_duration_seconds",
    "failure_score",
    "smoothed_failure_rate",
]


ALGORITHMS = [
    "random",
    "shortest_first",
    "historical_risk",
    "ml_risk",
    "greedy",
    "ortools_cp_sat",
]


def safe_divide(numerator, denominator):
    if denominator == 0:
        return None

    return float(numerator / denominator)


def validate_data(dataframe):
    missing = set(REQUIRED_COLUMNS) - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if dataframe.empty:
        raise ValueError("Empty dataset.")

    if dataframe[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Missing values detected.")

    if dataframe.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError("Duplicate test executions.")

    if not dataframe["failed"].isin([0, 1]).all():
        raise ValueError("Invalid failure labels.")

    numerical = [
        "duration_seconds",
        "estimated_duration_seconds",
        "failure_score",
        "smoothed_failure_rate",
    ]

    if not np.isfinite(
        dataframe[numerical].to_numpy(dtype=float)
    ).all():
        raise ValueError("Nonfinite numerical values.")

    if (dataframe["duration_seconds"] < 0).any():
        raise ValueError("Negative actual duration.")

    if (dataframe["estimated_duration_seconds"] <= 0).any():
        raise ValueError("Invalid estimated duration.")

    for column in [
        "failure_score",
        "smoothed_failure_rate",
    ]:
        if not dataframe[column].between(0, 1).all():
            raise ValueError(f"Invalid scores: {column}")


def load_data():
    dataframe = pd.read_csv(
        FINAL_TEST_PREDICTIONS,
        usecols=REQUIRED_COLUMNS,
    )

    validate_data(dataframe)

    dataframe["estimated_duration_seconds"] = (
        dataframe["estimated_duration_seconds"]
        .clip(lower=MIN_ESTIMATED_DURATION)
    )

    if len(dataframe) != HELDOUT_EXPECTED_ROWS:
        raise RuntimeError("Unexpected test row count.")

    if int(dataframe["failed"].sum()) != HELDOUT_EXPECTED_FAILURES:
        raise RuntimeError("Unexpected failure count.")

    if dataframe["run_id"].nunique() != HELDOUT_EXPECTED_RUNS:
        raise RuntimeError("Unexpected run count.")

    return dataframe


def select_by_order(dataframe, budget):
    remaining = float(budget)

    selected = []

    for row in dataframe.itertuples(index=False):
        duration = float(
            row.estimated_duration_seconds
        )

        if duration <= remaining + OPTIMIZATION_TOLERANCE:
            selected.append(row.test_id)
            remaining -= duration

    return selected


def select_baseline(
    candidates,
    algorithm,
    budget,
    seed,
):
    ordered = candidates.copy()

    if algorithm == "random":
        ordered = ordered.sample(
            frac=1,
            random_state=seed,
        )

    elif algorithm == "shortest_first":
        ordered = ordered.sort_values(
            ["estimated_duration_seconds", "test_id"],
            ascending=[True, True],
        )

    elif algorithm == "historical_risk":
        ordered = ordered.sort_values(
            ["smoothed_failure_rate", "test_id"],
            ascending=[False, True],
        )

    elif algorithm == "ml_risk":
        ordered = ordered.sort_values(
            ["failure_score", "test_id"],
            ascending=[False, True],
        )

    else:
        raise ValueError(
            f"Unknown baseline: {algorithm}"
        )

    return select_by_order(
        ordered,
        budget,
    )


def evaluate_run(
    dataframe,
    candidates,
    selected_ids,
    budget,
):
    selected_ids = list(selected_ids)

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate selected tests.")

    if not set(selected_ids).issubset(
        set(candidates["test_id"])
    ):
        raise ValueError("Selection outside candidate set.")

    selected = dataframe[
        dataframe["test_id"].isin(selected_ids)
    ]

    total_tests = len(dataframe)

    total_failures = int(
        dataframe["failed"].sum()
    )

    detected_failures = int(
        selected["failed"].sum()
    )

    candidate_failures = int(
        candidates["failed"].sum()
    )

    estimated_duration = float(
        selected["estimated_duration_seconds"].sum()
    )

    actual_duration = float(
        selected["duration_seconds"].sum()
    )

    full_duration = float(
        dataframe["duration_seconds"].sum()
    )

    if estimated_duration > budget + OPTIMIZATION_TOLERANCE:
        raise RuntimeError("Estimated budget exceeded.")

    return {
        "total_tests": total_tests,
        "candidate_tests": len(candidates),
        "selected_tests": len(selected),
        "total_failures": total_failures,
        "candidate_failures": candidate_failures,
        "detected_failures": detected_failures,
        "missed_failures": total_failures - detected_failures,
        "failure_recall": safe_divide(
            detected_failures,
            total_failures,
        ),
        "failure_precision": safe_divide(
            detected_failures,
            len(selected),
        ),
        "candidate_recall": safe_divide(
            candidate_failures,
            total_failures,
        ),
        "test_reduction": 1 - len(selected) / total_tests,
        "estimated_duration_seconds": estimated_duration,
        "actual_duration_seconds": actual_duration,
        "full_duration_seconds": full_duration,
        "actual_runtime_reduction": (
            1 - actual_duration / full_duration
            if full_duration > 0
            else None
        ),
        "estimated_budget_feasible": True,
        "actual_budget_overrun": bool(
            actual_duration > budget + OPTIMIZATION_TOLERANCE
        ),
        "actual_overrun_seconds": max(
            0.0,
            actual_duration - budget,
        ),
        "failure_run": total_failures > 0,
        "failure_run_detected": detected_failures > 0,
    }


def summarize_results(dataframe):
    records = []

    for (algorithm, budget), group in dataframe.groupby(
        ["algorithm", "budget_seconds"],
        sort=False,
    ):
        total_tests = int(group["total_tests"].sum())

        selected_tests = int(
            group["selected_tests"].sum()
        )

        total_failures = int(
            group["total_failures"].sum()
        )

        detected_failures = int(
            group["detected_failures"].sum()
        )

        candidate_failures = int(
            group["candidate_failures"].sum()
        )

        failure_runs = int(
            group["failure_run"].sum()
        )

        detected_runs = int(
            group["failure_run_detected"].sum()
        )

        full_duration = float(
            group["full_duration_seconds"].sum()
        )

        actual_duration = float(
            group["actual_duration_seconds"].sum()
        )

        records.append({
            "algorithm": algorithm,
            "budget_seconds": float(budget),
            "runs": len(group),
            "total_tests": total_tests,
            "selected_tests": selected_tests,
            "total_failures": total_failures,
            "detected_failures": detected_failures,
            "missed_failures": total_failures - detected_failures,
            "failure_recall": safe_divide(
                detected_failures,
                total_failures,
            ),
            "failure_precision": safe_divide(
                detected_failures,
                selected_tests,
            ),
            "candidate_recall": safe_divide(
                candidate_failures,
                total_failures,
            ),
            "test_reduction": 1 - selected_tests / total_tests,
            "actual_runtime_reduction": (
                1 - actual_duration / full_duration
                if full_duration > 0
                else None
            ),
            "failure_runs": failure_runs,
            "detected_failure_runs": detected_runs,
            "failure_run_detection_rate": safe_divide(
                detected_runs,
                failure_runs,
            ),
            "estimated_feasibility_rate": float(
                group["estimated_budget_feasible"].mean()
            ),
            "actual_overrun_rate": float(
                group["actual_budget_overrun"].mean()
            ),
            "actual_overrun_runs": int(
                group["actual_budget_overrun"].sum()
            ),
            "mean_estimated_duration_seconds": float(
                group["estimated_duration_seconds"].mean()
            ),
            "mean_actual_duration_seconds": float(
                group["actual_duration_seconds"].mean()
            ),
            "mean_solver_seconds": float(
                group["solver_seconds"].mean()
            ),
        })

    return pd.DataFrame(records)


def generate_plot(
    summary,
    metric,
    title,
    ylabel,
    path,
):
    plt.figure(figsize=(11, 6))

    for algorithm, group in summary.groupby("algorithm"):
        group = group.sort_values("budget_seconds")

        plt.plot(
            group["budget_seconds"],
            group[metric] * 100,
            marker="o",
            label=algorithm,
        )

    plt.title(title)
    plt.xlabel("Estimated Execution Budget (Seconds)")
    plt.ylabel(ylabel)
    plt.xticks(HELDOUT_BUDGETS)
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def generate_plots(summary):
    generate_plot(
        summary,
        "failure_recall",
        "QuantCI - Held-Out Failure Detection",
        "Failure Recall (%)",
        HELDOUT_RECALL_PLOT,
    )

    generate_plot(
        summary,
        "test_reduction",
        "QuantCI - Test Suite Reduction",
        "Test Reduction (%)",
        HELDOUT_REDUCTION_PLOT,
    )

    generate_plot(
        summary,
        "actual_runtime_reduction",
        "QuantCI - Observed Execution Cost Reduction",
        "Cumulative Runtime Reduction (%)",
        HELDOUT_RUNTIME_PLOT,
    )

    generate_plot(
        summary,
        "actual_overrun_rate",
        "QuantCI - Actual Budget Overruns",
        "Budget Overrun Rate (%)",
        HELDOUT_OVERRUN_PLOT,
    )


def main():
    start = time.perf_counter()

    HELDOUT_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_data()

    records = []
    selections = []

    grouped = list(
        dataframe.groupby("run_id", sort=False)
    )

    print("=" * 70)
    print("QuantCI - Held-Out Optimization Evaluation")
    print("=" * 70)

    print("Test executions:", len(dataframe))
    print("Failures:", int(dataframe["failed"].sum()))
    print("CI runs:", len(grouped))

    for run_index, (run_id, run) in enumerate(grouped):
        run = run.reset_index(drop=True)

        shortlist = prepare_candidates(
            run,
            run_id,
            max_candidates=MAX_CLASSICAL_CANDIDATES,
        )

        candidates = shortlist.merge(
            run[
                [
                    "test_id",
                    "failed",
                    "duration_seconds",
                    "smoothed_failure_rate",
                ]
            ],
            on="test_id",
            how="left",
            validate="one_to_one",
        )

        for budget in HELDOUT_BUDGETS:
            problem = create_problem(
                candidates,
                budget=budget,
            )

            for algorithm in ALGORITHMS:
                solver_start = time.perf_counter()

                status = None

                if algorithm == "greedy":
                    selection, result, trace = solve_greedy(
                        problem
                    )

                    selected_ids = [
                        test_id
                        for test_id, value in zip(
                            problem.test_ids,
                            selection,
                        )
                        if value == 1
                    ]

                elif algorithm == "ortools_cp_sat":
                    selection, result = solve_ortools(
                        problem
                    )

                    status = result["solver_status"]

                    selected_ids = [
                        test_id
                        for test_id, value in zip(
                            problem.test_ids,
                            selection,
                        )
                        if value == 1
                    ]

                else:
                    selected_ids = select_baseline(
                        candidates,
                        algorithm,
                        budget,
                        HELDOUT_RANDOM_SEED + run_index,
                    )

                solver_seconds = (
                    time.perf_counter() - solver_start
                )

                metrics = evaluate_run(
                    run,
                    candidates,
                    selected_ids,
                    budget,
                )

                records.append({
                    "run_id": run_id,
                    "algorithm": algorithm,
                    "budget_seconds": budget,
                    "solver_seconds": solver_seconds,
                    "solver_status": status,
                    **metrics,
                })

                for test_id in selected_ids:
                    selections.append({
                        "run_id": run_id,
                        "algorithm": algorithm,
                        "budget_seconds": budget,
                        "test_id": test_id,
                    })

        print(
            f"[{run_index + 1}/{len(grouped)}] "
            f"{run_id} completed",
            flush=True,
        )

    results = pd.DataFrame(records)

    summary = summarize_results(results)

    results.to_csv(
        HELDOUT_RUNS_FILE,
        index=False,
    )

    summary.to_csv(
        HELDOUT_SUMMARY_FILE,
        index=False,
    )

    pd.DataFrame(selections).to_csv(
        HELDOUT_SELECTIONS_FILE,
        index=False,
    )

    generate_plots(summary)

    report = {
        "dataset": "LRTS",
        "project": "activemq",
        "partition": "test",
        "test_rows": len(dataframe),
        "test_failures": int(dataframe["failed"].sum()),
        "test_runs": len(grouped),
        "algorithms": ALGORITHMS,
        "budgets_seconds": HELDOUT_BUDGETS,
        "maximum_candidates": MAX_CLASSICAL_CANDIDATES,
        "total_experiments": len(results),
        "model_retrained": False,
        "candidate_roster": "retrospectively_observed",
        "candidate_policy": "shared_ml_risk_shortlist",
        "duration_constraint": "historical_estimated_duration",
        "runtime_measurement": "cumulative_serial_test_duration",
        "random_repetitions_per_run": 1,
        "all_estimated_budgets_feasible": bool(
            results["estimated_budget_feasible"].all()
        ),
        "results": json.loads(
            summary.to_json(orient="records")
        ),
        "total_evaluation_seconds": (
            time.perf_counter() - start
        ),
    }

    HELDOUT_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        summary[
            [
                "algorithm",
                "budget_seconds",
                "detected_failures",
                "failure_recall",
                "test_reduction",
                "actual_overrun_rate",
            ]
        ].to_string(index=False)
    )

    print("=" * 70)
    print("HELD-OUT EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()