import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.optimization.candidates import prepare_candidates
from src.optimization.objective import create_problem
from src.optimization.greedy import solve_greedy


BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "results"

OUTPUT_DIR = RESULTS_DIR / "runtime_uncertainty"

VALIDATION_FILE = RESULTS_DIR / "final_validation_predictions.csv"

TEST_FILE = RESULTS_DIR / "final_test_predictions.csv"

BUDGETS = [60, 120, 300, 600]

QUANTILES = [0.80, 0.90, 0.95]

MAX_CANDIDATES = 200

REDUNDANCY_PENALTY = 0.1

MIN_DURATION = 0.1

REQUIRED_COLUMNS = [
    "run_id",
    "test_id",
    "timestamp",
    "failed",
    "failure_score",
    "duration_seconds",
    "estimated_duration_seconds",
]


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def validate_data(dataframe):
    missing = set(REQUIRED_COLUMNS).difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if dataframe.empty:
        raise ValueError("Dataset is empty")

    if dataframe[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Missing required values")

    if dataframe.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError("Duplicate test executions")

    if not dataframe["failed"].isin([0, 1]).all():
        raise ValueError("Invalid failure labels")

    numeric_columns = [
        "failure_score",
        "duration_seconds",
        "estimated_duration_seconds",
    ]

    values = dataframe[numeric_columns].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError("Invalid numerical values")

    if (
        dataframe["duration_seconds"] < 0
    ).any():
        raise ValueError("Negative actual duration")

    if (
        dataframe["estimated_duration_seconds"] < 0
    ).any():
        raise ValueError("Negative estimated duration")

    if not dataframe["failure_score"].between(
        0, 1
    ).all():
        raise ValueError("Invalid failure score")

    return dataframe.copy()


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return validate_data(pd.read_csv(path))


def calculate_duration_ratios(
    actual,
    estimated,
):
    actual = np.asarray(actual, dtype=float)
    estimated = np.asarray(estimated, dtype=float)

    if actual.ndim != 1 or estimated.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if len(actual) != len(estimated):
        raise ValueError("Input lengths do not match")

    if len(actual) == 0:
        raise ValueError("Inputs cannot be empty")

    if not np.isfinite(actual).all():
        raise ValueError("Invalid actual durations")

    if not np.isfinite(estimated).all():
        raise ValueError("Invalid estimated durations")

    if (actual < 0).any():
        raise ValueError("Negative actual duration")

    if (estimated < 0).any():
        raise ValueError("Negative estimated duration")

    estimated = np.maximum(
        estimated,
        MIN_DURATION,
    )

    return actual / estimated


def fit_uncertainty_model(dataframe):
    dataframe = validate_data(dataframe)

    ratios = calculate_duration_ratios(
        dataframe["duration_seconds"],
        dataframe["estimated_duration_seconds"],
    )

    factors = {
        "point_estimate": 1.0,
    }

    for quantile in QUANTILES:
        name = f"p{int(quantile * 100)}"

        factor = np.quantile(
            ratios,
            quantile,
            method="higher",
        )

        factors[name] = float(
            max(1.0, factor)
        )

    return factors


def conservative_duration(
    estimated,
    factor,
):
    estimated = np.asarray(
        estimated,
        dtype=float,
    )

    if not np.isfinite(factor) or factor < 1:
        raise ValueError("Invalid safety factor")

    if not np.isfinite(estimated).all():
        raise ValueError("Invalid estimated duration")

    if (estimated < 0).any():
        raise ValueError("Negative estimated duration")

    return (
        np.maximum(estimated, MIN_DURATION)
        * factor
    )


def evaluate_coverage(
    dataframe,
    factors,
):
    actual = dataframe[
        "duration_seconds"
    ].to_numpy()

    estimated = dataframe[
        "estimated_duration_seconds"
    ].to_numpy()

    records = []

    for strategy, factor in factors.items():
        adjusted = conservative_duration(
            estimated,
            factor,
        )

        coverage = float(
            np.mean(actual <= adjusted)
        )

        records.append({
            "strategy": strategy,
            "safety_factor": factor,
            "individual_duration_coverage": coverage,
            "mean_actual_duration": float(
                actual.mean()
            ),
            "mean_adjusted_duration": float(
                adjusted.mean()
            ),
        })

    return pd.DataFrame(records)


def evaluate_selection(
    run,
    candidates,
    selected_ids,
    budget,
):
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate selected tests")

    if not set(selected_ids).issubset(
        set(candidates["test_id"])
    ):
        raise ValueError("Invalid selected test IDs")

    selected = candidates[
        candidates["test_id"].isin(selected_ids)
    ]

    actual_duration = float(
        selected["duration_seconds"].sum()
    )

    adjusted_duration = float(
        selected["estimated_duration_seconds"].sum()
    )

    total_duration = float(
        run["duration_seconds"].sum()
    )

    total_failures = int(
        run["failed"].sum()
    )

    detected_failures = int(
        selected["failed"].sum()
    )

    if adjusted_duration > budget + 1e-6:
        raise ValueError("Estimated budget exceeded")

    return {
        "total_tests": len(run),
        "selected_tests": len(selected),
        "total_failures": total_failures,
        "detected_failures": detected_failures,
        "missed_failures": (
            total_failures - detected_failures
        ),
        "full_duration_seconds": total_duration,
        "actual_duration_seconds": actual_duration,
        "adjusted_duration_seconds": adjusted_duration,
        "estimated_budget_feasible": True,
        "actual_budget_overrun": (
            actual_duration > budget
        ),
        "actual_overrun_seconds": max(
            0.0,
            actual_duration - budget,
        ),
        "failure_run": total_failures > 0,
        "failure_run_detected": detected_failures > 0,
    }


def run_experiments(
    dataframe,
    factors,
):
    records = []

    run_ids = dataframe["run_id"].unique()

    for run_id in run_ids:
        run = dataframe[
            dataframe["run_id"] == run_id
        ].copy()

        shortlist = prepare_candidates(
            dataframe,
            run_id,
            max_candidates=MAX_CANDIDATES,
        )

        shortlist = shortlist[
            [
                "run_id",
                "test_id",
                "failure_score",
                "estimated_duration_seconds",
            ]
        ].merge(
            run[
                [
                    "run_id",
                    "test_id",
                    "failed",
                    "duration_seconds",
                ]
            ],
            on=["run_id", "test_id"],
            how="left",
            validate="one_to_one",
        )

        for budget in BUDGETS:
            for strategy, factor in factors.items():
                candidates = shortlist.copy()

                candidates[
                    "estimated_duration_seconds"
                ] = conservative_duration(
                    candidates[
                        "estimated_duration_seconds"
                    ],
                    factor,
                )

                problem = create_problem(
                    candidates,
                    budget=budget,
                    redundancy_penalty=(
                        REDUNDANCY_PENALTY
                    ),
                )

                started = time.perf_counter()

                _, result, _ = solve_greedy(
                    problem
                )

                elapsed = (
                    time.perf_counter() - started
                )

                selected_ids = result[
                    "selected_tests"
                ]

                metrics = evaluate_selection(
                    run,
                    candidates,
                    selected_ids,
                    budget,
                )

                records.append({
                    "run_id": run_id,
                    "strategy": strategy,
                    "safety_factor": factor,
                    "budget_seconds": budget,
                    "solver_seconds": elapsed,
                    **metrics,
                })

    return pd.DataFrame(records)


def summarize_results(dataframe):
    records = []

    for (
        strategy,
        budget,
    ), group in dataframe.groupby(
        ["strategy", "budget_seconds"],
        sort=False,
    ):
        failures = int(
            group["total_failures"].sum()
        )

        detected = int(
            group["detected_failures"].sum()
        )

        failure_runs = int(
            group["failure_run"].sum()
        )

        detected_runs = int(
            group["failure_run_detected"].sum()
        )

        actual_duration = float(
            group["actual_duration_seconds"].sum()
        )

        full_duration = float(
            group["full_duration_seconds"].sum()
        )

        records.append({
            "strategy": strategy,
            "budget_seconds": int(budget),
            "runs": len(group),
            "total_failures": failures,
            "detected_failures": detected,
            "failure_recall": safe_divide(
                detected,
                failures,
            ),
            "failure_run_detection_rate": safe_divide(
                detected_runs,
                failure_runs,
            ),
            "missed_run_rate": (
                1.0 - safe_divide(
                    detected_runs,
                    failure_runs,
                )
                if failure_runs
                else 0.0
            ),
            "actual_budget_overrun_rate": float(
                group["actual_budget_overrun"].mean()
            ),
            "actual_budget_overrun_runs": int(
                group["actual_budget_overrun"].sum()
            ),
            "mean_overrun_seconds": float(
                group["actual_overrun_seconds"].mean()
            ),
            "estimated_feasibility_rate": float(
                group["estimated_budget_feasible"].mean()
            ),
            "actual_runtime_reduction": (
                1.0 - safe_divide(
                    actual_duration,
                    full_duration,
                )
            ),
            "mean_selected_tests": float(
                group["selected_tests"].mean()
            ),
            "mean_solver_seconds": float(
                group["solver_seconds"].mean()
            ),
        })

    return pd.DataFrame(records)


def create_plots(summary):
    metrics = [
        "failure_recall",
        "actual_budget_overrun_rate",
        "actual_runtime_reduction",
        "mean_selected_tests",
    ]

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for strategy, group in summary.groupby(
            "strategy"
        ):
            group = group.sort_values(
                "budget_seconds"
            )

            plt.plot(
                group["budget_seconds"],
                group[metric],
                marker="o",
                label=strategy,
            )

        plt.xlabel("Execution budget in seconds")

        plt.ylabel(
            metric.replace("_", " ").title()
        )

        plt.title(
            f"Runtime Uncertainty: {metric}"
        )

        plt.legend()

        plt.grid(alpha=0.25)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{metric}.png",
            dpi=200,
        )

        plt.close()


def main():
    started = time.perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation = load_data(
        VALIDATION_FILE
    )

    test = load_data(
        TEST_FILE
    )

    if len(validation) != 19738:
        raise ValueError(
            "Expected corrected validation partition"
        )

    if len(test) != 27786:
        raise ValueError(
            "Unexpected test partition size"
        )

    if int(validation["failed"].sum()) != 37:
        raise ValueError(
            "Unexpected validation failure count"
        )

    if int(test["failed"].sum()) != 15:
        raise ValueError(
            "Unexpected test failure count"
        )

    if set(validation["run_id"]).intersection(
        test["run_id"]
    ):
        raise ValueError(
            "Validation and test runs overlap"
        )

    validation_end = pd.to_datetime(
        validation["timestamp"],
        utc=True,
    ).max()

    test_start = pd.to_datetime(
        test["timestamp"],
        utc=True,
    ).min()

    if validation_end >= test_start:
        raise ValueError(
            "Invalid chronological boundary"
        )

    factors = fit_uncertainty_model(
        validation
    )

    validation_coverage = evaluate_coverage(
        validation,
        factors,
    )

    test_coverage = evaluate_coverage(
        test,
        factors,
    )

    experiments = run_experiments(
        test,
        factors,
    )

    summary = summarize_results(
        experiments
    )

    validation_coverage.to_csv(
        OUTPUT_DIR / "validation_coverage.csv",
        index=False,
    )

    test_coverage.to_csv(
        OUTPUT_DIR / "test_coverage.csv",
        index=False,
    )

    experiments.to_csv(
        OUTPUT_DIR / "runtime_experiments.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "runtime_summary.csv",
        index=False,
    )

    create_plots(summary)

    report = {
        "dataset": "LRTS",
        "project": "activemq",
        "method": (
            "empirical_duration_ratio_quantiles"
        ),
        "fitting_partition": "validation",
        "evaluation_partition": "test",
        "test_data_used_for_fitting": False,
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_runs": int(
            test["run_id"].nunique()
        ),
        "test_failures": int(
            test["failed"].sum()
        ),
        "safety_factors": factors,
        "budgets_seconds": BUDGETS,
        "maximum_candidates": MAX_CANDIDATES,
        "optimizer": "greedy",
        "risk_score": "original_uncalibrated",
        "total_experiments": len(experiments),
        "all_estimated_budgets_feasible": bool(
            experiments[
                "estimated_budget_feasible"
            ].all()
        ),
        "validation_coverage": (
            validation_coverage.to_dict(
                orient="records"
            )
        ),
        "test_coverage": (
            test_coverage.to_dict(
                orient="records"
            )
        ),
        "results": summary.to_dict(
            orient="records"
        ),
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    report_path = (
        OUTPUT_DIR / "runtime_uncertainty_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("=" * 70)
    print("QUANTCI RUNTIME UNCERTAINTY MODELING")
    print("=" * 70)

    print("\nSafety Factors")

    for strategy, factor in factors.items():
        print(f"{strategy}: {factor:.6f}")

    print("\nHeld-Out Duration Coverage")

    print(
        test_coverage.to_string(
            index=False
        )
    )

    print("\nOptimization Results")

    print(
        summary[
            [
                "strategy",
                "budget_seconds",
                "detected_failures",
                "failure_recall",
                "actual_budget_overrun_rate",
                "actual_runtime_reduction",
            ]
        ].to_string(index=False)
    )

    print("\nTotal experiments:", len(experiments))

    print("Report:", report_path)

    print("\nRUNTIME UNCERTAINTY EVALUATION COMPLETED")


if __name__ == "__main__":
    main()