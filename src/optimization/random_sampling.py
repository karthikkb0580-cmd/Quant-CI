import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import (
    BENCHMARK_RESULTS_DIR,
    RANDOM_SAMPLING_RESULTS_DIR,
    RANDOM_SAMPLING_REPORT_FILE,
    RANDOM_SAMPLING_SUMMARY_FILE,
    RANDOM_SAMPLING_AGGREGATE_FILE,
    RANDOM_QAOA_COMPARISON_FILE,
    RANDOM_QAOA_RATIO_PLOT,
    RANDOM_QAOA_FEASIBILITY_PLOT,
    RANDOM_SAMPLING_TOLERANCE,
    PRUNED_REPORT_FILE,
    PRUNED_EXACT_FILE,
    PRUNED_MATRIX_FILE,
    PRUNED_CANDIDATES_FILE,
    QAOA_EXPERIMENT_SUMMARY_FILE,
    HYBRID_SUMMARY_FILE,
)

from src.quantum.ising import (
    state_to_bitstring,
)

from src.quantum.qaoa_experiments import (
    evaluate_counts,
    summarize_measurements,
)

from src.quantum.hybrid_solver import (
    evaluate_repaired_measurements,
    summarize_repair,
)

from src.optimization.objective import (
    create_problem,
)


def read_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def generate_random_states(
    variables,
    shots,
    seed,
):
    if variables <= 0:
        raise ValueError(
            "Variable count must be positive."
        )

    if shots <= 0:
        raise ValueError(
            "Shot count must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    return rng.integers(
        0,
        2,
        size=(shots, variables),
        dtype=np.int8,
    )


def generate_random_counts(
    variables,
    shots,
    seed,
):
    states = generate_random_states(
        variables,
        shots,
        seed,
    )

    counts = Counter(
        state_to_bitstring(state)
        for state in states
    )

    if sum(counts.values()) != shots:
        raise RuntimeError(
            "Shot count mismatch."
        )

    return dict(counts)


def run_random_experiment(
    experiment,
    shots,
    seed,
    matrix,
    report,
    exact,
    problem,
):
    start = time.perf_counter()

    counts = generate_random_counts(
        report["reduced_qubits"],
        shots,
        seed,
    )

    generation_seconds = (
        time.perf_counter() - start
    )

    start_evaluation = time.perf_counter()

    measurements = evaluate_counts(
        counts,
        matrix,
        report,
        problem,
        exact["energy"],
    )

    raw_summary = summarize_measurements(
        measurements,
        exact["objective_value"],
    )

    evaluation_seconds = (
        time.perf_counter()
        - start_evaluation
    )

    start_repair = time.perf_counter()

    repaired = evaluate_repaired_measurements(
        measurements,
        matrix,
        report,
        problem,
        exact["objective_value"],
    )

    hybrid_summary = summarize_repair(
        measurements,
        repaired,
        exact["objective_value"],
    )

    repair_seconds = (
        time.perf_counter()
        - start_repair
    )

    total_seconds = (
        time.perf_counter() - start
    )

    output_dir = (
        RANDOM_SAMPLING_RESULTS_DIR
        / experiment
        / f"seed_{seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    measurements.to_csv(
        output_dir / "measurements.csv",
        index=False,
    )

    repaired.to_csv(
        output_dir / "repaired_measurements.csv",
        index=False,
    )

    result = {
        "experiment": experiment,
        "seed": seed,
        "shots": shots,
        "variables": report["reduced_qubits"],
        "unique_measurements": len(measurements),
        "raw_feasible_shots": raw_summary[
            "feasible_shots"
        ],
        "raw_optimal_shots": raw_summary[
            "optimal_shots"
        ],
        "raw_feasibility_rate": raw_summary[
            "feasible_measurement_rate"
        ],
        "raw_best_objective": raw_summary[
            "best_objective"
        ],
        "raw_approximation_ratio": raw_summary[
            "approximation_ratio"
        ],
        "raw_best_bitstring": raw_summary[
            "best_bitstring"
        ],
        "hybrid_feasible_shots": hybrid_summary[
            "hybrid_feasible_shots"
        ],
        "hybrid_optimal_shots": hybrid_summary[
            "hybrid_optimal_shots"
        ],
        "hybrid_feasibility_rate": hybrid_summary[
            "hybrid_feasibility_rate"
        ],
        "hybrid_best_objective": hybrid_summary[
            "hybrid_best_objective"
        ],
        "hybrid_approximation_ratio": hybrid_summary[
            "hybrid_approximation_ratio"
        ],
        "hybrid_best_bitstring": hybrid_summary[
            "hybrid_best_bitstring"
        ],
        "recovered_shots": hybrid_summary[
            "recovered_shots"
        ],
        "generation_seconds": generation_seconds,
        "evaluation_seconds": evaluation_seconds,
        "repair_seconds": repair_seconds,
        "raw_total_seconds": (
            generation_seconds
            + evaluation_seconds
        ),
        "hybrid_total_seconds": total_seconds,
    }

    (
        output_dir / "report.json"
    ).write_text(
        json.dumps(
            result,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return result


def aggregate_results(dataframe):
    return dataframe.groupby(
        "experiment",
        sort=False,
    ).agg(
        runs=("seed", "count"),
        shots=("shots", "first"),
        raw_mean_ratio=(
            "raw_approximation_ratio",
            "mean",
        ),
        hybrid_mean_ratio=(
            "hybrid_approximation_ratio",
            "mean",
        ),
        raw_mean_feasibility=(
            "raw_feasibility_rate",
            "mean",
        ),
        hybrid_mean_feasibility=(
            "hybrid_feasibility_rate",
            "mean",
        ),
        raw_optimal_runs=(
            "raw_optimal_shots",
            lambda values: int(
                (values > 0).sum()
            ),
        ),
        hybrid_optimal_runs=(
            "hybrid_optimal_shots",
            lambda values: int(
                (values > 0).sum()
            ),
        ),
        raw_mean_seconds=(
            "raw_total_seconds",
            "mean",
        ),
        hybrid_mean_seconds=(
            "hybrid_total_seconds",
            "mean",
        ),
    ).reset_index()


def create_comparison(
    random_results,
    qaoa_results,
    hybrid_results,
):
    keys = [
        "experiment",
        "seed",
    ]

    random_data = random_results[
        keys + [
            "shots",
            "raw_approximation_ratio",
            "hybrid_approximation_ratio",
            "raw_feasibility_rate",
            "hybrid_feasibility_rate",
            "raw_optimal_shots",
            "hybrid_optimal_shots",
            "raw_total_seconds",
            "hybrid_total_seconds",
        ]
    ].rename(
        columns={
            "raw_approximation_ratio":
                "random_raw_ratio",
            "hybrid_approximation_ratio":
                "random_hybrid_ratio",
            "raw_feasibility_rate":
                "random_raw_feasibility",
            "hybrid_feasibility_rate":
                "random_hybrid_feasibility",
            "raw_optimal_shots":
                "random_raw_optimal_shots",
            "hybrid_optimal_shots":
                "random_hybrid_optimal_shots",
            "raw_total_seconds":
                "random_raw_seconds",
            "hybrid_total_seconds":
                "random_hybrid_seconds",
        }
    )

    qaoa_data = qaoa_results[
        keys + [
            "shots",
            "approximation_ratio",
            "feasible_measurement_rate",
            "optimal_shots",
            "total_seconds",
        ]
    ].rename(
        columns={
            "shots": "qaoa_shots",
            "approximation_ratio":
                "qaoa_raw_ratio",
            "feasible_measurement_rate":
                "qaoa_raw_feasibility",
            "optimal_shots":
                "qaoa_raw_optimal_shots",
            "total_seconds":
                "qaoa_raw_seconds",
        }
    )

    hybrid_data = hybrid_results[
        keys + [
            "hybrid_approximation_ratio",
            "hybrid_feasibility_rate",
            "hybrid_optimal_shots",
            "hybrid_total_seconds",
        ]
    ].rename(
        columns={
            "hybrid_approximation_ratio":
                "qaoa_hybrid_ratio",
            "hybrid_feasibility_rate":
                "qaoa_hybrid_feasibility",
            "hybrid_optimal_shots":
                "qaoa_hybrid_optimal_shots",
            "hybrid_total_seconds":
                "qaoa_hybrid_seconds",
        }
    )

    comparison = random_data.merge(
        qaoa_data,
        on=keys,
        validate="one_to_one",
    ).merge(
        hybrid_data,
        on=keys,
        validate="one_to_one",
    )

    if len(comparison) != len(random_results):
        raise RuntimeError(
            "Comparison row count mismatch."
        )

    if not comparison["shots"].eq(
        comparison["qaoa_shots"]
    ).all():
        raise RuntimeError(
            "Sampling budgets differ."
        )

    comparison["raw_qaoa_advantage"] = (
        comparison["qaoa_raw_ratio"]
        - comparison["random_raw_ratio"]
    )

    comparison["hybrid_qaoa_advantage"] = (
        comparison["qaoa_hybrid_ratio"]
        - comparison["random_hybrid_ratio"]
    )

    return comparison


def generate_plots(comparison):
    specifications = [
        (
            [
                "random_raw_ratio",
                "random_hybrid_ratio",
                "qaoa_raw_ratio",
                "qaoa_hybrid_ratio",
            ],
            "Mean Approximation Ratio",
            RANDOM_QAOA_RATIO_PLOT,
        ),
        (
            [
                "random_raw_feasibility",
                "random_hybrid_feasibility",
                "qaoa_raw_feasibility",
                "qaoa_hybrid_feasibility",
            ],
            "Mean Feasibility Rate",
            RANDOM_QAOA_FEASIBILITY_PLOT,
        ),
    ]

    labels = [
        "Random",
        "Random + Repair",
        "QAOA",
        "QAOA + Repair",
    ]

    for columns, title, path in specifications:
        values = [
            comparison[column].mean()
            for column in columns
        ]

        plt.figure(
            figsize=(10, 6)
        )

        bars = plt.bar(
            labels,
            np.nan_to_num(
                values,
                nan=0.0,
            ),
        )

        for bar, value in zip(
            bars,
            values,
        ):
            label = (
                "N/A"
                if pd.isna(value)
                else f"{value:.4f}"
            )

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height() + 0.02,
                label,
                ha="center",
            )

        plt.ylabel(title)

        plt.title(
            f"QuantCI - {title}"
        )

        plt.ylim(
            0,
            1.15,
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=200,
        )

        plt.close()


def safe_mean(values):
    value = values.mean()

    return (
        None
        if pd.isna(value)
        else float(value)
    )


def main():
    start = time.perf_counter()

    BENCHMARK_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RANDOM_SAMPLING_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = read_json(
        PRUNED_REPORT_FILE
    )

    exact = read_json(
        PRUNED_EXACT_FILE
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
        budget=report["budget_seconds"],
        redundancy_penalty=report[
            "redundancy_penalty"
        ],
    )

    qaoa_results = pd.read_csv(
        QAOA_EXPERIMENT_SUMMARY_FILE
    )

    hybrid_results = pd.read_csv(
        HYBRID_SUMMARY_FILE
    )

    results = []

    for row in qaoa_results.itertuples(
        index=False
    ):
        print(
            f"Random sampling | "
            f"{row.experiment} | "
            f"seed={row.seed} | "
            f"shots={row.shots}",
            flush=True,
        )

        result = run_random_experiment(
            row.experiment,
            int(row.shots),
            int(row.seed),
            matrix,
            report,
            exact,
            problem,
        )

        results.append(
            result
        )

        print(
            "Random ratio:",
            result["raw_approximation_ratio"],
        )

        print(
            "Random + repair ratio:",
            result["hybrid_approximation_ratio"],
        )

    dataframe = pd.DataFrame(
        results
    )

    dataframe.to_csv(
        RANDOM_SAMPLING_SUMMARY_FILE,
        index=False,
    )

    aggregate = aggregate_results(
        dataframe
    )

    aggregate.to_csv(
        RANDOM_SAMPLING_AGGREGATE_FILE,
        index=False,
    )

    comparison = create_comparison(
        dataframe,
        qaoa_results,
        hybrid_results,
    )

    comparison.to_csv(
        RANDOM_QAOA_COMPARISON_FILE,
        index=False,
    )

    algorithms = {}

    for name in [
        "random_raw",
        "random_hybrid",
        "qaoa_raw",
        "qaoa_hybrid",
    ]:
        ratios = comparison[
            f"{name}_ratio"
        ]

        algorithms[name] = {
            "successful_runs": int(
                ratios.notna().sum()
            ),
            "mean_approximation_ratio": safe_mean(
                ratios
            ),
            "mean_feasibility_rate": safe_mean(
                comparison[
                    f"{name}_feasibility"
                ]
            ),
            "optimal_runs": int(
                (
                    comparison[
                        f"{name}_optimal_shots"
                    ] > 0
                ).sum()
            ),
            "mean_runtime_seconds": safe_mean(
                comparison[
                    f"{name}_seconds"
                ]
            ),
        }

    differences = comparison[
        "hybrid_qaoa_advantage"
    ].dropna()

    tolerance = RANDOM_SAMPLING_TOLERANCE

    final_report = {
        "partition": report["partition"],
        "run_id": report["run_id"],
        "matched_experiments": len(comparison),
        "independent_random_seed_shot_pairs": int(
            comparison[
                ["seed", "shots"]
            ].drop_duplicates().shape[0]
        ),
        "variables": report["reduced_qubits"],
        "budget_seconds": report["budget_seconds"],
        "exact_objective": exact[
            "objective_value"
        ],
        "total_random_shots": int(
            dataframe["shots"].sum()
        ),
        "algorithms": algorithms,
        "hybrid_comparison": {
            "comparable_runs": len(differences),
            "qaoa_better": int(
                (differences > tolerance).sum()
            ),
            "random_better": int(
                (differences < -tolerance).sum()
            ),
            "equal": int(
                (
                    differences.abs()
                    <= tolerance
                ).sum()
            ),
            "mean_ratio_difference": safe_mean(
                differences
            ),
        },
        "same_final_sampling_budget": True,
        "same_slack_repair": True,
        "quantum_optimizer_evaluations_matched": False,
        "evaluation_type": "single_validation_instance",
        "total_seconds": (
            time.perf_counter() - start
        ),
    }

    RANDOM_SAMPLING_REPORT_FILE.write_text(
        json.dumps(
            final_report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    generate_plots(
        comparison
    )

    print("=" * 70)
    print("QuantCI - Random Sampling Benchmark")
    print("=" * 70)

    print(
        aggregate.to_string(
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
    print("RANDOM SAMPLING BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()