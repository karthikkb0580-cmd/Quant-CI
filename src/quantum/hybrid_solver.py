import json
import time
from itertools import product

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import (
    QUANTUM_EXPERIMENTS_DIR,
    PRUNED_REPORT_FILE,
    PRUNED_MATRIX_FILE,
    PRUNED_EXACT_FILE,
    PRUNED_CANDIDATES_FILE,
    QAOA_EXPERIMENT_SUMMARY_FILE,
    HYBRID_RESULTS_DIR,
    HYBRID_REPORT_FILE,
    HYBRID_SUMMARY_FILE,
    HYBRID_AGGREGATE_FILE,
    HYBRID_RATIO_PLOT,
    HYBRID_FEASIBILITY_PLOT,
    HYBRID_TOLERANCE,
)

from src.quantum.ising import (
    bitstring_to_state,
    state_to_bitstring,
)

from src.optimization.objective import (
    create_problem,
    evaluate_selection,
)


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def encode_slack(value, weights):
    weights = np.asarray(weights)

    if weights.ndim != 1:
        raise ValueError("Invalid slack weights.")

    if not np.isfinite(weights).all():
        raise ValueError("Invalid slack weights.")

    if np.any(weights <= 0):
        raise ValueError("Slack weights must be positive.")

    if np.any(weights != np.floor(weights)):
        raise ValueError("Slack weights must be integers.")

    if not np.isfinite(value):
        raise ValueError("Invalid slack value.")

    if value < 0 or value != int(value):
        raise ValueError("Slack must be a nonnegative integer.")

    for bits in product(
        [0, 1],
        repeat=len(weights),
    ):
        state = np.asarray(
            bits,
            dtype=np.int8,
        )

        if int(state @ weights) == value:
            return state

    raise ValueError(
        "Slack value cannot be represented."
    )


def repair_state(state, report):
    state = np.asarray(state)

    if state.ndim != 1:
        raise ValueError("Invalid state dimensions.")

    if not np.isin(state, [0, 1]).all():
        raise ValueError("State must be binary.")

    if len(state) != report["reduced_qubits"]:
        raise ValueError("State size mismatch.")

    state = state.astype(np.int8)

    test_count = report[
        "retained_test_variables"
    ]

    durations = np.asarray(
        report["duration_units"],
        dtype=int,
    )

    weights = np.asarray(
        report["slack_weights"],
        dtype=int,
    )

    if len(durations) != test_count:
        raise ValueError("Duration size mismatch.")

    if test_count + len(weights) != len(state):
        raise ValueError("Slack size mismatch.")

    selection = state[:test_count]

    used_units = int(
        selection @ durations
    )

    budget_units = int(
        report["budget_units"]
    )

    if used_units > budget_units:
        return None

    required_slack = (
        budget_units - used_units
    )

    original_slack = int(
        state[test_count:] @ weights
    )

    if original_slack == required_slack:
        return state.copy()

    slack_bits = encode_slack(
        required_slack,
        weights,
    )

    repaired = np.concatenate(
        [
            selection,
            slack_bits,
        ]
    )

    return repaired


def evaluate_repaired_measurements(
    measurements,
    matrix,
    report,
    problem,
    exact_objective,
):
    records = []

    test_count = report[
        "retained_test_variables"
    ]

    durations = np.asarray(
        report["duration_units"],
        dtype=int,
    )

    weights = np.asarray(
        report["slack_weights"],
        dtype=int,
    )

    budget_units = report[
        "budget_units"
    ]

    for row in measurements.itertuples(
        index=False
    ):
        state = bitstring_to_state(
            str(row.bitstring)
        )

        repaired = repair_state(
            state,
            report,
        )

        if repaired is None:
            continue

        selection = repaired[:test_count]

        if not np.array_equal(
            selection,
            state[:test_count],
        ):
            raise RuntimeError(
                "Repair changed test selection."
            )

        used_units = int(
            selection @ durations
        )

        slack_units = int(
            repaired[test_count:] @ weights
        )

        if used_units + slack_units != budget_units:
            raise RuntimeError(
                "Repaired constraint violated."
            )

        evaluation = evaluate_selection(
            problem,
            selection,
        )

        if not evaluation["feasible"]:
            raise RuntimeError(
                "Repaired solution exceeds budget."
            )

        objective = float(
            evaluation["objective_value"]
        )

        energy = float(
            repaired @ matrix @ repaired
            + report["constant_offset"]
        )

        if not np.isclose(
            energy,
            -objective,
            atol=HYBRID_TOLERANCE,
            rtol=0,
        ):
            raise RuntimeError(
                "Repaired energy mismatch."
            )

        if objective > exact_objective + HYBRID_TOLERANCE:
            raise RuntimeError(
                "Objective exceeds exact optimum."
            )

        records.append(
            {
                "original_bitstring": str(
                    row.bitstring
                ),
                "repaired_bitstring": (
                    state_to_bitstring(repaired)
                ),
                "count": int(row.count),
                "originally_feasible": bool(
                    row.feasible
                ),
                "slack_changed": bool(
                    not np.array_equal(
                        state,
                        repaired,
                    )
                ),
                "used_duration_units": used_units,
                "slack_units": slack_units,
                "selected_count": evaluation[
                    "selected_count"
                ],
                "estimated_duration_seconds": evaluation[
                    "estimated_duration_seconds"
                ],
                "objective": objective,
                "energy": energy,
                "optimal": bool(
                    np.isclose(
                        objective,
                        exact_objective,
                        atol=HYBRID_TOLERANCE,
                        rtol=0,
                    )
                ),
            }
        )

    columns = [
        "original_bitstring",
        "repaired_bitstring",
        "count",
        "originally_feasible",
        "slack_changed",
        "used_duration_units",
        "slack_units",
        "selected_count",
        "estimated_duration_seconds",
        "objective",
        "energy",
        "optimal",
    ]

    return pd.DataFrame(
        records,
        columns=columns,
    )


def summarize_repair(
    original,
    repaired,
    exact_objective,
):
    total_shots = int(
        original["count"].sum()
    )

    if total_shots <= 0:
        raise ValueError("No measurements.")

    raw_feasible_shots = int(
        original.loc[
            original["feasible"],
            "count",
        ].sum()
    )

    raw_optimal_shots = int(
        original.loc[
            original["optimal"],
            "count",
        ].sum()
    )

    repaired_feasible_shots = int(
        repaired["count"].sum()
    )

    repaired_optimal_shots = int(
        repaired.loc[
            repaired["optimal"].astype(bool),
            "count",
        ].sum()
    )

    recovered_shots = (
        repaired_feasible_shots
        - raw_feasible_shots
    )

    if recovered_shots < 0:
        raise RuntimeError(
            "Repair lost feasible measurements."
        )

    if repaired_feasible_shots > total_shots:
        raise RuntimeError(
            "Measurement count exceeded."
        )

    if repaired_optimal_shots < raw_optimal_shots:
        raise RuntimeError(
            "Repair lost optimal measurements."
        )

    raw = original[
        original["feasible"]
    ]

    raw_best = (
        float(raw["objective"].max())
        if not raw.empty
        else None
    )

    result = {
        "total_shots": total_shots,
        "raw_feasible_shots": raw_feasible_shots,
        "hybrid_feasible_shots": repaired_feasible_shots,
        "recovered_shots": recovered_shots,
        "rejected_shots": (
            total_shots
            - repaired_feasible_shots
        ),
        "raw_feasibility_rate": (
            raw_feasible_shots / total_shots
        ),
        "hybrid_feasibility_rate": (
            repaired_feasible_shots / total_shots
        ),
        "raw_optimal_shots": raw_optimal_shots,
        "hybrid_optimal_shots": repaired_optimal_shots,
        "raw_optimal_rate": (
            raw_optimal_shots / total_shots
        ),
        "hybrid_optimal_rate": (
            repaired_optimal_shots / total_shots
        ),
        "raw_best_objective": raw_best,
        "hybrid_best_objective": None,
        "raw_approximation_ratio": None,
        "hybrid_approximation_ratio": None,
        "objective_improvement": None,
        "hybrid_best_bitstring": None,
        "selected_count": None,
        "estimated_duration_seconds": None,
        "feasible_solution_found": (
            not repaired.empty
        ),
    }

    if raw_best is not None:
        if exact_objective > 0:
            result["raw_approximation_ratio"] = (
                raw_best / exact_objective
            )
        elif abs(raw_best) <= HYBRID_TOLERANCE:
            result["raw_approximation_ratio"] = 1.0

    if repaired.empty:
        return result

    best = repaired.sort_values(
        [
            "objective",
            "count",
            "repaired_bitstring",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).iloc[0]

    best_objective = float(
        best["objective"]
    )

    if raw_best is not None:
        if best_objective < raw_best - HYBRID_TOLERANCE:
            raise RuntimeError(
                "Hybrid objective decreased."
            )

        result["objective_improvement"] = (
            best_objective - raw_best
        )

    ratio = None

    if exact_objective > 0:
        ratio = (
            best_objective / exact_objective
        )
    elif abs(best_objective) <= HYBRID_TOLERANCE:
        ratio = 1.0

    result.update(
        {
            "hybrid_best_objective": best_objective,
            "hybrid_approximation_ratio": ratio,
            "hybrid_best_bitstring": str(
                best["repaired_bitstring"]
            ),
            "selected_count": int(
                best["selected_count"]
            ),
            "estimated_duration_seconds": float(
                best["estimated_duration_seconds"]
            ),
        }
    )

    return result


def process_experiment(
    experiment,
    seed,
    matrix,
    report,
    problem,
    exact_objective,
):
    start = time.perf_counter()

    path = (
        QUANTUM_EXPERIMENTS_DIR
        / experiment
        / f"seed_{seed}"
        / "measurements.csv"
    )

    measurements = pd.read_csv(
        path,
        dtype={"bitstring": str},
    )

    if not measurements["bitstring"].str.len().eq(
        report["reduced_qubits"]
    ).all():
        raise RuntimeError(
            "Invalid measurement bitstring length."
        )

    if not measurements["count"].gt(0).all():
        raise RuntimeError(
            "Invalid measurement counts."
        )

    if measurements["bitstring"].duplicated().any():
        raise RuntimeError(
            "Duplicate measurement bitstrings."
        )

    repaired = evaluate_repaired_measurements(
        measurements,
        matrix,
        report,
        problem,
        exact_objective,
    )

    summary = summarize_repair(
        measurements,
        repaired,
        exact_objective,
    )

    output_dir = (
        HYBRID_RESULTS_DIR
        / experiment
        / f"seed_{seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    repaired.to_csv(
        output_dir / "repaired_measurements.csv",
        index=False,
    )

    summary.update(
        {
            "experiment": experiment,
            "seed": seed,
            "repair_seconds": (
                time.perf_counter() - start
            ),
        }
    )

    (
        output_dir / "report.json"
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
    return dataframe.groupby(
        "experiment",
        sort=False,
    ).agg(
        runs=("seed", "count"),
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
        recovered_shots=(
            "recovered_shots",
            "sum",
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
        mean_repair_seconds=(
            "repair_seconds",
            "mean",
        ),
    ).reset_index()


def generate_plots(aggregate):
    configurations = aggregate[
        "experiment"
    ]

    positions = np.arange(
        len(configurations)
    )

    width = 0.35

    specifications = [
        (
            "raw_mean_ratio",
            "hybrid_mean_ratio",
            "Approximation Ratio",
            HYBRID_RATIO_PLOT,
        ),
        (
            "raw_mean_feasibility",
            "hybrid_mean_feasibility",
            "Feasible Measurement Rate",
            HYBRID_FEASIBILITY_PLOT,
        ),
    ]

    for raw, hybrid, label, path in specifications:
        plt.figure(
            figsize=(11, 6)
        )

        plt.bar(
            positions - width / 2,
            aggregate[raw],
            width,
            label="Raw QAOA",
        )

        plt.bar(
            positions + width / 2,
            aggregate[hybrid],
            width,
            label="Hybrid QAOA",
        )

        plt.xticks(
            positions,
            configurations,
            rotation=20,
            ha="right",
        )

        plt.ylabel(label)

        plt.title(
            f"QuantCI - {label}"
        )

        plt.ylim(0, 1.1)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=200,
        )

        plt.close()


def main():
    start = time.perf_counter()

    HYBRID_RESULTS_DIR.mkdir(
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
    ).to_numpy(dtype=float)

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

    experiments = pd.read_csv(
        QAOA_EXPERIMENT_SUMMARY_FILE
    )

    results = []

    for row in experiments.itertuples(
        index=False
    ):
        print(
            f"Processing {row.experiment} | "
            f"seed={row.seed}",
            flush=True,
        )

        result = process_experiment(
            row.experiment,
            int(row.seed),
            matrix,
            report,
            problem,
            exact["objective_value"],
        )

        if result["total_shots"] != int(row.shots):
            raise RuntimeError(
                "Original shot count mismatch."
            )

        result["original_qaoa_seconds"] = float(
            row.total_seconds
        )

        result["hybrid_total_seconds"] = (
            float(row.total_seconds)
            + result["repair_seconds"]
        )

        results.append(result)

        print(
            "Raw ratio:",
            result["raw_approximation_ratio"],
        )

        print(
            "Hybrid ratio:",
            result["hybrid_approximation_ratio"],
        )

        print(
            "Recovered shots:",
            result["recovered_shots"],
        )

    dataframe = pd.DataFrame(
        results
    )

    dataframe.to_csv(
        HYBRID_SUMMARY_FILE,
        index=False,
    )

    aggregate = aggregate_results(
        dataframe
    )

    aggregate.to_csv(
        HYBRID_AGGREGATE_FILE,
        index=False,
    )

    final_report = {
        "algorithm": "qaoa_with_classical_slack_repair",
        "partition": report["partition"],
        "run_id": report["run_id"],
        "total_experiments": len(dataframe),
        "total_shots": int(
            dataframe["total_shots"].sum()
        ),
        "raw_feasible_shots": int(
            dataframe["raw_feasible_shots"].sum()
        ),
        "hybrid_feasible_shots": int(
            dataframe["hybrid_feasible_shots"].sum()
        ),
        "total_recovered_shots": int(
            dataframe["recovered_shots"].sum()
        ),
        "raw_optimal_runs": int(
            (
                dataframe["raw_optimal_shots"] > 0
            ).sum()
        ),
        "hybrid_optimal_runs": int(
            (
                dataframe["hybrid_optimal_shots"] > 0
            ).sum()
        ),
        "raw_mean_approximation_ratio": float(
            dataframe[
                "raw_approximation_ratio"
            ].mean()
        ),
        "hybrid_mean_approximation_ratio": float(
            dataframe[
                "hybrid_approximation_ratio"
            ].mean()
        ),
        "raw_mean_feasibility_rate": float(
            dataframe[
                "raw_feasibility_rate"
            ].mean()
        ),
        "hybrid_mean_feasibility_rate": float(
            dataframe[
                "hybrid_feasibility_rate"
            ].mean()
        ),
        "exact_objective": exact[
            "objective_value"
        ],
        "test_selection_modified": False,
        "quantum_circuit_rerun": False,
        "classical_fallback_used": False,
        "total_repair_seconds": (
            time.perf_counter() - start
        ),
    }

    HYBRID_REPORT_FILE.write_text(
        json.dumps(
            final_report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    generate_plots(aggregate)

    print("=" * 70)
    print("QuantCI - Hybrid Quantum Optimization")
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
    print("HYBRID OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()