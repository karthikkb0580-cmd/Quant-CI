import json

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import (
    MAX_QAOA_CANDIDATES,
    QUBO_DURATION_UNIT_SECONDS,
    QUBO_BUDGET_SECONDS,
    QUBO_PENALTY_MARGIN,
    QUBO_RESULTS_DIR,
    QUBO_REPORT_FILE,
    QUBO_MATRIX_FILE,
    QUBO_VARIABLES_FILE,
    QUBO_CANDIDATES_FILE,
    QUBO_MATRIX_PLOT,
)

from src.optimization.candidates import (
    load_predictions,
    prepare_candidates,
    get_run_ids,
)

from src.optimization.objective import (
    OptimizationProblem,
    create_problem,
)


@dataclass
class QUBOProblem:
    matrix: np.ndarray
    offset: float
    variable_names: list
    test_count: int
    duration_units: np.ndarray
    budget_units: int
    slack_weights: np.ndarray
    penalty: float
    duration_unit_seconds: float
    original_problem: OptimizationProblem

    @property
    def size(self):
        return len(self.variable_names)

    @property
    def slack_count(self):
        return len(self.slack_weights)


def discretize_durations(
    durations,
    budget,
    unit_seconds=QUBO_DURATION_UNIT_SECONDS,
):
    durations = np.asarray(
        durations,
        dtype=float,
    )

    if not np.isfinite(unit_seconds) or unit_seconds <= 0:
        raise ValueError("Invalid duration unit.")

    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Invalid budget.")

    if not np.isfinite(durations).all():
        raise ValueError("Invalid durations.")

    if (durations <= 0).any():
        raise ValueError("Durations must be positive.")

    duration_units = np.ceil(
        durations / unit_seconds
    ).astype(np.int64)

    budget_units = int(
        np.floor(
            budget / unit_seconds
        )
    )

    return duration_units, budget_units


def create_slack_weights(budget_units):
    if budget_units < 0:
        raise ValueError("Invalid budget units.")

    if budget_units == 0:
        return np.array([], dtype=np.int64)

    weights = []
    remaining = int(budget_units)
    power = 1

    while remaining > 0:
        weight = min(
            power,
            remaining,
        )

        weights.append(weight)

        remaining -= weight

        power *= 2

    return np.asarray(
        weights,
        dtype=np.int64,
    )


def calculate_penalty(
    problem,
    margin=QUBO_PENALTY_MARGIN,
):
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Invalid penalty margin.")

    return float(
        problem.risks.sum() + margin
    )


def create_qubo(
    problem,
    unit_seconds=QUBO_DURATION_UNIT_SECONDS,
    penalty=None,
):
    duration_units, budget_units = discretize_durations(
        problem.durations,
        problem.budget,
        unit_seconds,
    )

    slack_weights = create_slack_weights(
        budget_units
    )

    minimum_penalty = float(
        problem.risks.sum()
    )

    if penalty is None:
        penalty = calculate_penalty(
            problem
        )

    penalty = float(penalty)

    if not np.isfinite(penalty):
        raise ValueError("Invalid QUBO penalty.")

    if penalty <= minimum_penalty:
        raise ValueError(
            "Penalty must exceed total risk."
        )

    weights = np.concatenate(
        [
            duration_units,
            slack_weights,
        ]
    ).astype(float)

    test_count = problem.size

    size = len(weights)

    matrix = np.zeros(
        (size, size),
        dtype=float,
    )

    for i in range(test_count):
        matrix[i, i] -= problem.risks[i]

    for i in range(test_count):
        for j in range(i + 1, test_count):
            matrix[i, j] += (
                problem.redundancy_penalty
                * problem.similarity[i, j]
            )

    for i in range(size):
        matrix[i, i] += penalty * (
            weights[i] ** 2
            - 2 * budget_units * weights[i]
        )

    for i in range(size):
        for j in range(i + 1, size):
            matrix[i, j] += (
                2
                * penalty
                * weights[i]
                * weights[j]
            )

    offset = float(
        penalty * budget_units ** 2
    )

    variable_names = [
        f"x_{i}"
        for i in range(test_count)
    ]

    variable_names.extend(
        [
            f"slack_{i}"
            for i in range(len(slack_weights))
        ]
    )

    if not np.isfinite(matrix).all():
        raise ValueError("Invalid QUBO matrix.")

    return QUBOProblem(
        matrix=matrix,
        offset=offset,
        variable_names=variable_names,
        test_count=test_count,
        duration_units=duration_units,
        budget_units=budget_units,
        slack_weights=slack_weights,
        penalty=penalty,
        duration_unit_seconds=float(unit_seconds),
        original_problem=problem,
    )


def validate_binary_state(qubo, state):
    state = np.asarray(
        state,
        dtype=float,
    )

    if state.shape != (qubo.size,):
        raise ValueError("Invalid state size.")

    if not np.isin(state, [0, 1]).all():
        raise ValueError("State must be binary.")

    return state


def qubo_energy(qubo, state):
    state = validate_binary_state(
        qubo,
        state,
    )

    return float(
        state @ qubo.matrix @ state
        + qubo.offset
    )


def decode_state(qubo, state):
    state = validate_binary_state(
        qubo,
        state,
    ).astype(np.int8)

    selection = state[
        :qubo.test_count
    ]

    slack_bits = state[
        qubo.test_count:
    ]

    used_units = int(
        qubo.duration_units @ selection
    )

    slack_units = int(
        qubo.slack_weights @ slack_bits
    )

    residual = (
        used_units
        + slack_units
        - qubo.budget_units
    )

    return {
        "selection": selection,
        "slack_bits": slack_bits,
        "used_duration_units": used_units,
        "slack_units": slack_units,
        "constraint_residual": residual,
        "discrete_feasible": (
            used_units <= qubo.budget_units
        ),
        "constraint_satisfied": residual == 0,
    }


def direct_energy(qubo, state):
    decoded = decode_state(
        qubo,
        state,
    )

    selection = decoded["selection"]

    problem = qubo.original_problem

    risk = float(
        problem.risks @ selection
    )

    redundancy = float(
        0.5
        * selection
        @ problem.similarity
        @ selection
    )

    violation = decoded[
        "constraint_residual"
    ]

    return float(
        -risk
        + problem.redundancy_penalty * redundancy
        + qubo.penalty * violation ** 2
    )


def generate_matrix_plot(qubo):
    QUBO_MATRIX_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 8))

    plt.imshow(
        qubo.matrix,
        cmap="coolwarm",
        aspect="auto",
    )

    plt.colorbar(
        label="QUBO coefficient"
    )

    plt.xticks(
        range(qubo.size),
        qubo.variable_names,
        rotation=90,
    )

    plt.yticks(
        range(qubo.size),
        qubo.variable_names,
    )

    plt.title("QuantCI QUBO Coefficient Matrix")

    plt.tight_layout()

    plt.savefig(
        QUBO_MATRIX_PLOT,
        dpi=200,
    )

    plt.close()


def main():
    print("=" * 70)
    print("QuantCI - QUBO Formulation")
    print("=" * 70)

    QUBO_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = load_predictions(
        "validation"
    )

    run_id = get_run_ids(
        predictions
    )[0]

    candidates = prepare_candidates(
        predictions,
        run_id,
        MAX_QAOA_CANDIDATES,
    )

    problem = create_problem(
        candidates,
        QUBO_BUDGET_SECONDS,
    )

    qubo = create_qubo(
        problem
    )

    candidates.to_csv(
        QUBO_CANDIDATES_FILE,
        index=False,
    )

    pd.DataFrame(
        qubo.matrix,
        index=qubo.variable_names,
        columns=qubo.variable_names,
    ).to_csv(
        QUBO_MATRIX_FILE
    )

    variables = pd.DataFrame(
        {
            "variable": qubo.variable_names,
            "type": (
                ["test"] * qubo.test_count
                + ["slack"] * qubo.slack_count
            ),
            "weight": np.concatenate(
                [
                    qubo.duration_units,
                    qubo.slack_weights,
                ]
            ),
        }
    )

    variables.to_csv(
        QUBO_VARIABLES_FILE,
        index=False,
    )

    rng = np.random.default_rng(42)

    maximum_error = 0.0

    for _ in range(100):
        state = rng.integers(
            0,
            2,
            size=qubo.size,
        )

        error = abs(
            qubo_energy(qubo, state)
            - direct_energy(qubo, state)
        )

        maximum_error = max(
            maximum_error,
            error,
        )

    if maximum_error > 1e-7:
        raise RuntimeError(
            "QUBO energy verification failed."
        )

    report = {
        "partition": "validation",
        "run_id": run_id,
        "test_variables": qubo.test_count,
        "slack_variables": qubo.slack_count,
        "total_binary_variables": qubo.size,
        "possible_states": 2 ** qubo.size,
        "matrix_shape": list(
            qubo.matrix.shape
        ),
        "budget_seconds": problem.budget,
        "duration_unit_seconds": (
            qubo.duration_unit_seconds
        ),
        "budget_units": qubo.budget_units,
        "duration_units": (
            qubo.duration_units.tolist()
        ),
        "slack_weights": (
            qubo.slack_weights.tolist()
        ),
        "constraint_penalty": qubo.penalty,
        "constant_offset": qubo.offset,
        "total_risk": float(
            problem.risks.sum()
        ),
        "redundancy_penalty": (
            problem.redundancy_penalty
        ),
        "matrix_nonzero_coefficients": int(
            np.count_nonzero(
                qubo.matrix
            )
        ),
        "maximum_energy_verification_error": (
            maximum_error
        ),
        "energy_verification_passed": True,
        "matrix_convention": "upper_triangular",
    }

    QUBO_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    generate_matrix_plot(
        qubo
    )

    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    print("=" * 70)
    print("QUBO FORMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()