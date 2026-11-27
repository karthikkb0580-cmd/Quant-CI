import json

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_BUDGET_SECONDS,
    REDUNDANCY_PENALTY,
    OPTIMIZATION_SIMILARITY_THRESHOLD,
    OPTIMIZATION_TOLERANCE,
    OPTIMIZATION_REPORT_FILE,
    MAX_CLASSICAL_CANDIDATES,
)

from src.features.text_features import tokenize

from src.optimization.candidates import (
    load_predictions,
    prepare_candidates,
    get_run_ids,
)


@dataclass
class OptimizationProblem:
    test_ids: list
    risks: np.ndarray
    durations: np.ndarray
    similarity: np.ndarray
    budget: float
    redundancy_penalty: float

    def __post_init__(self):
        self.test_ids = list(self.test_ids)

        self.risks = np.asarray(
            self.risks,
            dtype=float,
        )

        self.durations = np.asarray(
            self.durations,
            dtype=float,
        )

        self.similarity = np.asarray(
            self.similarity,
            dtype=float,
        )

        self.budget = float(self.budget)

        self.redundancy_penalty = float(
            self.redundancy_penalty
        )

        self.validate()

    @property
    def size(self):
        return len(self.test_ids)

    def validate(self):
        n = self.size

        if n == 0:
            raise ValueError(
                "Optimization problem is empty."
            )

        if len(set(self.test_ids)) != n:
            raise ValueError(
                "Duplicate test IDs."
            )

        if self.risks.shape != (n,):
            raise ValueError(
                "Invalid risk dimensions."
            )

        if self.durations.shape != (n,):
            raise ValueError(
                "Invalid duration dimensions."
            )

        if self.similarity.shape != (n, n):
            raise ValueError(
                "Invalid similarity dimensions."
            )

        if not np.isfinite(self.risks).all():
            raise ValueError(
                "Non-finite risk scores."
            )

        if not np.isfinite(self.durations).all():
            raise ValueError(
                "Non-finite durations."
            )

        if not np.isfinite(self.similarity).all():
            raise ValueError(
                "Non-finite similarity values."
            )

        if np.any(self.risks < 0):
            raise ValueError(
                "Negative risk scores."
            )

        if np.any(self.risks > 1):
            raise ValueError(
                "Risk scores exceed one."
            )

        if np.any(self.durations <= 0):
            raise ValueError(
                "Durations must be positive."
            )

        if np.any(self.similarity < 0):
            raise ValueError(
                "Negative similarity."
            )

        if np.any(self.similarity > 1):
            raise ValueError(
                "Similarity exceeds one."
            )

        if not np.allclose(
            self.similarity,
            self.similarity.T,
        ):
            raise ValueError(
                "Similarity matrix must be symmetric."
            )

        if not np.allclose(
            np.diag(self.similarity),
            0,
        ):
            raise ValueError(
                "Similarity diagonal must be zero."
            )

        if not np.isfinite(self.budget):
            raise ValueError(
                "Invalid budget."
            )

        if self.budget < 0:
            raise ValueError(
                "Budget cannot be negative."
            )

        if not np.isfinite(
            self.redundancy_penalty
        ):
            raise ValueError(
                "Invalid redundancy penalty."
            )

        if self.redundancy_penalty < 0:
            raise ValueError(
                "Redundancy penalty cannot be negative."
            )


def test_tokens(test_id):
    class_name = str(test_id).split(".")[-1]

    tokens = tokenize(class_name)

    return tokens.difference(
        {"test", "tests", "case"}
    )


def compute_similarity(
    first,
    second,
):
    first_tokens = test_tokens(first)

    second_tokens = test_tokens(second)

    union = first_tokens | second_tokens

    if not union:
        return 0.0

    return len(
        first_tokens & second_tokens
    ) / len(union)


def build_similarity_matrix(
    test_ids,
    threshold=OPTIMIZATION_SIMILARITY_THRESHOLD,
):
    if not 0 <= threshold <= 1:
        raise ValueError(
            "Invalid similarity threshold."
        )

    n = len(test_ids)

    matrix = np.zeros(
        (n, n),
        dtype=np.float64,
    )

    token_sets = [
        test_tokens(test_id)
        for test_id in test_ids
    ]

    for i in range(n):
        for j in range(i + 1, n):
            union = (
                token_sets[i] | token_sets[j]
            )

            if not union:
                continue

            similarity = len(
                token_sets[i] & token_sets[j]
            ) / len(union)

            if similarity >= threshold:
                matrix[i, j] = similarity

                matrix[j, i] = similarity

    return matrix


def create_problem(
    candidates,
    budget=DEFAULT_BUDGET_SECONDS,
    redundancy_penalty=REDUNDANCY_PENALTY,
):
    test_ids = candidates[
        "test_id"
    ].tolist()

    risks = candidates[
        "failure_score"
    ].to_numpy(dtype=float)

    durations = candidates[
        "estimated_duration_seconds"
    ].to_numpy(dtype=float)

    similarity = build_similarity_matrix(
        test_ids
    )

    return OptimizationProblem(
        test_ids=test_ids,
        risks=risks,
        durations=durations,
        similarity=similarity,
        budget=budget,
        redundancy_penalty=redundancy_penalty,
    )


def validate_selection(
    problem,
    selection,
):
    selection = np.asarray(
        selection,
        dtype=float,
    )

    if selection.shape != (
        problem.size,
    ):
        raise ValueError(
            "Invalid selection dimensions."
        )

    if not np.isin(
        selection,
        [0, 1],
    ).all():
        raise ValueError(
            "Selection must be binary."
        )

    return selection


def calculate_risk(
    problem,
    selection,
):
    x = validate_selection(
        problem,
        selection,
    )

    return float(
        problem.risks @ x
    )


def calculate_duration(
    problem,
    selection,
):
    x = validate_selection(
        problem,
        selection,
    )

    return float(
        problem.durations @ x
    )


def calculate_redundancy(
    problem,
    selection,
):
    x = validate_selection(
        problem,
        selection,
    )

    return float(
        0.5 * x @ problem.similarity @ x
    )


def objective_value(
    problem,
    selection,
):
    risk = calculate_risk(
        problem,
        selection,
    )

    redundancy = calculate_redundancy(
        problem,
        selection,
    )

    return float(
        risk
        - problem.redundancy_penalty
        * redundancy
    )


def is_feasible(
    problem,
    selection,
):
    duration = calculate_duration(
        problem,
        selection,
    )

    return bool(
        duration
        <= problem.budget
        + OPTIMIZATION_TOLERANCE
    )


def marginal_gain(
    problem,
    selection,
    candidate_index,
):
    x = validate_selection(
        problem,
        selection,
    )

    if not 0 <= candidate_index < problem.size:
        raise IndexError(
            "Invalid candidate index."
        )

    if x[candidate_index] == 1:
        return 0.0

    penalty = (
        problem.redundancy_penalty
        * problem.similarity[candidate_index]
        @ x
    )

    return float(
        problem.risks[candidate_index]
        - penalty
    )


def evaluate_selection(
    problem,
    selection,
):
    x = validate_selection(
        problem,
        selection,
    )

    duration = calculate_duration(
        problem,
        x,
    )

    risk = calculate_risk(
        problem,
        x,
    )

    redundancy = calculate_redundancy(
        problem,
        x,
    )

    selected_indices = np.flatnonzero(x)

    return {
        "total_candidates": problem.size,
        "selected_count": int(x.sum()),
        "selected_tests": [
            problem.test_ids[i]
            for i in selected_indices
        ],
        "estimated_duration_seconds": duration,
        "budget_seconds": problem.budget,
        "remaining_budget_seconds": (
            problem.budget - duration
        ),
        "total_risk_score": risk,
        "redundancy": redundancy,
        "redundancy_penalty": (
            problem.redundancy_penalty
            * redundancy
        ),
        "objective_value": objective_value(
            problem,
            x,
        ),
        "feasible": is_feasible(
            problem,
            x,
        ),
        "selection_rate": float(
            x.mean()
        ),
    }


def main():
    print("=" * 60)
    print("QuantCI - Optimization Objective")
    print("=" * 60)

    dataframe = load_predictions(
        "validation"
    )

    run_id = get_run_ids(
        dataframe
    )[0]

    candidates = prepare_candidates(
        dataframe,
        run_id,
        max_candidates=MAX_CLASSICAL_CANDIDATES,
    )

    problem = create_problem(
        candidates
    )

    empty_selection = np.zeros(
        problem.size,
        dtype=int,
    )

    single_selection = empty_selection.copy()

    feasible_indices = np.flatnonzero(
        problem.durations <= problem.budget
    )

    if len(feasible_indices):
        best_index = feasible_indices[
            np.argmax(
                problem.risks[feasible_indices]
            )
        ]

        single_selection[best_index] = 1

    full_selection = np.ones(
        problem.size,
        dtype=int,
    )

    report = {
        "run_id": run_id,
        "partition": "validation",
        "candidate_count": problem.size,
        "budget_seconds": problem.budget,
        "redundancy_penalty": (
            problem.redundancy_penalty
        ),
        "similarity_threshold": (
            OPTIMIZATION_SIMILARITY_THRESHOLD
        ),
        "similarity_pairs": int(
            np.count_nonzero(
                np.triu(
                    problem.similarity,
                    k=1,
                )
            )
        ),
        "empty_selection": evaluate_selection(
            problem,
            empty_selection,
        ),
        "single_selection": evaluate_selection(
            problem,
            single_selection,
        ),
        "full_selection": evaluate_selection(
            problem,
            full_selection,
        ),
    }

    OPTIMIZATION_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OPTIMIZATION_REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print(f"\nRun ID: {run_id}")
    print(f"Candidates: {problem.size}")
    print(f"Budget: {problem.budget} seconds")

    print(
        f"Similarity pairs: "
        f"{report['similarity_pairs']}"
    )

    for name in [
        "empty_selection",
        "single_selection",
        "full_selection",
    ]:
        result = report[name]

        print(f"\n{name}")

        print(
            f"Selected: "
            f"{result['selected_count']}"
        )

        print(
            f"Duration: "
            f"{result['estimated_duration_seconds']:.4f}"
        )

        print(
            f"Objective: "
            f"{result['objective_value']:.6f}"
        )

        print(
            f"Feasible: "
            f"{result['feasible']}"
        )

    print(
        f"\nReport: {OPTIMIZATION_REPORT_FILE}"
    )

    print("=" * 60)
    print("OPTIMIZATION OBJECTIVE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()