import numpy as np
import pytest

from src.optimization.objective import (
    OptimizationProblem,
    objective_value,
    is_feasible,
)

from src.optimization.greedy import solve_greedy


def make_problem(
    risks,
    durations,
    budget,
    similarity=None,
    penalty=0.1,
):
    n = len(risks)

    if similarity is None:
        similarity = np.zeros((n, n))

    return OptimizationProblem(
        test_ids=[
            f"test_{i}"
            for i in range(n)
        ],
        risks=np.array(risks),
        durations=np.array(durations),
        similarity=np.array(similarity),
        budget=budget,
        redundancy_penalty=penalty,
    )


def test_feasible_solution():
    problem = make_problem(
        [0.9, 0.7, 0.4],
        [10, 20, 30],
        30,
    )

    selection, result, trace = solve_greedy(
        problem
    )

    assert is_feasible(
        problem,
        selection,
    )

    assert result["feasible"]


def test_binary_selection():
    problem = make_problem(
        [0.9, 0.8, 0.7],
        [10, 10, 10],
        20,
    )

    selection, _, _ = solve_greedy(
        problem
    )

    assert np.isin(
        selection,
        [0, 1],
    ).all()


def test_zero_budget():
    problem = make_problem(
        [0.9, 0.8],
        [10, 20],
        0,
    )

    selection, result, _ = solve_greedy(
        problem
    )

    assert selection.sum() == 0

    assert result["feasible"]


def test_no_feasible_candidates():
    problem = make_problem(
        [0.9, 0.8],
        [100, 200],
        50,
    )

    selection, _, _ = solve_greedy(
        problem
    )

    assert selection.sum() == 0


def test_all_candidates_selected():
    problem = make_problem(
        [0.9, 0.8, 0.7],
        [10, 10, 10],
        100,
    )

    selection, _, _ = solve_greedy(
        problem
    )

    assert selection.sum() == 3


def test_priority_selection():
    problem = make_problem(
        [0.9, 0.6, 0.5],
        [30, 5, 10],
        15,
    )

    selection, _, trace = solve_greedy(
        problem
    )

    assert trace[0]["test_id"] == "test_1"

    assert selection.tolist() == [0, 1, 1]


def test_redundancy_penalty():
    problem = make_problem(
        [0.9, 0.8, 0.7],
        [10, 10, 10],
        20,
        similarity=[
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        penalty=0.5,
    )

    selection, _, _ = solve_greedy(
        problem
    )

    assert selection.tolist() == [1, 0, 1]


def test_negative_marginal_gain():
    problem = make_problem(
        [0.9, 0.2],
        [10, 10],
        20,
        similarity=[
            [0, 1],
            [1, 0],
        ],
        penalty=0.5,
    )

    selection, _, _ = solve_greedy(
        problem
    )

    assert selection.tolist() == [1, 0]


def test_singleton_fallback():
    problem = make_problem(
        [0.4, 0.9],
        [4, 10],
        10,
    )

    selection, result, _ = solve_greedy(
        problem
    )

    assert selection.tolist() == [0, 1]

    assert result[
        "solution_type"
    ] == "best_singleton"


def test_objective_consistency():
    problem = make_problem(
        [0.9, 0.8, 0.7],
        [10, 10, 10],
        20,
    )

    selection, result, _ = solve_greedy(
        problem
    )

    assert result[
        "objective_value"
    ] == pytest.approx(
        objective_value(
            problem,
            selection,
        )
    )


def test_objective_progress():
    problem = make_problem(
        [0.9, 0.8, 0.7],
        [10, 10, 10],
        30,
    )

    _, _, trace = solve_greedy(
        problem
    )

    values = [
        item["objective_value"]
        for item in trace
    ]

    assert all(
        second >= first
        for first, second in zip(
            values,
            values[1:],
        )
    )


def test_deterministic_solution():
    problem = make_problem(
        [0.8, 0.8, 0.8],
        [10, 10, 10],
        20,
    )

    first, _, _ = solve_greedy(
        problem
    )

    second, _, _ = solve_greedy(
        problem
    )

    assert np.array_equal(
        first,
        second,
    )