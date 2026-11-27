import itertools

import numpy as np
import pytest

from src.optimization.objective import (
    OptimizationProblem,
    objective_value,
    is_feasible,
)

from src.optimization.ortools_solver import (
    solve_ortools,
    scale_durations,
    scale_objective,
)


@pytest.fixture
def problem():
    return OptimizationProblem(
        test_ids=["A", "B", "C"],
        risks=np.array([0.9, 0.7, 0.4]),
        durations=np.array([10.0, 20.0, 30.0]),
        similarity=np.array(
            [
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.2],
                [0.0, 0.2, 0.0],
            ]
        ),
        budget=30.0,
        redundancy_penalty=0.1,
    )


def test_duration_scaling(problem):
    durations, budget = scale_durations(
        problem
    )

    assert durations.tolist() == [
        10000,
        20000,
        30000,
    ]

    assert budget == 30000


def test_objective_scaling(problem):
    risks, penalties = scale_objective(
        problem
    )

    assert risks.tolist() == [
        900000,
        700000,
        400000,
    ]

    assert penalties[0, 1] == 50000


def test_feasible_solution(problem):
    selection, result = solve_ortools(
        problem
    )

    assert result["feasible"]

    assert is_feasible(
        problem,
        selection,
    )


def test_binary_selection(problem):
    selection, _ = solve_ortools(
        problem
    )

    assert set(
        selection.tolist()
    ).issubset({0, 1})


def test_known_optimum(problem):
    selection, result = solve_ortools(
        problem
    )

    assert selection.tolist() == [
        1,
        1,
        0,
    ]

    assert result[
        "objective_value"
    ] == pytest.approx(1.55)


def test_zero_budget(problem):
    problem.budget = 0.0

    selection, result = solve_ortools(
        problem
    )

    assert selection.sum() == 0

    assert result["feasible"]


def test_no_feasible_candidates(problem):
    problem.budget = 1.0

    selection, result = solve_ortools(
        problem
    )

    assert selection.sum() == 0

    assert result["feasible"]


def test_redundancy_penalty(problem):
    problem.redundancy_penalty = 2.0

    selection, result = solve_ortools(
        problem
    )

    assert selection.tolist() == [
        1,
        0,
        0,
    ]

    assert result[
        "objective_value"
    ] == pytest.approx(0.9)


def test_brute_force_comparison(problem):
    best = float("-inf")

    for values in itertools.product(
        [0, 1],
        repeat=problem.size,
    ):
        selection = np.array(
            values
        )

        if is_feasible(
            problem,
            selection,
        ):
            best = max(
                best,
                objective_value(
                    problem,
                    selection,
                ),
            )

    _, result = solve_ortools(
        problem
    )

    assert result[
        "objective_value"
    ] == pytest.approx(best)


def test_optimality_status(problem):
    _, result = solve_ortools(
        problem
    )

    assert result[
        "solver_status"
    ] == "OPTIMAL"

    assert result[
        "optimal_scaled_model"
    ]

    assert result[
        "scaled_optimality_gap"
    ] == pytest.approx(0.0)


def test_invalid_time_limit(problem):
    with pytest.raises(ValueError):
        solve_ortools(
            problem,
            time_limit=0,
        )


def test_fractional_budget_safety():
    problem = OptimizationProblem(
        test_ids=["A", "B"],
        risks=np.array([0.9, 0.8]),
        durations=np.array(
            [0.5004, 0.5004]
        ),
        similarity=np.zeros((2, 2)),
        budget=1.0005,
        redundancy_penalty=0.1,
    )

    selection, result = solve_ortools(
        problem
    )

    assert selection.sum() == 1

    assert result["feasible"]