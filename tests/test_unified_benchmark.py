import numpy as np
import pandas as pd
import pytest

from src.optimization.objective import (
    OptimizationProblem,
)

from src.evaluation.unified_benchmark import (
    calculate_ratio,
    create_discrete_problem,
    solve_exact_selection,
    validate_solution,
    aggregate_results,
)


@pytest.fixture
def problem():
    return OptimizationProblem(
        test_ids=["A", "B", "C"],
        risks=np.array([0.8, 0.6, 0.4]),
        durations=np.array([4.1, 8.2, 3.0]),
        similarity=np.zeros((3, 3)),
        budget=15,
        redundancy_penalty=0.1,
    )


@pytest.fixture
def report():
    return {
        "duration_units": [1, 2, 1],
        "duration_unit_seconds": 5,
        "budget_units": 3,
    }


def test_perfect_ratio():
    assert calculate_ratio(
        2.0,
        2.0,
    ) == 1.0


def test_partial_ratio():
    assert calculate_ratio(
        1.0,
        2.0,
    ) == 0.5


def test_missing_objective():
    assert calculate_ratio(
        None,
        2.0,
    ) is None


def test_zero_optimum():
    assert calculate_ratio(
        0.0,
        0.0,
    ) == 1.0


def test_invalid_ratio():
    with pytest.raises(RuntimeError):
        calculate_ratio(
            3.0,
            2.0,
        )


def test_discrete_durations(
    problem,
    report,
):
    discrete = create_discrete_problem(
        problem,
        report,
    )

    assert discrete.durations.tolist() == [
        5.0,
        10.0,
        5.0,
    ]


def test_discrete_budget(
    problem,
    report,
):
    discrete = create_discrete_problem(
        problem,
        report,
    )

    assert discrete.budget == 15


def test_invalid_discretization(
    problem,
    report,
):
    report["duration_units"] = [
        1,
        1,
        1,
    ]

    with pytest.raises(RuntimeError):
        create_discrete_problem(
            problem,
            report,
        )


def test_exact_enumeration(
    problem,
    report,
):
    discrete = create_discrete_problem(
        problem,
        report,
    )

    selection, result = solve_exact_selection(
        discrete
    )

    assert selection.tolist() == [
        1,
        1,
        0,
    ]

    assert result[
        "objective_value"
    ] == pytest.approx(1.4)


def test_exact_state_count(problem):
    selection, result = solve_exact_selection(
        problem
    )

    assert result[
        "states_evaluated"
    ] == 8


def test_valid_solution(
    problem,
    report,
):
    discrete = create_discrete_problem(
        problem,
        report,
    )

    result = validate_solution(
        problem,
        discrete,
        np.array([1, 1, 0]),
        1.4,
    )

    assert result["feasible"]

    assert result["optimal"]


def test_invalid_solution(
    problem,
    report,
):
    discrete = create_discrete_problem(
        problem,
        report,
    )

    with pytest.raises(RuntimeError):
        validate_solution(
            problem,
            discrete,
            np.array([1, 1, 1]),
            1.4,
        )


def test_aggregation():
    dataframe = pd.DataFrame(
        {
            "algorithm": ["greedy", "greedy"],
            "repetition": [1, 2],
            "objective": [1.0, 1.0],
            "approximation_ratio": [0.5, 0.5],
            "runtime_seconds": [0.1, 0.2],
            "feasible": [True, True],
            "optimal": [False, False],
        }
    )

    result = aggregate_results(
        dataframe
    )

    assert len(result) == 1

    assert result.iloc[0][
        "mean_runtime_seconds"
    ] == pytest.approx(0.15)

    assert result.iloc[0][
        "feasibility_rate"
    ] == 1.0

def test_missing_duration_unit(problem, report):
    report.pop("duration_unit_seconds")

    discrete = create_discrete_problem(
        problem,
        report,
    )

    assert discrete.durations.tolist() == [
        5.0,
        10.0,
        5.0,
    ]


def test_invalid_budget_units(problem, report):
    report["budget_units"] = 4

    with pytest.raises(RuntimeError):
        create_discrete_problem(
            problem,
            report,
        )


def test_invalid_duration_unit(problem, report):
    report["duration_unit_seconds"] = 0

    with pytest.raises(ValueError):
        create_discrete_problem(
            problem,
            report,
        )