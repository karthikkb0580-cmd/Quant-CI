import numpy as np
import pytest

from src.optimization.objective import (
    OptimizationProblem,
    compute_similarity,
    build_similarity_matrix,
    calculate_risk,
    calculate_duration,
    calculate_redundancy,
    objective_value,
    is_feasible,
    marginal_gain,
    evaluate_selection,
)


@pytest.fixture
def problem():
    return OptimizationProblem(
        test_ids=["A", "B", "C"],
        risks=np.array(
            [0.9, 0.7, 0.4]
        ),
        durations=np.array(
            [10.0, 20.0, 30.0]
        ),
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


def test_problem_size(problem):
    assert problem.size == 3


def test_empty_selection(problem):
    assert objective_value(
        problem,
        [0, 0, 0],
    ) == 0


def test_risk_calculation(problem):
    assert calculate_risk(
        problem,
        [1, 1, 0],
    ) == pytest.approx(1.6)


def test_duration_calculation(problem):
    assert calculate_duration(
        problem,
        [1, 1, 0],
    ) == pytest.approx(30.0)


def test_redundancy_calculation(problem):
    assert calculate_redundancy(
        problem,
        [1, 1, 0],
    ) == pytest.approx(0.5)


def test_objective_value(problem):
    assert objective_value(
        problem,
        [1, 1, 0],
    ) == pytest.approx(1.55)


def test_feasible_selection(problem):
    assert is_feasible(
        problem,
        [1, 1, 0],
    )


def test_infeasible_selection(problem):
    assert not is_feasible(
        problem,
        [1, 1, 1],
    )


def test_marginal_gain(problem):
    assert marginal_gain(
        problem,
        [1, 0, 0],
        1,
    ) == pytest.approx(0.65)


def test_invalid_selection(problem):
    with pytest.raises(ValueError):
        objective_value(
            problem,
            [1, 2, 0],
        )


def test_invalid_selection_size(problem):
    with pytest.raises(ValueError):
        objective_value(
            problem,
            [1, 0],
        )


def test_identical_similarity():
    assert compute_similarity(
        "BrokerConnectionTest",
        "BrokerConnectionTest",
    ) == pytest.approx(1.0)


def test_unrelated_similarity():
    assert compute_similarity(
        "AuthenticationTest",
        "MessageQueueTest",
    ) == 0.0


def test_similarity_matrix():
    matrix = build_similarity_matrix(
        [
            "BrokerConnectionTest",
            "BrokerConnectionFactoryTest",
            "AuthenticationTest",
        ]
    )

    assert matrix.shape == (3, 3)

    assert np.allclose(
        matrix,
        matrix.T,
    )

    assert np.allclose(
        np.diag(matrix),
        0,
    )

    assert matrix[0, 1] > 0


def test_selection_evaluation(problem):
    result = evaluate_selection(
        problem,
        [1, 1, 0],
    )

    assert result["selected_count"] == 2

    assert result["feasible"]

    assert result[
        "objective_value"
    ] == pytest.approx(1.55)

    assert result[
        "selected_tests"
    ] == ["A", "B"]


def test_invalid_budget(problem):
    with pytest.raises(ValueError):
        OptimizationProblem(
            test_ids=problem.test_ids,
            risks=problem.risks,
            durations=problem.durations,
            similarity=problem.similarity,
            budget=-1,
            redundancy_penalty=0.1,
        )