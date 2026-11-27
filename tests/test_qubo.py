import itertools

import numpy as np
import pytest

from src.optimization.objective import (
    OptimizationProblem,
    objective_value,
)

from src.quantum.qubo import (
    create_qubo,
    create_slack_weights,
    discretize_durations,
    calculate_penalty,
    qubo_energy,
    direct_energy,
    decode_state,
)

from src.quantum.exact_solver import (
    solve_exact_qubo,
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


def test_duration_discretization():
    durations, budget = discretize_durations(
        [4.1, 5.0, 5.1],
        12.0,
        5.0,
    )

    assert durations.tolist() == [
        1, 1, 2
    ]

    assert budget == 2


def test_slack_weights():
    weights = create_slack_weights(24)

    assert weights.tolist() == [
        1, 2, 4, 8, 9
    ]

    assert weights.sum() == 24


def test_slack_representation():
    weights = create_slack_weights(24)

    values = set()

    for bits in itertools.product(
        [0, 1],
        repeat=len(weights),
    ):
        values.add(
            int(
                weights @ np.array(bits)
            )
        )

    assert values == set(
        range(25)
    )


def test_zero_budget_slack():
    assert len(
        create_slack_weights(0)
    ) == 0


def test_penalty(problem):
    penalty = calculate_penalty(
        problem
    )

    assert penalty > problem.risks.sum()


def test_matrix_shape(problem):
    qubo = create_qubo(
        problem
    )

    assert qubo.matrix.shape == (
        qubo.size,
        qubo.size,
    )


def test_upper_triangular_matrix(problem):
    qubo = create_qubo(
        problem
    )

    assert np.allclose(
        np.tril(
            qubo.matrix,
            -1,
        ),
        0,
    )


def test_matrix_energy_equivalence(problem):
    qubo = create_qubo(
        problem
    )

    for bits in itertools.product(
        [0, 1],
        repeat=qubo.size,
    ):
        assert qubo_energy(
            qubo,
            bits,
        ) == pytest.approx(
            direct_energy(
                qubo,
                bits,
            ),
            abs=1e-8,
        )


def test_invalid_state(problem):
    qubo = create_qubo(
        problem
    )

    state = np.zeros(
        qubo.size
    )

    state[0] = 2

    with pytest.raises(ValueError):
        qubo_energy(
            qubo,
            state,
        )


def test_invalid_state_size(problem):
    qubo = create_qubo(
        problem
    )

    with pytest.raises(ValueError):
        qubo_energy(
            qubo,
            [0, 1],
        )


def test_insufficient_penalty(problem):
    with pytest.raises(ValueError):
        create_qubo(
            problem,
            penalty=0.1,
        )


def test_exact_solution(problem):
    qubo = create_qubo(
        problem
    )

    selection, result = solve_exact_qubo(
        qubo
    )

    assert selection.tolist() == [
        1, 1, 0
    ]

    assert result["objective_value"] == pytest.approx(
        1.55
    )

    assert result["feasible"]


def test_exact_energy(problem):
    qubo = create_qubo(
        problem
    )

    _, result = solve_exact_qubo(
        qubo
    )

    assert result["energy"] == pytest.approx(
        -1.55
    )


def test_exact_constraint(problem):
    qubo = create_qubo(
        problem
    )

    _, result = solve_exact_qubo(
        qubo
    )

    decoded = decode_state(
        qubo,
        result["binary_state"],
    )

    assert decoded["constraint_residual"] == 0


def test_exact_state_count(problem):
    qubo = create_qubo(
        problem
    )

    _, result = solve_exact_qubo(
        qubo
    )

    assert result["states_evaluated"] == (
        2 ** qubo.size
    )


def test_zero_budget(problem):
    problem.budget = 0.0

    qubo = create_qubo(
        problem
    )

    selection, result = solve_exact_qubo(
        qubo
    )

    assert selection.sum() == 0

    assert result["feasible"]


def test_conservative_duration_rounding():
    problem = OptimizationProblem(
        test_ids=["A", "B"],
        risks=np.array(
            [0.9, 0.8]
        ),
        durations=np.array(
            [5.1, 5.1]
        ),
        similarity=np.zeros((2, 2)),
        budget=15.0,
        redundancy_penalty=0.1,
    )

    qubo = create_qubo(
        problem,
        unit_seconds=5.0,
    )

    selection, result = solve_exact_qubo(
        qubo
    )

    assert selection.sum() == 1

    assert result["feasible"]


def test_exact_against_discrete_brute_force(problem):
    qubo = create_qubo(
        problem
    )

    best = float("-inf")

    for bits in itertools.product(
        [0, 1],
        repeat=problem.size,
    ):
        selection = np.array(bits)

        if (
            qubo.duration_units @ selection
            <= qubo.budget_units
        ):
            best = max(
                best,
                objective_value(
                    problem,
                    selection,
                ),
            )

    _, result = solve_exact_qubo(
        qubo
    )

    assert result["objective_value"] == pytest.approx(
        best
    )