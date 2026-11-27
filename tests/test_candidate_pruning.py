import numpy as np
import pytest

from src.quantum.candidate_pruning import (
    find_retained_indices,
    prune_qubo_matrix,
    restore_original_state,
    solve_reduced_exact,
)


def test_candidate_pruning():
    retained, removed = (
        find_retained_indices(
            [2, 5, 30, 10],
            20,
        )
    )

    assert retained.tolist() == [
        0, 1, 3
    ]

    assert removed.tolist() == [2]


def test_equal_budget_candidate():
    retained, removed = (
        find_retained_indices(
            [5, 10, 15],
            10,
        )
    )

    assert retained.tolist() == [0, 1]

    assert removed.tolist() == [2]


def test_no_removal():
    retained, removed = (
        find_retained_indices(
            [1, 2, 3],
            10,
        )
    )

    assert len(retained) == 3

    assert len(removed) == 0


def test_all_removed():
    retained, removed = (
        find_retained_indices(
            [10, 20, 30],
            5,
        )
    )

    assert len(retained) == 0

    assert len(removed) == 3


def test_invalid_duration():
    with pytest.raises(ValueError):
        find_retained_indices(
            [1, -2, 3],
            10,
        )


def test_invalid_budget():
    with pytest.raises(ValueError):
        find_retained_indices(
            [1, 2, 3],
            -1,
        )


def test_matrix_pruning():
    matrix = np.arange(
        25,
        dtype=float,
    ).reshape(
        5, 5
    )

    reduced, variables = (
        prune_qubo_matrix(
            matrix,
            [0, 2],
            3,
        )
    )

    assert reduced.shape == (4, 4)

    assert variables.tolist() == [
        0, 2, 3, 4
    ]


def test_restore_original_state():
    restored = restore_original_state(
        [1, 0, 1, 1],
        [0, 2, 3, 4],
        5,
    )

    assert restored.tolist() == [
        1, 0, 0, 1, 1
    ]


def test_energy_preservation():
    matrix = np.array(
        [
            [-2.0, 3.0, 1.0],
            [0.0, 10.0, 4.0],
            [0.0, 0.0, -1.0],
        ]
    )

    reduced, variables = (
        prune_qubo_matrix(
            matrix,
            [0, 2],
            3,
        )
    )

    state = np.array(
        [1, 1]
    )

    restored = restore_original_state(
        state,
        variables,
        3,
    )

    assert (
        state @ reduced @ state
    ) == pytest.approx(
        restored @ matrix @ restored
    )


def test_exact_solver():
    matrix = np.array(
        [
            [-2.0, 0.0],
            [0.0, -1.0],
        ]
    )

    result = solve_reduced_exact(
        matrix,
        0.0,
    )

    assert result[
        "energy"
    ] == pytest.approx(-3.0)

    assert result[
        "state"
    ].tolist() == [1, 1]


def test_exact_state_count():
    result = solve_reduced_exact(
        np.eye(4),
        0.0,
    )

    assert result[
        "states_evaluated"
    ] == 16


def test_optimality_preservation():
    matrix = np.array(
        [
            [-2.0, 0.0, 1.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, -3.0],
        ]
    )

    original = solve_reduced_exact(
        matrix,
        0.0,
    )

    reduced, _ = prune_qubo_matrix(
        matrix,
        [0, 2],
        3,
    )

    pruned = solve_reduced_exact(
        reduced,
        0.0,
    )

    assert original[
        "energy"
    ] == pytest.approx(
        pruned["energy"]
    )