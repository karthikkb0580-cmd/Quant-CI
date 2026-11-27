import numpy as np
import pandas as pd
import pytest

from src.optimization.random_sampling import (
    generate_random_states,
    generate_random_counts,
    safe_mean,
    create_comparison,
)


def test_random_state_shape():
    states = generate_random_states(
        14,
        1024,
        42,
    )

    assert states.shape == (
        1024,
        14,
    )


def test_binary_states():
    states = generate_random_states(
        14,
        1024,
        42,
    )

    assert np.isin(
        states,
        [0, 1],
    ).all()


def test_reproducibility():
    first = generate_random_states(
        14,
        1024,
        42,
    )

    second = generate_random_states(
        14,
        1024,
        42,
    )

    assert np.array_equal(
        first,
        second,
    )


def test_different_seeds():
    first = generate_random_states(
        14,
        1024,
        42,
    )

    second = generate_random_states(
        14,
        1024,
        123,
    )

    assert not np.array_equal(
        first,
        second,
    )


def test_invalid_variables():
    with pytest.raises(ValueError):
        generate_random_states(
            0,
            1024,
            42,
        )


def test_invalid_shots():
    with pytest.raises(ValueError):
        generate_random_states(
            14,
            0,
            42,
        )


def test_count_total():
    counts = generate_random_counts(
        14,
        4096,
        42,
    )

    assert sum(
        counts.values()
    ) == 4096


def test_bitstring_lengths():
    counts = generate_random_counts(
        14,
        1024,
        42,
    )

    assert all(
        len(bitstring) == 14
        for bitstring in counts
    )


def test_bitstring_values():
    counts = generate_random_counts(
        14,
        1024,
        42,
    )

    assert all(
        set(bitstring).issubset(
            {"0", "1"}
        )
        for bitstring in counts
    )


def test_uniform_distribution():
    states = generate_random_states(
        14,
        10000,
        42,
    )

    means = states.mean(
        axis=0
    )

    assert np.all(
        np.abs(means - 0.5)
        < 0.03
    )


def test_safe_mean():
    values = pd.Series(
        [0.5, 1.0]
    )

    assert safe_mean(
        values
    ) == pytest.approx(0.75)


def test_missing_mean():
    values = pd.Series(
        [np.nan, np.nan]
    )

    assert safe_mean(
        values
    ) is None


@pytest.fixture
def comparison_data():
    random_data = pd.DataFrame(
        {
            "experiment": ["E1"],
            "seed": [42],
            "shots": [1024],
            "raw_approximation_ratio": [0.7],
            "hybrid_approximation_ratio": [1.0],
            "raw_feasibility_rate": [0.1],
            "hybrid_feasibility_rate": [0.8],
            "raw_optimal_shots": [0],
            "hybrid_optimal_shots": [2],
            "raw_total_seconds": [0.1],
            "hybrid_total_seconds": [0.2],
        }
    )

    qaoa_data = pd.DataFrame(
        {
            "experiment": ["E1"],
            "seed": [42],
            "shots": [1024],
            "approximation_ratio": [0.9],
            "feasible_measurement_rate": [0.2],
            "optimal_shots": [0],
            "total_seconds": [2.0],
        }
    )

    hybrid_data = pd.DataFrame(
        {
            "experiment": ["E1"],
            "seed": [42],
            "hybrid_approximation_ratio": [1.0],
            "hybrid_feasibility_rate": [0.9],
            "hybrid_optimal_shots": [3],
            "hybrid_total_seconds": [2.2],
        }
    )

    return (
        random_data,
        qaoa_data,
        hybrid_data,
    )


def test_comparison(comparison_data):
    result = create_comparison(
        *comparison_data
    )

    assert len(result) == 1

    assert result.iloc[0][
        "raw_qaoa_advantage"
    ] == pytest.approx(0.2)

    assert result.iloc[0][
        "hybrid_qaoa_advantage"
    ] == pytest.approx(0.0)


def test_mismatched_shots(comparison_data):
    random_data, qaoa_data, hybrid_data = (
        comparison_data
    )

    qaoa_data["shots"] = 4096

    with pytest.raises(RuntimeError):
        create_comparison(
            random_data,
            qaoa_data,
            hybrid_data,
        )